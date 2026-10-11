# Ingest a benchmark and get its runtimes ready

Account for every submitted task. Each must have a ready runtime or a recorded failure you can
investigate. Registration alone does not prove readiness, and a successful build does not prove
that the solver or grader works.

For your first project, follow [T Factory](t_factory/README.md), then the
[GSM8K workflow](gsm8k/README.md). This reference covers ingestion and recovery for your own benchmark.

> Availability: `list_inputs()` and `retrieve_build_logs()` require the upcoming inspection
> API/SDK release. Automatic context packaging is a separate change whose released-client
> verification is also pending. The existing submission and image-status
> methods remain usable without them.

## 1. Submit your source and keep the operation ID

Set `TRAJECTORY_API_KEY` and [choose or create an agent](../README.md#choose-or-create-an-agent).
Managed uploads use authorization supplied by Trajectory; customers do not need GCS or sandbox
provider credentials. Supply runtime service secrets through `TaskSpec.env_vars` with `SecretRef`.

Preserve the benchmark's task IDs, splits, commands, source files and actor/private-grader boundary.
See the [runtime integration reference](../README.md#package-a-benchmark-runtime) for harness setup.

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push

client = Client()
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=[TaskSpec(
        name="source-task-001",
        split="test",
        run_command="python /app/harness.py",
    )],
)
operation = start_push(
    client, benchmark,
    agent_name="YOUR_AGENT_NAME",
    root=Path("my-benchmark"),
    build_images=True,
    idempotency_key="YOUR_SAVED_SUBMISSION_KEY",
)
print(operation.id)
status = operation.refresh()
print(status.status, status.stage, status.registered_tasks, status.total_tasks,
      status.ready_runtimes, status.total_runtimes, status.last_error)
```

The context is `my-benchmark/runtime/`, filtered by `.dockerignore` at the benchmark root. The
SDK always includes the selected Dockerfile; it does not infer required files from `COPY`.
With automatic packaging available, it uploads each selected context as an archive without
changing the Dockerfile or requiring you to create archives. Different contexts remain separate.
Archives stay in local temporary storage until submission finishes. Allow free space for all
selected contexts plus archive overhead, without assuming compression savings. Set `TMPDIR`
before starting Python to use a larger volume; temporary archives are removed on exit or failure.

Save the idempotency key before submission and keep the local inputs through finalization.
Retry unchanged input with the same key. Changed source, tasks or settings require a new key.

`start_push()` returns after upload and acceptance. Registration and builds continue on the server.
The progress display reports client preparation and transfer, not runtime readiness. A client timeout
does not cancel server work: reconnect with `get_operation(client, operation_id)` from `trajectory.lib`.
`push()` is the convenience that also waits for the requested builds.

## 2. Account for every submitted input

Use the operation ID even if registration failed before creating the benchmark:

```python
operation_id = "iop_YOUR_OPERATION"
page_options = {"limit": 100}
while True:
    page = client.benchmarks.ingestion.list_inputs(operation_id, **page_options)
    for item in page.items:
        print(item.part_path, item.task_index, item.name, item.status,
              item.task_id, item.runtime_id, item.runtime, item.failure)
    if page.unavailable_parts:
        print("Unresolved input parts:", page.unavailable_parts)
    if page.next_cursor is None:
        break
    page_options["cursor"] = page.next_cursor
```

| Input status | Meaning |
| --- | --- |
| `pending` | Registration is still in progress. |
| `registered` | A task was registered; inspect its runtime separately. |
| `rejected` | The input has a recorded task failure. |
| `not_registered` | The operation ended without a registration receipt. |

`(operation_id, part_path, task_index)` identifies an input before it receives a task ID.
Follow the cursor even when a page is empty; unreadable or unverified parts appear in
`unavailable_parts`. Keep your original task count because those parts may hide task identities.
Refresh from the first page to observe progress.

Existing receipt and failure pages remain available through `get_operation(client, operation_id)`
and `operation.pages("result")`, `operation.pages("runtime")`, or `operation.pages("failure")`.
A shared `runtime_id` identifies tasks affected by the same build failure.

## 3. Read the failed build before retrying

```python
complete = False
with client.benchmarks.ingestion.retrieve_build_logs(
    operation_id, runtime_id="rt_YOUR_RUNTIME",
) as logs:
    for chunk in logs:
        print(chunk.message, end="")
        complete = chunk.complete
if not complete:
    raise RuntimeError("Build-log read was interrupted; retry the read")
```

Failure listings retain a short error summary. This read streams the available logs for the
operation's recorded build, including after a retry creates another image. It does not rebuild
or clip output to an entry or character limit. Modal is the first supported provider; its
adapter retrieves all available build layers.

`complete=True` means the provider finished returning available logs, not that the build
succeeded or that it retains every historical log. Empty output can change on a later read;
provider-read failures raise an SDK error. A build without a recorded image or a
recoverable provider identity returns HTTP 409; unsupported providers return HTTP 501. Registered secrets are masked; other
credentials printed by build code may remain visible to authorized users in your organization.

## 4. Repair the cause and retry affected work

| Failure | Action |
| --- | --- |
| Interrupted upload; inputs unchanged | Repeat the submission with its saved key. |
| Invalid source, Dockerfile, dependencies or task settings | Correct the cause and submit the intended task set with a new key. |
| Transient image-build failure; stored source unchanged | After ingestion settles, call `wait_for_benchmark_images(client, bench_id)` from `trajectory.lib` to request builds and wait for them. |

The helper calls `images.build()` for the benchmark's eligible runtimes, not a selected task. Ready runtimes
and identified in-flight builds are skipped. Diagnose shared failures once; healthy tasks need not
wait for unrelated repairs. Keep original failures and recovered outcomes separate.
The wait helper raises `BenchmarkImageBuildError` if builds fail; inspect its `partial_result`
for each task's outcome. A timeout leaves server work running.

For corrected source, reuse the agent and benchmark name, and save the new operation and benchmark
IDs. A new version contains only the submitted tasks: a subset does not inherit the rest of the old
version. Preserve the original denominator when reporting readiness.

## 5. Confirm readiness, then test execution

```python
from trajectory.lib import get_operation
from trajectory.lib.benchmark_operations import BenchmarkOperationError

try:
    result = get_operation(client, operation_id).result()
except BenchmarkOperationError as error:
    result = error.partial_result
    for failure in result.failures():
        print(failure)

if result.status.registered_tasks == 0:
    raise RuntimeError("No registered tasks; inspect the input failures before continuing")
bench_id = result.status.bench_id
images = client.benchmarks.images.list(bench_id)
for image in images.images:
    print(image.task_id, image.build_status, image.provider_ref, image.failure_message)
ready_task_ids = [image.task_id for image in images.images if image.build_status == "ready"]
print("Ready tasks:", ready_task_ids)
```

`result()` waits for this submission's registration and requested builds. Its default 30-minute
timeout leaves server work running; reconnect with the saved operation ID to continue waiting.
Separately requested build retries use the wait helper in step 4.

Match these results to the original input inventory, retaining unregistered tasks and failures.
Use ready task IDs for targeted evaluation; failed tasks need not block them. The
[task-diagnostic walkthrough](task_diagnostics.md) checks startup, imports, credentials and grading.
Its benchmark-wide diagnostic samples tasks automatically, so use it after the benchmark's
registered runtimes are ready; it does not filter out failed builds.

## Managed context limits

These limits apply to each selected context, not the benchmark's combined size. The archive and
path limits apply when automatic packaging is available.

| Dimension | Maximum |
| --- | --- |
| Uploaded archive or expanded regular-file bytes | 3 GiB (3,221,225,472 bytes) each |
| Files and symbolic links | 4,096 |
| Context path depth, including filename | 128 components |
| Archive entries and restored paths, including directories | 81,920 |

Use ordinary files for Modal builds; symbolic links fail with the offending path. Exclude unrelated
files or supply dependencies through the base image when a context exceeds its limits. Runloop
limits the selected Dockerfile to 16 MiB. Worker-capacity errors report `worker_bytes` and
`allowed_bytes`; retain the error's dimension and values when investigating.
