# Ingest a benchmark and get its runtimes ready

Choose how to supply the runtime, submit the original task inventory, then use the ingestion
records to fix failures. You are done with ingestion when every submitted task is accounted
for and its required runtime is ready, or its failure is recorded with a cause and next action.
Runtime readiness does not prove that the task's solver or grader works.

Preserve the original task inventory, files and commands. See the
[runtime integration reference](../README.md#package-a-benchmark-runtime) for harness setup.

> Release note: automatic context archives and expanded ingestion history target the upcoming
> matching API/SDK release. Released-client verification is still pending. Do not infer
> availability from installing an older published SDK.

## 1. Choose the runtime path

Both paths produce the same task/runtime records and use the same inspection and repair loop.
Keep the benchmark's native files, commands, task IDs, splits and actor/private-grader boundary.

| You have | Use | Trajectory does |
| --- | --- | --- |
| An ordinary Dockerfile and local files | `DockerfileBuild("runtime/Dockerfile")` | Uploads the selected context privately and builds the runtime. |
| A public image you already build and publish | `ImageRef("registry.example.com/runtime@sha256:...")` | Pulls the pinned image and adds the execution bootstrap. |
| A ready runtime in your organization | `RuntimeRef("rt_YOUR_RUNTIME")` | Reuses that runtime. This is not a fresh build or registry-pull test. |

### Managed build: keep your Dockerfile and files

```python
from trajectory.lib import DockerfileBuild

runtime = DockerfileBuild("runtime/Dockerfile")
```

With `root=Path("my-benchmark")`, the context is `my-benchmark/runtime/`. The SDK selects
files beneath that Dockerfile's directory, filtered by `.dockerignore` at the benchmark root.
It always includes the selected Dockerfile; it does not infer the context from `COPY` statements.
A root-level Dockerfile therefore selects the checkout unless you exclude files.

The SDK packages each selected context automatically and uses the existing upload session.
You do not create an archive, change `COPY` commands or upload each file yourself. The worker
verifies and restores the context before building. The archive format preserves file modes and
safe relative symlinks; escaping links, links to excluded files and unsupported file types fail
validation. Hardlinked source files are packaged as separate regular files. Modal managed
builds reject symbolic links before build, with the offending path in the runtime error.
Use ordinary files or a prebuilt image when the provider cannot preserve required source behavior.

Put shared code in each context or provide it through your base image. Packaging does not
deduplicate different contexts into shared cloud layers.

Runtime identity follows the selected files, permissions, links and Dockerfile; the upload
checksum separately verifies transferred bytes. Restored timestamps use the Unix epoch.
Changing compression alone does not change source identity. Older SDKs can still upload loose
files, though changing representations may require one new build.

In the matching release, each selected managed context has these limits:

| Dimension | Maximum |
| --- | --- |
| Uploaded archive size | 3 GiB (3,221,225,472 bytes) |
| Sum of expanded regular-file sizes | 3 GiB (3,221,225,472 bytes) |
| Prepared build context | 3 GiB (3,221,225,472 bytes) |
| Context path depth | 128 components, including the filename |
| Files and symbolic links | 65,536 |
| Archive entries and restored paths, including directories | 81,920 |

The prepared context includes Trajectory's bootstrap and any required Dockerfile copy, so leave
room below 3 GiB when selecting source files. A preparation-size error reports `prepared_bytes` and
`allowed_bytes`. Runloop also limits the selected Dockerfile to 16 MiB; put large assets in ordinary
context files.

These limits apply independently to each context, not to the benchmark's combined size or
prebuilt image layers. If required inputs exceed them, use a focused context or build and publish
a prebuilt image. The provider's image constraints still apply to that image.

### Public prebuilt image: build and publish once

```python
from trajectory.lib import ImageRef

runtime = ImageRef("registry.example.com/runtime@sha256:YOUR_DIGEST")
```

Use the image digest returned by your registry. On the initial Modal path, build for
`linux/amd64`. Include the harness, task files, verifier dependencies and the harness's SDK
installation in the image; the execution bootstrap does not install them. Public image
references require no pull-secret field and keep their existing behavior.

Publish the image in a public registry. Private registry access is outside this release.
Your local or CI credentials authorize building and publishing the image; Trajectory needs no
registry credentials to pull it.

### Authenticate each operation

Set `TRAJECTORY_API_KEY` for the organization receiving the benchmark. `Client()` uses that key
for Trajectory API calls. Trajectory supplies signed upload authorization and its worker/provider
credentials; you do not need cloud-storage credentials for managed context delivery. Declare
runtime service secrets separately through `TaskSpec.env_vars` with `SecretRef`.

## 2. Submit and keep the operation ID

Use the chosen `runtime` with your original tasks. Preserve dataset identifiers in `name` and
set each original `split`; omit platform task IDs for new tasks. Here is a one-task example:

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import start_push

client = Client()
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=runtime,
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

Save a stable idempotency key **before** starting a large submission. Repeating unchanged input
with that key lets the SDK reuse completed objects and resume a large archive from the storage
service's acknowledged offset, including after a client restart. If its session expired, only that
unfinished object restarts. Keep the local inputs until finalization. Changed files, tasks or settings
require a new key. Do not rename the benchmark merely to retry: keep its agent and name so versions stay grouped.

A prebuilt-only submission can use `root=Path(".")`; image layers stay in the registry and task
definitions use the uploader. Include files needed by the harness in the image.
`start_push()` returns after transfer and acceptance; registration/builds continue on the server.
`operation.refresh()` reconnects to that work. A client timeout does not cancel it. After restarting
your client, use `get_operation(client, operation_id)` from `trajectory.lib` before submitting again.
`push()` is the convenience that also waits for requested image builds.

The `progress` callback first reports `phase="preparing"`, then `phase="packaging"` with
`prepared_contexts` and `total_contexts`. During `phase="uploading"`, its counters report completed
or reused objects/bytes, not instantaneous network traffic. Upload counters are zero during
preparation; transfer totals become known when uploading starts. The default terminal display
shows preparation and packaging too, before `start_push()` returns.
Retrieving an operation returns `upload_progress`: declared and checksum-verified object/byte
totals, including task parts. History entries may omit it; unknown progress is `None`, not zero.
Declared bytes do not prove transfer, and verified bytes do not establish registration or readiness.

## 3. Inspect every input and failed runtime

Use ingestion records to inspect work before task registration. An allocated benchmark ID
does not yet imply a registered benchmark.

Find an operation in history, including one that failed before registration:

```python
for entry in client.benchmarks.ingestion.list():
    print(entry.operation_id, entry.bench_id, entry.status)
```

Use `bench_id=...` to restrict that history to one benchmark version. For a selected operation,
account for original inputs even when they have no registered task ID:

```python
operation_id = "iop_YOUR_OPERATION"
page_options = {"limit": 100}
while True:
    page = client.benchmarks.ingestion.list_inputs(operation_id, **page_options)
    for item in page.items:
        print(item.part_path, item.task_index, item.name, item.status,
              item.task_id, item.runtime, item.failure)
    if page.unavailable_parts:
        print("Unresolved input parts:", page.unavailable_parts)
    if page.next_cursor is None:
        break
    page_options["cursor"] = page.next_cursor
```

`pending` means registration has not finished; `rejected` has a recorded task failure;
`not_registered` means the operation ended without a registration receipt. `registered` does
not establish runtime readiness. Use `(operation_id, part_path, task_index)` to identify an input
before it receives a task ID. These reads use retained upload parts and registration receipts;
they do not register or rebuild anything.

Malformed or unverified parts appear in `unavailable_parts`. Follow the cursor even when a page
contains no task rows, and refresh from the first page to observe progress. Keep your original
source count: the platform cannot infer identities from an unreadable part or a task never submitted.

Inspect task results, failures and runtime records through the same paginated interface:

```python
for kind in ("result", "failure", "runtime"):
    after = ""
    while True:
        page = client.benchmarks.ingestion.list_items(
            operation_id, kind=kind, after=after, limit=100,
        )
        for item in page.items:
            print(kind, item.model_dump())
        if page.next_cursor is None:
            break
        after = page.next_cursor
```

A shared-runtime failure can affect many tasks. Use `kind="runtime_task"` and
`runtime_id="rt_YOUR_RUNTIME"` with that pagination loop to identify them. Preserve unaffected
work and the original task denominator while repairing the shared cause.

Read fresh build output from the failed operation/runtime pair before rebuilding:

```python
logs = client.benchmarks.ingestion.build_logs(
    operation_id, runtime_id="rt_YOUR_RUNTIME",
)
print(logs.observed_at, logs.provider_ref, logs.available)
if logs.available:
    print(logs.excerpt)
```

This is a bounded read of the original build, with registered secret values redacted. It returns
up to 6,000 characters. `available=True` means text arrived, not that the provider's final error
has arrived. A later read can reveal the final lines; rebuilding is unnecessary to obtain them.
A timeout or missing historical reference may return `available=False`. Keep that diagnostic
limitation alongside the original failure. Credentials printed by build code outside the known
redaction set may remain visible to your organization's authorized users.

## 4. Fix the cause and retry the affected work

| Observed failure | Next action |
| --- | --- |
| Transfer interrupted; inputs unchanged | Repeat the original submission with the saved key. Reuse completed objects and resume active large-object sessions; an expired session restarts only its unfinished object. |
| Invalid context, changed Dockerfile/dependencies or task settings | Correct the owning source or integration, then submit the intended task set with a new key. |
| Public image cannot be pulled | Check that the digest exists and the registry permits anonymous reads. Correct the reference and submit a new version; retain the original failure. |
| Transient runtime build failure; stored source unchanged | After ingestion settles, call `client.benchmarks.images.build(bench_id)`, then inspect `images.list(bench_id)`. |

`images.build` retries eligible pending/failed runtime work for the benchmark, not a selected
single task. In the matching release, ingestion history shows the new retry operation and its
progress. Ready runtimes and identified in-flight builds are skipped. Do not overlap it with
an active ingestion operation to bypass scheduling. Earlier operation failures remain historical;
a successful retry does not erase them. Retry operations contain runtime results, with zero new
submitted/registered tasks. Use `images.list(bench_id)` for current per-task readiness; their
failures do not provide an affected-task receipt cursor.

For corrected source, keep the same agent and benchmark name and save the new operation and
benchmark IDs. A new version contains only the tasks you submit; a subset does not inherit the
rest of the old version. Keep first-pass and recovered outcomes separate. A deterministic shared
failure should be diagnosed once rather than repeated across the queue; task-local failures need
not block independent healthy tasks.

Managed upload/context bounds protect worker resources and are distinct from provider image
constraints. If a bound rejects input, retain its dimension, observed value and allowed value.

## 5. Confirm readiness, then test the task

After registration, refresh the selected operation before inspecting its runtimes:

```python
from trajectory.lib import get_operation

status = get_operation(client, operation_id).refresh()
bench_id = status.bench_id
images = client.benchmarks.images.list(bench_id)
for image in images.images:
    print(image.task_id, image.build_status, image.provider_ref, image.failure_message)
```

Reconcile every input with its registered task and runtime result. Registration counts, operation
completion and image readiness answer different questions. Keep unresolved inputs and failed
runtimes visible; a subset of ready images is not full-population success.

Continue with the [task-diagnostic walkthrough](task_diagnostics.md) to check startup, imports
and credentials using the submitted `run_command`. A successful build does not prove solver
or grader correctness; those checks come after ingestion.
