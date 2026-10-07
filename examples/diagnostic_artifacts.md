# Ingest, inspect and repair a native benchmark

A correctly integrated task executes its native solver and grader, preserves the native
score (including zero), and retains the reports required by the path it actually ran.
Image readiness, trajectory completion or a numerical reward alone cannot establish this.
Execution outcome and native grade are separate. Keep failures visible and leave the grade
missing when the native scorer produced no valid score. Preserve scores the benchmark defines
for the executed path, including zero penalties and native fallback scores; record their
reasons and auxiliary failures separately. Neither a score nor an average over graded rows
establishes successful execution or complete integration.

This draft's complete-attempt and ingestion-history examples require the upcoming API/SDK
release that adds those methods. The release and ordinary-customer checks are still pending.

## 1. Account for the submission

Keep the pinned source revision, original task names/splits and submitted identities. Save the
operation ID returned by ingestion. A client polling timeout does not stop server processing:
inspect that operation before uploading again. You can also discover it in public history,
including a failure before a benchmark was registered:

```python
from trajectory import Client

client = Client()
for operation in client.benchmarks.ingestion.list():
    print(operation.operation_id, operation.bench_id, operation.status,
          operation.registered_tasks, operation.total_tasks,
          operation.ready_runtimes, operation.total_runtimes)
```

Use `bench_id=...` to restrict history to one benchmark version. In the product, open
**Benchmarks → Ingestion history**, or **Ingestion history** on a benchmark's spec.
Inspect every submission outcome, including failures and cancelled work.

Read each task result and failure, following every page. For example:

```python
operation_id = "iop_YOUR_OPERATION"
for kind in ("result", "failure", "runtime"):
    after = ""
    while True:
        page = client.benchmarks.ingestion.list_items(
            operation_id=operation_id, kind=kind, after=after, limit=100,
        )
        for item in page.items:
            print(kind, item.model_dump())
        if page.next_cursor is None:
            break
        after = page.next_cursor
```

Compare task results and task-level rejections against your submitted inventory. The platform
cannot account for source tasks you never submitted. A shared runtime failure affects every
task linked to it: use `kind="runtime_task", runtime_id=...` with the same pagination loop,
or **Show affected tasks** in the UI. Keep the original denominator when fixing a subset.

### Read the failed build before rebuilding

A provider's final error lines may arrive after failure is recorded. Use the original
operation/runtime pair to request a fresh, bounded, redacted excerpt:

```python
logs = client.benchmarks.ingestion.build_logs(
    operation_id=operation_id, runtime_id="rt_YOUR_RUNTIME",
)
print(logs.observed_at, logs.provider_ref, logs.available)
if logs.available:
    print(logs.excerpt)
```

The UI's **Read build logs** uses the same public interface. This read never rebuilds or changes
execution state. It returns up to 6,000 characters from the operation's original failed build,
with registered secret values redacted. `available` means text was returned, not that the
provider has delivered its final error. Credentials embedded in build output that are not
registered for redaction may remain visible to members of the owning organization.
If it still shows installation progress, make a bounded
later read. Do not rebuild just to obtain logs. A provider timeout or missing historical build
reference can return `available=false`; retain the original failure and diagnostic limitation.

## 2. Check native prerequisites cheaply

Use the pinned harness's own setup instructions and entrypoint to check imports, executables,
plugins, native services, architecture and credentials in the intended runtime. Preserve
actor/private-grader isolation. The [runtime packaging guide](../README.md#package-a-benchmark-runtime)
explains what the platform starts and what your image must install.

Before making a model request, check that the installed client accepts the native call's
actual arguments; importing or constructing the client does not test this. Trajectory SDK
chat calls pass additional fields such as `tools` and `tool_choice` through `extra_body`;
see the [model-client adapter example](big_finance_benchmark.md#adapt-the-existing-model-client).

Check changed model-client connections with a short request and a finite timeout before
expensive evaluation. Fixed auxiliary models need their [own client and credentials](auxiliary_clients.md).
A capped connection probe verifies routing; it does not replace native budgets or qualify a
solver/grader. Reuse evidence for unchanged prerequisites.

## 3. Run a small managed evaluation and inspect it while it runs

Choose representative tasks before looking at outcomes, covering the native paths you need.
Use an explicit evaluation split and a bounded cohort; preserve original dataset roles if
creating a diagnostic benchmark. Discover supported model/options with
`client.evals.list_options(...)`, then use the [managed evaluation example](../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform).
Verify the returned task selection and resolved configuration rather than assuming defaults.

Keep the run ID. Inspect both selected work and recorded attempts:

```python
run_id = "evr_YOUR_RUN"
selection = client.evals.runs.list_selected_tasks(run_id)
print(selection.supported, selection.total_tasks)
for task in selection:
    print(task.task_id, task.expected_attempts,
          task.recorded_attempts, task.unstarted_attempts)

attempts = client.evals.runs.list_attempts(run_id)
print(attempts.supported)
for attempt in attempts:
    print(attempt.task_id, attempt.sample_id, attempt.status,
          attempt.trajectory_id, attempt.grade,
          attempt.trajectory_termination_reason, attempt.rollout)
```

These listings paginate when iterated. Unfiltered live attempt pages can reset as new attempts appear;
`page_reset=true` means restart reconciliation and identify rows by `sample_id` so a repeated
page cannot inflate counts. When filtering by `status` or `graded`, all pages use the first
page's database snapshot so changing outcomes cannot skip matching rows. Those cursors expire
after 30 minutes; start a new listing to see newer outcomes or recover from an expired cursor.
A paginated read during execution is not a final report.
`supported=false` means the required historical records
are unavailable, not that the run had no selected work. A selected task without an attempt is
unstarted. An attempt can fail before a trajectory exists; a trajectory-only listing misses
that failure. Zero and `None` are different results. The attempt's `grade` is the recorded
evaluation score, which can include a platform limit penalty. For example,
`trajectory_termination_reason="LIMIT_REACHED"`, rollout `CANCELLED` and `grade=0` can occur
without any native grading. Keep that score in the run's accounting, but leave native grade
qualification missing until the required native report exists. Genuine native zeros remain
valid. Inspect both the trajectory stop reason and rollout diagnostics even if trajectory
capture says completed. Filters such as `task_id=...`, `status=...` and `graded=False`
help investigate; keep the unfiltered accounting separately.

Harness exception messages are available when capture verified redaction using the launch
credentials. Older records without that verification retain bounded error categories and
traceback context, but omit freeform messages. Use available native reports and artifacts to
investigate; absent diagnostic text does not mean execution succeeded.

In the evaluation UI, use **Selected tasks** and **All attempts**. Task details and **Run history**
connect prior and repaired attempts by exact task identity. Inspect a new run when a repair
creates a new benchmark/task version; matching display names alone do not prove identity.

## 4. Check native reports and required outputs

For an attempt with a trajectory, read its recorded events:

```python
for event in client.trajectories.list_events("traj_YOUR_TRAJECTORY"):
    print(event.event_id, event.event)
```

The UI's **Events and artifacts** shows the same records and offers authorized downloads for
artifact IDs. Retrieve an artifact with `client.artifacts.retrieve(artifact_id)` and download
its `download_url`. Preserve the integration's filename, compression, part count and checksum
metadata: one chunk is not necessarily a complete report. Download links expire; retrieve a
new link when needed.

Check native test/criterion identities, counts, errors and penalties. Required outputs depend
on the native path actually executed; distinguish legitimate skips from failed required
components. A genuine native zero is valid evidence. A native fallback score is not proof that
the auxiliary component worked. Missing grading, missing required reports or omitted failed
criteria cannot qualify the task, but must not erase an independently valid native score.

## 5. Repair the owner and confirm on additional tasks

Use the public error and pinned source to identify whether the cause belongs to packaging,
the integration, a platform interface or an upstream prerequisite. Fix the owning cause and
retry affected work. Stop repeating an unchanged deterministic failure; keep independent
tasks moving. Substantiate upstream exceptions without modifying native tests or hiding the
selected failure.

Retain original and repaired operation/run IDs, task versions and reports. Report first-pass
and recovered results separately; retries do not increase the number of unique source tasks.
After the small cohort works, preselect additional tasks covering different required
capabilities. Confirm material repairs on unused cases so success is not confined to the tasks
used for debugging. Preserve held-out data for later measurement.

## Record events and artifacts

`log_event` records a named JSON payload on a trajectory without changing its reward. Use a
different `event_id` for each distinct event and reuse that ID only when retrying the same
write. An existing ID is deduplicated, not updated. Record new events before completing the
trajectory; read them later with `client.trajectories.list_events(tid)`.

Use events for structured diagnostics and artifacts for files such as test logs or reports.
Completing an artifact upload attaches the file to the trajectory. The event in this example
also records its ID and filename. If a diagnostic upload fails, report it separately from the
task's execution result and computed reward.

An artifact upload can contain up to 16 MiB. Compress larger text reports or split them into
files before uploading.

With an active trajectory ID `tid`, upload a compressed report:

```python
import base64
import gzip
import hashlib
from pathlib import Path

import httpx
from trajectory import Client

client = Client()
content = gzip.compress(Path("report.json").read_bytes(), mtime=0)
upload = client.trajectories.artifacts.create_upload(
    tid,
    media_type="application/gzip",
    size_bytes=len(content),
    md5=base64.b64encode(hashlib.md5(content).digest()).decode(),
)
response = httpx.put(upload.upload_url, content=content, headers=upload.headers, timeout=120)
response.raise_for_status()
artifact = client.trajectories.artifacts.complete_upload(tid, upload.artifact_id)
client.trajectories.log_event(
    tid,
    event_id="diagnostic-report",
    name="diagnostic_report",
    payload={"artifact_id": artifact.artifact_id, "filename": "report.json.gz"},
)
```

Finish the upload before completing the trajectory. To read the file later, call
`client.artifacts.retrieve(artifact.artifact_id)` and download its `download_url`.
