# Inspect native execution, grading and reports

A correctly integrated task executes its native solver and grader, preserves the native
score (including zero), and retains the reports required by the path it actually ran.
Image readiness, trajectory completion or a numerical reward alone cannot establish this.
Execution outcome and native grade are separate. Keep failures visible and leave the grade
missing when the native scorer produced no valid score. Preserve scores the benchmark defines
for the executed path, including zero penalties and native fallback scores; record their
reasons and auxiliary failures separately. Neither a score nor an average over graded rows
establishes successful execution or complete integration.

Complete the [ingestion and runtime-readiness loop](ingestion.md) first. It covers managed
builds and prebuilt images, credentials, every submitted input, failed build logs and retries.
Then use this page to validate native execution, grading and report retention.

This draft's complete-attempt examples require the upcoming matching API/SDK release.
Publishing diagnostics while cancelling also requires the matching backend update.
Release and ordinary-customer checks remain pending.

## 1. Check native prerequisites cheaply

Use the pinned harness's own setup instructions and entrypoint to check imports, executables,
plugins, native services, architecture and credentials in the intended runtime. Preserve
actor/private-grader isolation. The [runtime packaging guide](../README.md#package-a-benchmark-runtime)
explains what the platform starts and what your image must install.
Run the harness's prerequisite validation in that runtime before model work; executable version
checks and imports alone do not establish that its setup path will work.
Use the same command, shell, working directory and environment as the solver and grader.
A tool available during image construction may be unavailable to the harness at execution time.

Before making a model request, check that the installed client accepts the native call's
actual arguments; importing or constructing the client does not test this. Trajectory SDK
chat calls pass additional fields such as `tools` and `tool_choice` through `extra_body`;
see the [model-client adapter example](big_finance_benchmark.md#adapt-the-existing-model-client).

Check changed model-client connections with a short request and a finite timeout before
expensive evaluation. Fixed auxiliary models need their [own client and credentials](auxiliary_clients.md).
A capped connection probe verifies routing; it does not replace native budgets or prove that
the solver or grader works. Reuse evidence for unchanged prerequisites.
For managed tasks, run the probe with the same `SecretRef` bindings and client configuration
as the harness. A successful request from the onboarding workspace does not verify the
values injected into a managed task. Keep the helper's SDK origin separate from its native
inference base, as the auxiliary-client example shows.

## 2. Run a small managed evaluation and inspect it while it runs

Choose representative tasks before looking at outcomes, covering the native paths you need.
Use an explicit evaluation split and a bounded cohort; preserve original dataset roles if
creating a diagnostic benchmark. Discover supported model/options with
`client.evals.list_options(...)`, then use the [managed evaluation example](../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform).
Verify the returned task selection and resolved configuration rather than assuming defaults.

Finish registration and reconcile registered IDs against your original submitted inventory
before admitting a full-population run. `task_split="all"` freezes the registered TRAIN and
TEST tasks visible at admission; it cannot include inputs that have not registered.

In the upcoming API release, an ALL evaluation may include tasks whose runtime build
explicitly failed, alongside ready tasks. Those failed runtimes remain selected and produce
failed attempts with no grade; they are not silently excluded from coverage. Pending,
building, missing or unknown runtimes, invalid commands/providers and missing secrets still
block admission. TEST-only evaluations and training retain their existing readiness checks.
A mixed evaluation can end with run status `failed` while other tasks finish and receive
grades. Inspect the selected-task and attempt listings; an average over graded attempts is
not a score over the full submitted population.

Read the execution settings retained for the accepted run separately from its requested options:

```python
run_id = "evr_YOUR_RUN"
run = client.evals.runs.retrieve(run_id)
configuration = client.evals.runs.configuration(run_id)
print("Requested options:", run.options)
print("Recorded execution:", configuration.model_dump())
```

`available=False` or a null field means recorded evidence is unavailable; the API does not
substitute today's model catalog defaults. Check model identity, context, sample count,
maximum active rollouts and execution limits against your intended experiment. The recorded
concurrency cap is an admission setting, not a measurement of simultaneous active tasks.
These settings do not replace native-report checks or prove which native grading paths ran.

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
page cannot inflate counts. Automatic item iteration raises `RuntimeError` on a reset: discard
that partial listing and start again, or use `iter_pages()` to handle the reset explicitly.
When filtering by `status` or `graded`, all pages use the first
page's database snapshot so changing outcomes cannot skip matching rows. Those cursors expire
after 30 minutes; start a new listing to see newer outcomes or recover from an expired cursor.
A paginated read during execution is not a final report.
`supported=false` means the required historical records
are unavailable, not that the run had no selected work. A selected task without an attempt is
unstarted. An attempt can fail before a trajectory exists; a trajectory-only listing misses
that failure. Zero and `None` are different results. The attempt's `grade` is the recorded
evaluation score, which can include a platform limit penalty. For example,
`trajectory_termination_reason="LIMIT_REACHED"`, rollout `CANCELLED` and `grade=0` can occur
without any native grading. Keep that score in the run's accounting and record whether the
native grader ran, using the reports required by the benchmark's source. Genuine native zeros
remain valid. Inspect both the trajectory stop reason and rollout diagnostics even if trajectory
capture says completed. Filters such as `task_id=...`, `status=...` and `graded=False`
help investigate; keep the unfiltered accounting separately.

Harness exception messages come from your benchmark command and are visible only through
the owning organization's authorized interfaces. Capture masks its launch credentials, and
the API masks registered secrets. This is not complete secret scrubbing: credentials acquired
or transformed by your harness outside that set may remain visible to members and API keys of
your organization. Older records without verified launch-secret masking retain bounded error
categories and traceback context, but omit freeform messages. Platform provisioning errors
use separate fixed messages. Use available native reports and artifacts to investigate;
absent diagnostic text does not mean execution succeeded.

In the evaluation UI, use **Selected tasks** and **All attempts**. Task details and **Run history**
connect prior and repaired attempts by exact task identity. Inspect a new run when a repair
creates a new benchmark/task version; matching display names alone do not prove identity.
The benchmark table lets you select the exact evaluation; its details retain every attempt,
including failures without trajectories. Use **Benchmark version** to inspect an older
submission without mixing its task IDs or results with the repaired version.

## 3. Check native reports and required outputs

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

Use the pinned benchmark source to determine which tests, criteria and reports are required
for the path that ran, including its rules for skips, penalties and fallback scores. Compare
those requirements with the returned identities, results and errors. Record missing required
outputs separately from optional omissions and genuine negative verdicts. Preserve valid
native scores, including zero, alongside that evidence; a fallback score alone does not prove
that every component ran successfully.

## 4. Repair the owner and confirm on additional tasks

Use the public error and pinned source to identify whether the cause belongs to packaging,
the integration, a platform interface or an upstream prerequisite. Fix the owning cause and
retry affected work. Stop repeating an unchanged deterministic failure; keep independent
tasks moving. Substantiate upstream exceptions without modifying native tests or hiding the
selected failure.

Retain original and repaired operation/run IDs, task versions and reports. Report first-pass
and recovered results separately; retries do not increase the number of unique source tasks.
For a repaired snapshot of the same benchmark, keep the same `agent_id` and
`BenchmarkSpec.name`, and submit the changed content with a new idempotency key. Each
submission gets a separate immutable benchmark ID; Platform groups version history by
agent and benchmark name. Changing the name creates a separate history, even when
`family` is unchanged. Record the repair label in your submission ledger or task tags
rather than changing the benchmark name. A subset submission contains only that subset;
it does not inherit omitted tasks from an earlier snapshot.
Once the common execution and reporting path works, scale to the intended population through
the managed scheduler. A task-local failure need not hold back independent tasks or require
a perfect cohort. Confirm material repairs on unused cases from that population so success
is not confined to debugging cases. Preserve held-out data for later measurement.

## Record events and artifacts

`log_event` records a named JSON payload on a trajectory without changing its reward. Use a
different `event_id` for each distinct event and reuse that ID only when retrying the same
write. An existing ID is deduplicated, not updated. Record new events before completing the
trajectory; read them later with `client.trajectories.list_events(tid)`.

When a model call returns `trajectory_closed` with `x-should-retry: false`, stop model calls.
While the trajectory is `cancelling` and the harness is still running, it can publish events
and finish artifact uploads to retain the failure report. New model steps and reward writes
remain closed. Finalization can start when the harness exits or is stopped; once it starts,
new events and artifact attachments are rejected. Reserving an upload does not extend this
window, so complete the upload and record its event before exiting or calling `complete`.

A retained report does not itself establish a native grade. Distinguish missing grading
outputs from a genuine native zero and from a score imposed by a platform limit.

Use events for structured diagnostics and artifacts for files such as test logs or reports.
Completing an artifact upload attaches the file to the trajectory. The event in this example
also records its ID and filename. If a diagnostic upload fails, report it separately from the
task's execution result and computed reward.

An artifact upload can contain up to 16 MiB. Compress larger text reports or split them into
files before uploading.

Before finalization, upload a compressed report for trajectory `tid`:

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
