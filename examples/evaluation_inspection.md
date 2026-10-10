# Inspect an evaluation and repair the integration

Use this loop after [ingestion and runtime readiness](../README.md#choose-the-runtime-and-its-files): account for the submitted
population, run a small representative evaluation, inspect execution and grading, repair the
owning cause, and retry affected work without losing the original results.

**Release requirement:** the complete-attempt, recorded-configuration, grading-file and optional
evidence examples below require the matching customer inspection API/SDK release. These
interfaces are being qualified together; do not assume an older SDK or deployed API supports
them. The same requirement applies to the Rollout / Grade / Files & records UI described here.

A recorded score answers what was saved. It does not establish that the intended grader ran
or that your integration preserved the original benchmark. Determine the required scoring
behavior from your source, then inspect the execution and supplied evidence. A genuine zero
is a valid result; a missing score is not zero. A failed execution can also carry a Platform
penalty, which must remain distinct from your grader's output.

## 1. Account for the submitted tasks and check prerequisites

Use the ingestion guide to reconcile every submitted input, registered task and runtime
outcome. Keep failed inputs and build logs even when no task was registered. A subsequent
evaluation only selects registered tasks; its task count is not the original upload count.

Check required clients, imports, executables, services and credentials in the intended runtime
before expensive evaluation. Use the same command, working directory and injected configuration
as the task. A successful request from your onboarding workspace does not verify the managed
runtime's credentials or model-client routing. Keep actor and auxiliary clients separate when
your harness requires it; preserve the source's scoring behavior and native execution limits.

The Python SDK requires Python 3.11 or later wherever it is imported, including inside a task
image. If the benchmark needs a different interpreter or dependency set, isolate the SDK's
environment and keep the source tests on their intended interpreter.

Choose a small representative cohort before examining its outcomes. Use the
[managed evaluation example](../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform),
keeping the intended task split and explicit model/options. Preserve the accepted run ID:

```python
from trajectory import Client

client = Client()
run_id = "evr_YOUR_RUN"
run = client.evals.runs.retrieve(run_id)
configuration = client.evals.runs.configuration(run_id)
print("Requested options:", run.options)
print("Recorded execution:", configuration.model_dump())
```

Inspect the recorded model, context and execution limits. `available=False` or a null field
means retained evidence is unavailable; it is not replaced with today's defaults. A recorded
concurrency cap describes admission configuration, not measured simultaneous execution.

## 2. Inspect selected work and every attempt

In the UI, select the evaluation from the benchmark table. Inspect ingestion/runtime and
execution outcomes separately, then open the exact task attempt. **Selected tasks** and
**All attempts** retain unstarted work, failures and retries. Use version/run history to
return to an earlier submission rather than substituting a later successful attempt.

An onboarding agent can retrieve the same population and attempt records:

```python
selection = client.evals.runs.list_selected_tasks(run_id, limit=100)
print(selection.supported, selection.total_tasks)
for task in selection:
    print(task.task_id, task.expected_attempts,
          task.recorded_attempts, task.unstarted_attempts)

attempts = client.evals.runs.list_attempts(run_id, limit=100)
print(attempts.supported)
for attempt in attempts:
    print(attempt.task_id, attempt.sample_id, attempt.status,
          attempt.trajectory_id, attempt.grade,
          attempt.trajectory_termination_reason, attempt.startup_rpc_code,
          attempt.applied_reward, attempt.rollout)
```

Iteration reads subsequent pages. Keep an unfiltered accounting keyed by `sample_id`; use
filters only for an investigation. Live unfiltered pages may return `page_reset=true` when
records change. Automatic item iteration raises `RuntimeError` on a reset: discard that
partial traversal and start again, or handle resets explicitly with `iter_pages()`. Filtered
pages use their initial database snapshot; their cursors expire after 30 minutes. Start a new
listing to observe newer results. A live traversal is not a final population report.

`supported=False` means complete historical coverage is unavailable, not an empty run. Keep
any returned records. A selected task without an attempt is unstarted. An attempt can fail
before a trajectory is linked; inspect its startup RPC status and available rollout diagnostic
without requiring a trace ID. A lost startup response does not prove the server rejected the
request. Preserve the attempt and inspect its current state before submitting a retry. Older
attempts may have no recorded startup status.
Retain Platform-imposed scores and their reasons rather than interpreting them as native
scorer success. Keep execution coverage, recorded-score coverage and source-required grading
evidence separate from the score earned by the model.

Read the submitted prompt from the exact benchmark version used by the run:

```python
benchmark = client.benchmarks.specs.retrieve(run.bench_id, include_tasks=True)
if benchmark.tasks is None:
    raise RuntimeError("Task inventory is unavailable")
for task in benchmark.tasks:
    print(task.task_id, task.input_messages)
```

Submitted messages can differ from the actual model conversation. A null message field means
it was not declared or is unavailable; do not reconstruct it from a task's display name.

## 3. Follow one attempt through Rollout, Grade and Files & records

| View | What to inspect |
| --- | --- |
| Rollout | Execution outcome and available failure details, then the actual model/tool conversation. Expand system instructions when needed. A failure before trace creation remains inspectable. |
| Grade | Recorded score, selected source when captured, supplied explanation and components. Inspect the producing managed execution when one exists. Other saved rewards and later regrades are separate. |
| Files & records | Integration-supplied reports and files. Open one item to preview or download it; technical identifiers and raw fields remain available. A same-attempt attachment is not automatically grading evidence. |

Keep the selected attempt in the URL when sharing or returning to a view. A build failure
belongs to its ingestion/runtime operation; do not interpret it as a model failure. A simple
`log_reward` integration may have neither a sandbox nor a separate grader execution.
Missing optional evidence is not itself an execution failure.

For an attempt with a trajectory, inspect the corresponding SDK records:

```python
trajectory_id = "traj_YOUR_SELECTED_ATTEMPT"
for step in client.trajectories.steps.list(trajectory_id):
    print(step.model_dump())

grading = client.trajectories.grading(trajectory_id)
print(grading.reward_sets)
print(grading.executions)
for event in client.trajectories.list_events(trajectory_id):
    print(event.event_id, event.event, event.evidence)
```

Use the attempt's `applied_reward` to identify the source selected for its evaluation score.
A null selection means that relationship was not recorded. Matching a saved reward's value
to the score does not establish its source. Saved components expose their supplied values
and explanations; they can include several sources or later grading work.

For managed grading, use the recorded operation ID and, when present, its
`execution_trajectory_id` to inspect activity through the same steps/events APIs. Read retained
files from that particular execution:

```python
for execution in grading.executions:
    for retained_file in execution.files or []:
        if retained_file.availability != "available":
            print(retained_file.label, retained_file.availability)
            continue
        file = client.trajectories.grading_file(
            trajectory_id, execution.operation_id, retained_file.kind,
        )
        print(file.label, file.media_type, file.size_bytes)
        if file.source_preview is not None:
            for source_file in file.source_preview.files:
                print(source_file.path, source_file.text, source_file.truncated)
            print("Files omitted from preview:", file.source_preview.omitted_files)
        # Download file.download_url with an ordinary HTTP client.
```

The endpoint checks the retained object identity before returning a short-lived download.
For supported owned source bundles, the UI opens a file picker and readable source; the SDK
returns the same bounded text in `source_preview`. Nontext files, shortened text and omitted
files remain distinguishable. Use the original download for the complete archive. An absent
preview does not mean the source is missing or that its code was checked for correctness.
Permission to execute a shared grader does not grant access to private source. Missing files
or source do not erase the recorded score. An output written before publication is not yet a
published reward. For an owned grader, `execution.configuration` exposes retained standard
model inputs: `model`, `temperature` and `max_tokens`, when available. These are configured
values, not proof that the code used them; inspect recorded grader activity for actual calls.
`configuration_availability` distinguishes available, partial, private and unexposed data.
Nonstandard configuration fields are omitted, and private/shared grader configuration is not
returned. A source bundle hash excludes configuration and does not identify every execution
setting.

Execution diagnostics expose bounded, sanitized messages and context when retained. Launch
credentials and registered secrets are masked, but arbitrary credentials acquired or
transformed inside your harness may still appear in its output to your organization. Avoid
logging secrets. A missing diagnostic message does not imply successful execution.

## 4. Supply explanations and useful evidence

Existing reward calls work without an explanation or report. A short explanation can make
a result easier to inspect; it remains a claim supplied by your integration. Inside your
task, use its active trajectory ID `tid` before completing it:

```python
client.trajectories.log_reward(
    tid,
    name="output_validity",
    value=0,
    explanation="The submitted output file is empty.",
)
```

Use your scorer's actual result. This example is a negative verdict, not a replacement for an
execution error. Where more detail is useful, [attach an existing report](diagnostic_artifacts.md)
before completing the trajectory. Prefer a readable text report for human inspection; arbitrary
JSON stays technical data. No per-test logging or standard report schema is required.
After completing the artifact upload, an event can label it and
optionally declare its relationship to a logged reward:

```python
client.trajectories.log_event(
    tid,
    event_id="grading-report",
    name="grading_report",
    evidence=[{
        "artifact_id": artifact.artifact_id,
        "label": "Output validation report",
        "reward_source": {"source": "logged"},
    }],
)
```

`evidence` is optional; existing calls still work. Omit `reward_source` for an attachment
unrelated to grading. The association is declared by your integration, not independent proof
that a particular reward revision used the file. Generic telemetry cannot supply these
validated attachment relationships. Managed outputs have their own recorded producer links.
In the UI, open the labeled file directly from Files & records; explicitly associated logged
reports are also reachable from Grade. Raw event names and payloads are under technical records.

Read an attached file with `client.artifacts.retrieve(artifact.artifact_id)` and its
`download_url`. Links expire; retrieve a new one when needed. Respect report compression and
part counts: one downloaded chunk may not be the whole report. If a safe UI preview is
unavailable, use the original download. A failed read is not evidence of a missing grade.

## 5. Repair the owning cause and verify recovery

Compare the source's required scoring paths, outputs and test identities with the retained
results. Keep missing required outputs distinct from optional omissions and genuine negative
verdicts. Platform exposes your data; it does not certify that arbitrary grading code preserves
the original benchmark.

Diagnose packaging, integration, Platform or upstream failures from the available evidence.
Fix the owning cause and retry affected work. Do not repeatedly launch an unchanged deterministic
failure or hold independent tasks until every task succeeds. Preserve the original submitted
population, failed attempts, versions and reports; count unique source tasks separately from
attempts and report first-pass versus recovered outcomes.

Use the ingestion guide's versioning rules for a repaired submission. A subset submission
does not inherit omitted tasks. Verify material repairs on additional representative tasks,
then scale through the managed scheduler. Keep progress polling independent of expensive report
inspection and download immutable reports once. A timed-out observation does not mean the
underlying execution stopped.
