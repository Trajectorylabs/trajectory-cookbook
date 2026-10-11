# Inspect an evaluation and repair the integration

After [ingestion and runtime readiness](ingestion.md), run a small representative evaluation
and inspect its outcomes before scaling up. Keep every submitted task, failed attempt and
genuine zero. A recorded score says what was saved; your source defines whether the intended
grading actually happened.

**Release requirement:** the attempt, selected-task and grading APIs below require the
matching inspection API/SDK release. Use the SDK provided for your deployment. These examples
do not imply that an older SDK or API supports the new methods.

## Account for the work

Reconcile the original inputs using the [ingestion guide](ingestion.md#2-account-for-every-submitted-input).
Tasks rejected before registration will not appear in an evaluation. Check required imports,
commands, services and credentials in the intended runtime before spending time on rollouts.
A successful call from the onboarding workspace does not verify the managed runtime's setup.

Choose a representative cohort before examining results. Follow the existing
[managed evaluation example](../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform)
with explicit model and execution options, and retain the accepted run ID.

```python
from trajectory import Client

client = Client()
run_id = "evr_YOUR_RUN"
run = client.evals.runs.retrieve(run_id)
print(run.bench_id, run.options)
print("Recorded worker configuration:", run.execution_configuration)

selection = client.evals.runs.list_selected_tasks(run_id, limit=100)
print("Selected population:", selection.supported, selection.total_tasks)
for task in selection:
    print(task.task_id, task.expected_attempts,
          task.recorded_attempts, task.unstarted_attempts)

attempts = client.evals.runs.list_attempts(run_id, limit=100)
for attempt in attempts:
    print(attempt.task_id, attempt.sample_id, attempt.status,
          attempt.trajectory_id, attempt.grade, attempt.applied_reward,
          attempt.rollout)
```

Selected tasks preserve work that has not started. Attempts retain retries and failures,
including failures before a trajectory exists. Keep an unfiltered inventory keyed by
`sample_id`; filtering is for investigation. `supported=False` means complete historical
coverage is unavailable, not that no work happened. Requested options are not proof of the
settings used by a worker. `execution_configuration` exposes retained model, context and
concurrency settings when available; null means they were not retained.

Iteration reads subsequent pages. If new attempts invalidate a live traversal, automatic
iteration raises `RuntimeError`: discard the partial list and restart. Explicit page consumers
can use `iter_pages()` and inspect `page_reset`. Selected-task membership is frozen for the run.

## Follow one attempt through the product

Open the evaluation from the benchmark's task table, then choose a task and its exact attempt.
The URL preserves that choice when you share the page or return to it.

| View | Question it helps answer |
| --- | --- |
| Task table | Did this input register, become runtime-ready, execute and record a grade? Which work has not started? |
| Rollout | What model/tool conversation ran? If execution failed, what recorded error helps explain it? |
| Grade | What score was applied, which source supplied it when known, and what explanation or components were retained? |
| Files | What readable report or raw output did the integration attach? |

A missing trajectory does not hide the attempt's recorded failure. Build errors belong to
the ingestion/runtime operation; execution errors belong to the attempt. A simple
`log_reward` integration may have no sandbox or separate grader execution.
The declared task input is shown separately from the actual model conversation; inspect both
when checking that the prompt reached the model correctly.

For an attempt with a trajectory, an onboarding agent can inspect the same records:

```python
trajectory_id = "traj_YOUR_SELECTED_ATTEMPT"
for step in client.trajectories.steps.list(trajectory_id):
    print(step.model_dump())

grading = client.trajectories.grading(trajectory_id)
print(grading.reward_sets, grading.executions)
for event in client.trajectories.list_events(trajectory_id):
    print(event.event_id, event.event, event.evidence)
```

The attempt's `applied_reward` identifies the selected reward source when recorded. A null
source is unknown; matching a saved reward's value to the score cannot establish its source.
Later regrades and current reward sets can differ from the result that this attempt used.
Treat unavailable historical detail as unavailable.

For managed grading, follow the recorded execution trajectory when present. Authorized
retained files can be downloaded through `client.trajectories.grading_file(...)`; a file
reference does not guarantee that the storage read succeeds. Permission to run a shared
grader does not grant access to its private source.

## Make the result understandable

Existing reward calls need no new metadata. An optional explanation helps a human understand
the supplied result:

```python
client.trajectories.log_reward(
    tid,
    name="output_validity",
    value=0,
    explanation="The submitted output file is empty.",
)
```

Use the scorer's actual result. A zero is a valid negative verdict; it must not replace a
missing grade or execution error. A Platform-imposed penalty is also distinct from your
scorer's output.

When the scorer already produces a report, [attach that file](diagnostic_artifacts.md) before
completing the trajectory. Prefer readable text or Markdown for human inspection. Arbitrary
JSON remains technical data; Platform cannot infer a useful explanation from its field names.
After the upload completes, you can give the file a label and declare its relationship to a
logged reward:

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

Omit `reward_source` for a file unrelated to grading. Labels, explanations and associations
are supplied by the integration; they do not prove that a particular reward revision used
the file. Missing optional reports are not execution failures. No per-test logging or standard
report schema is required.

Read a file with `client.artifacts.retrieve(artifact.artifact_id)` and its `download_url`.
Retrieve a new URL if it expires. A failed download does not mean the grade is missing.
Avoid logging secrets in reports, explanations or execution output.

## Diagnose, repair and verify

Compare retained commands, outputs and scoring evidence with the source's required behavior.
Separate execution coverage, recorded-score coverage and complete grading evidence from the
score earned by the model. The Platform exposes the integration's records; it does not certify
that arbitrary grading code preserves the original benchmark.

Determine whether a failure belongs to packaging, integration, Platform or the original
source. Repair the owning cause and follow the ingestion guide's
[retry and versioning rules](ingestion.md#4-repair-the-cause-and-retry-affected-work).
Verify affected work, then test additional representative tasks before scaling. Do not repeat
an unchanged deterministic failure or hold healthy tasks until every task succeeds.

Retain the original denominator, prior versions, attempts and reports. Count unique source
tasks separately from attempts, and report first-pass coverage separately from recovery.
Poll progress independently of report downloads; a timed-out observation does not cancel
the underlying execution.
