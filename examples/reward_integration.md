# Connect an existing grader to Trajectory

The adapter is the part of your harness that reads the grader's output and calls the SDK.
A grader may return a scalar, a JSON object, or several metrics. The adapter reads the
grader's output and forwards the selected score to `log_reward(value=...)`. 

A successful grader command or runtime image build can sometimes hide that the submitted harness
failed to read and publish the real grade. If the adapter reports a missing `reward` key, 
check whether the grader uses a different score field and map that field explicitly 
to `log_reward(value=...)`.

## Choose the score explicitly

When adapting an existing benchmark, preserve the primary score used by its original
evaluation entrypoint and reporting path. Trace and preserve its scoring formula, including
any partial scores, component weights, and aggregation. The published aggregate should match
what the original benchmark would report for the same output.

Tip: if you want to keep additional numeric signals visible without affecting that score,
consider logging them as named reward components with weight `0`; see
[Log several reward components and grader output](#log-several-reward-components-and-grader-output).

For an existing benchmark, inspect its grader, evaluation reader, reporting code, and a real
output. Record:

- Where the result comes from: a file, an API response, or a return value etc.
  If several files exist, check which one the reader actually consumes.
- Which field or formula produces the original benchmark's primary reported score, with
  source references showing where it is computed and consumed.
- The score's range, units, direction, thresholds, and aggregation. Preserve partial credit
  only when it contributes to that primary score; do not impose binary scoring on benchmarks
  whose primary score is continuous. Preserve the original score conversion without adding
  normalization, rounding, or thresholds.
- How the grader distinguishes a valid failed answer from a grading or execution error.

Keep this mapping in the adapter and document it next to the task configuration. Tasks
may use different grading schemas. Do not assume that one task's output field applies
to the whole benchmark.

If documentation and executable evaluation behavior disagree, or multiple scoring modes
exist without a clear primary metric, report the ambiguity and resolve the intended
comparison target before completing ingestion. Do not silently choose a scoring mode.

## Put the adapter in your task harness

In the [main README](../README.md#log-reward-to-the-trajectory), `check_answer(...)`
returns a scalar that goes directly into `log_reward`. If your existing grader instead
returns a report such as `{"accuracy": 0.75, "latency_seconds": 12.0}`, add score selection
between grading and `log_reward` in your runtime harness, after the model finishes the task.
This code runs inside the submitted runtime, not in the script that uploads the benchmark.

The following shows the same task flow as the README. Here, `grade_answer` stands for your
benchmark's existing grader.

```python
from trajectory import Client
from your_benchmark.grading import grade_answer


def run_task(prompt: str, expected_answer: str) -> None:
    client = Client()
    tid = client.trajectories.create().tid
    response = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        x_trajectory_id=tid,
    )
    model_answer = response.choices[0].message.content
    scores = grade_answer(model_answer, expected_answer)
    score = select_score(scores, "accuracy")
    client.trajectories.log_reward(tid, name="accuracy", value=score, weight=1.0)
    client.trajectories.complete(tid)
```

Define `select_score` in the same harness module before calling `run_task`. It validates the
field selected for this grader; `"accuracy"` is this example's mapping, not an SDK requirement.
The task flow above shows the successful path; keep your harness's error reporting around
it as described in [Keep a zero score distinct from a missing grade](#keep-a-zero-score-distinct-from-a-missing-grade).

```python
import math


def select_score(scores: dict[str, object], score_key: str) -> float:
    if score_key not in scores:
        raise ValueError(
            f"Cannot publish reward: expected score field {score_key!r}; "
            f"available fields: {sorted(scores)}"
        )
    value = scores[score_key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Score field {score_key!r} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Score field {score_key!r} must be finite")
    return score

```

This example expects numeric JSON scores. If the native grader uses booleans, strings, or
nested results, explicitly parse its documented format first. Validate any documented range
as well. Do not silently clamp values, guess a field, or substitute zero for a missing value.

Publish the selected score using the same trajectory ID as the model calls, then complete
the trajectory after reward publication succeeds.

The reward is the weighted average of its components. Preserve the benchmark's formula by
publishing either its total with weight `1` or its weighted components, without counting both.
For failed or pending grading, keep diagnostics in [events or artifacts](../README.md#report-native-results).
Even a zero-weight reward record can prevent the default automatic grader from running.

## Log several reward components and grader output

Each `log_reward` call adds a named component, and the trajectory's logged reward is the
weighted average of its components. A component with weight `0` is recorded without counting
toward the reward. `log_event` records a named JSON object on the trajectory without affecting
the reward, so a grader's full result can sit next to the scores it produced.

In this example, `grade_rubric` stands for a rubric grader that returns how many required
criteria the answer met and the verdict for each criterion. The reward is the pass/fail score;
the fraction of criteria met is recorded alongside it:

```python
from trajectory import Client
from your_benchmark.grading import grade_rubric


def run_task(prompt: str, rubric: list[dict]) -> None:
    client = Client()
    tid = client.trajectories.create().tid
    response = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        x_trajectory_id=tid,
    )
    model_answer = response.choices[0].message.content
    result = grade_rubric(model_answer, rubric)
    met, total = result["n_required_met"], result["n_required"]

    client.trajectories.log_reward(
        tid,
        name="pass",
        value=float(met == total),
        weight=1.0,
        explanation=f"{met}/{total} required criteria met",
    )
    client.trajectories.log_reward(
        tid,
        name="criteria_met",
        value=met / total,
        weight=0.0,
        explanation=f"{met}/{total} required criteria met",
    )
    client.trajectories.log_event(tid, event_id="rubric", name="rubric", payload=result)
    client.trajectories.complete(tid)
```

Record events before completing the trajectory. Reuse an `event_id` only to retry the same
write; an existing ID is not updated. Read the events back with `list_events`, which returns
the newest first:

```python
for event in client.trajectories.list_events(tid):
    print(event.event_id, event.event)
```

In the trajectory's trace view, each component appears under **Reward** and each event under
**Events**.

## Keep a zero score distinct from a missing grade

A wrong answer can legitimately earn `0.0`: publish it and complete the graded attempt.
A missing result, invalid score, failed grader, or failed reward import is an error, not an
incorrect answer. Report `termination_reason="ERROR"` when the attempt cannot produce or
publish a valid grade, preserve the underlying exception, and do not mark it successful or
give it a zero reward.

Be careful that a completed grader isn't taken to mean successful completion. 
If chat completion fails after all retries and the core task cannot complete,
report `termination_reason="ERROR"`. Otherwise, we could end up training on infrastructure
failures rather than model performance. For Harbor (and other similar existing benchmarks),
check `exception_info` even when `verifier_result` contains a score.
Retain any grade only in diagnostic events or artifacts. 
See [Report native results](../README.md#report-native-results) and
[Ending a trajectory](termination_reasons.md) for every termination reason.

## Validate the complete submitted path

1. For an existing benchmark, compare the original evaluation path and the adapter on the
   same saved outputs, including failing, partially correct, and fully correct cases where
   applicable. Require exact equality between the published aggregate and the original
   benchmark score, including any partial-score weighting. A partially correct answer must
   still score zero when the original benchmark requires full correctness.
2. Check that missing fields and malformed results fail visibly. Inject an execution
   failure with a successful grader: the trajectory must end as `ERROR` with no published
   reward, while a valid incorrect answer must retain its native zero.
3. Run a [task diagnostic](gsm8k/README.md#3-validate-one-task) using the packaged
   runtime, task command, grader, adapter, and SDK publication path. Running the upstream
   grader alone skips the integration boundary you need to validate.
4. Wait for the diagnostic to finish and inspect individual task failures. Retrieve the
   resulting trajectory and confirm that its reward component, value, weight, and completion
   state match the native output. An aggregate zero alone cannot distinguish a valid zero
   from a missing grade; inspect the component and failure details.
5. Before scaling up, diagnose representative tasks from every distinct grading schema,
   including both train and test tasks where applicable. Repeat after changing the grader,
   its dependencies, the adapter mapping, or the packaged runtime.
