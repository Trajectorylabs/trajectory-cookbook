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
consider logging them as named reward components with weight `0`.

This score-equivalence requirement applies to adapting existing benchmarks. For a production
use case where you are designing a new reward, carefully choose the outcomes you care about
and define the components, scales, and weights around those objectives.

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

Trajectory combines reward components using a weighted sum. For an existing benchmark,
preserve its original formula: you can publish the computed total with weight `1`, or its
components with the original weights when their weighted sum reproduces that total.
Publishing both the total and its sub-scores with nonzero weights counts performance twice.
For example, a benchmark may average partial scores across criteria, while another requires
every criterion to pass. Preserve whichever rule the original benchmark uses.
For a new production reward, define component weights explicitly around the desired outcomes.
An all-zero-weight set does not provide a training signal. Publish auxiliary reward components
only alongside a valid primary reward; use events or artifacts for diagnostics when grading
fails or the primary reward is still pending, as described in
[Report native results](../README.md#report-native-results).

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
