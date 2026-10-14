# Connect an existing grader to Trajectory

A successful grader command or runtime image build does
not prove that the submitted harness can read and publish that grade.

The adapter is the part of your harness that reads the grader's output and calls the SDK.
A grader may return a scalar, a JSON object, or several metrics. The adapter reads the
grader's output and forwards the selected score to `log_reward(value=...)`. If the adapter
reports a missing `reward` key, check whether the grader uses a different score field and
map that field explicitly to `log_reward(value=...)`.

## Choose the score explicitly

Inspect the grader implementation and a real output from the benchmark. Record:

- Where the result comes from: a file, an API response, or a return value etc.
  If several files exist, check which one the reader actually consumes.
- Which field or documented formula represents task performance. A task can produce files
  containing accuracy, latency, and cost; explicitly choose which metric or formula
  determines reward.
- The score's range, units, direction, and aggregation. Preserve the grader's partial-credit
  semantics and document any transformation, including normalization or conversion of a
  lower-is-better metric.
- How the grader distinguishes a valid failed answer from a grading or execution error.

Keep this mapping in the adapter and document it next to the task configuration. Tasks
may use different grading schemas. Do not assume that one task's output field applies
to the whole benchmark.

For example, an adapter for a grader documented to emit `accuracy` could validate its
selected value like this:

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


score = select_score({"accuracy": 0.75, "latency_seconds": 12.0}, "accuracy")
```

This example expects numeric JSON scores. If the native grader uses booleans, strings, or
nested results, explicitly parse its documented format first. Validate any documented range
as well. Do not silently clamp values, guess a field, or substitute zero for a missing value.

Publish the selected score using the same trajectory ID as the model calls, then complete
the trajectory after reward publication succeeds.

Trajectory combines reward components using a weighted sum. If you publish a total and
its sub-scores with nonzero weights, you can count the same performance twice.
Use weight `0` for additional numeric components that should be visible only for diagnosis.
If you intentionally combine several objectives, define their weights explicitly. An all-zero-weight set does not provide a training signal.

## Keep a zero score distinct from a missing grade

A wrong answer can legitimately earn `0.0`: publish it and complete the graded attempt.
A missing result, invalid score, failed grader, or failed reward import is an error, not an
incorrect answer. Report `termination_reason="ERROR"` when the attempt cannot produce or
publish a valid grade, preserve the underlying exception, and do not mark it successful or
give it a zero reward.

## Validate the complete submitted path

1. Exercise the adapter with a known passing result, a genuine zero, and partial credit if
   supported. Also check that missing fields and malformed results fail visibly.
2. Run a [task diagnostic](gsm8k/README.md#3-validate-one-task) using the packaged
   runtime, task command, grader, adapter, and SDK publication path. Running the upstream
   grader alone skips the integration boundary you need to validate.
3. Wait for the diagnostic to finish and inspect individual task failures. Retrieve the
   resulting trajectory and confirm that its reward component, value, weight, and completion
   state match the native output. An aggregate zero alone cannot distinguish a valid zero
   from a missing grade; inspect the component and failure details.
4. Before scaling up, diagnose representative tasks from every distinct grading schema,
   including both train and test tasks where applicable. Repeat after changing the grader,
   its dependencies, the adapter mapping, or the packaged runtime.
