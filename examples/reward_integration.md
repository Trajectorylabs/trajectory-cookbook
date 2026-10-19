# Preserve an existing grader's score

Read the source evaluator's result reader, not just the grader's output. Identify the
field or formula it reports, its range, and whether partial credit contributes. Pass
that native value to `log_reward` with bounds that include it. A successful grader
command does not establish which score was published. If the source has several scoring
modes and no clear primary one, resolve the comparison target before ingestion.

## Select the score in the harness

Put score selection after grading in the runtime harness, not in the upload script.
This example's source reader uses `accuracy`; another benchmark may use a different key.

```python
import math

def select_score(scores: dict[str, object], key: str) -> float:
    if key not in scores:
        raise ValueError(f"Missing score {key!r}; available: {sorted(scores)}")
    value = scores[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Score {key!r} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Score {key!r} must be finite")
    return score
```

With the harness's `client`, `tid`, and `model_answer`, publish the source reader's
primary score after successful grading:

```python
report = grade_answer(model_answer, expected_answer)
score = select_score(report, "accuracy")
client.trajectories.log_reward(tid, name="accuracy", value=score)
```

Parse the source grader's documented format before
`select_score` when it returns booleans, strings, or nested data. Validate its range;
for a signed score, declare `min_value=-1, max_value=1` when logging. Do not clamp,
round, guess a missing field, or substitute zero for a failed grader.

The trajectory reward is the weighted average of scaled components. Publish either
the native total once or components that reproduce it, not both.

A wrong answer can have a valid zero. An unrecovered model, environment, or grader error
is `ERROR` with no published reward, even when an incidental verifier result exists.
Keep that result in an event or artifact instead. See the
[cut-off reply example](termination_reasons.md) and SDK's
[reward contract](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/benchmarks.mdx#report-the-grade-through-the-sdk)
and [termination reasons](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/termination-reasons.mdx)
before completing the trajectory.

## Show secondary scores without changing the reward

Publish the source's primary score once. If this grader also returns a
`criteria_fraction` in `[0, 1]`, insert these calls before `complete()` in the same
harness. Weight `0` makes it visible without changing reward; an event can hold grader
details separately:

```python
criteria_fraction = select_score(report, "criteria_fraction")
client.trajectories.log_reward(
    tid, name="criteria_fraction", value=criteria_fraction, weight=0.0,
)
client.trajectories.log_event(
    tid, event_id="rubric", name="rubric",
    payload={"criteria_fraction": criteria_fraction},
)
```

Keep private grader references out of events. Retrieve a trajectory with
`include_reward_components=True` to verify names, values, bounds, and weights; events
appear under **Events** in the trajectory trace, while components appear under **Reward**.
Even a zero-weight reward record can prevent the default automatic grader from running,
so use it only when the harness owns reward publication.

## Check the submitted path

Compare the adapter with the source evaluator on saved wrong, partial, and correct
outputs, including the source's aggregation. Check a missing score field and an
execution failure with an incidental grade: both must fail visibly, not publish zero.
Then run a [task diagnostic](task_diagnostics.md) using the packaged
runtime and inspect its trajectory component, ending, and failure. Repeat for distinct
grading schemas before scaling up. If the source uses external tools, confirm they
worked and grading followed the source's rules; a zero can hide a broken tool path.
