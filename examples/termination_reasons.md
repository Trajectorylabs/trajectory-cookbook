# Ending a trajectory

Every rollout ends with `client.trajectories.complete(tid, termination_reason=...)`. The
reason tells training and evaluation whether the attempt can be scored. The platform records
the reason your harness sends; it does not infer it from the model's replies, so the harness
has to check how the attempt actually ended. `termination_reason` defaults to `ENV_DONE`, which
means the attempt finished normally.

## Choose the reason

Check these in order and send the first that applies:

1. The attempt could not produce a valid grade: a model request kept failing after retries,
   the environment or grader broke, or the result is missing or malformed. Send `ERROR`, and
   do not log a reward for it, even if the grader returned one.
2. Your harness stopped the agent at its own time limit, for example Harbor's
   `AgentTimeoutError`. Send `TIMEOUT`.
3. The model's final reply was cut off at its output-token limit, so the agent stopped without
   finishing its turn. Send `TRUNCATION`. See [Detect a cut-off final reply](#detect-a-cut-off-final-reply).
4. The task requires an explicit submission and the agent stopped without one. Send
   `MISSING_SUBMIT`.
5. Otherwise send `ENV_DONE`, including when the agent reached its own stopping condition and
   the benchmark graded a wrong answer. A genuine zero is a valid reward.

Log the reward with `client.trajectories.log_reward(...)` before completing. Report the
failure reason before re-raising an exception; a process that exits without calling
`complete()` is recorded as `ERROR`.

## What each reason does

| Reason | Who sends it | Training | Evaluation |
| --- | --- | --- | --- |
| `ENV_DONE` | Your harness | Trained on with the logged reward | Scored with the logged or automatic grade |
| `ERROR` | Your harness, or the platform when a rollout fails | Not trained on; counts toward the run's failure limit | Counted as failed |
| `TIMEOUT` | Your harness, or the platform when execution exceeds its limit | Not trained on | Scored with any logged or automatic grade, otherwise failed |
| `TRUNCATION` | Your harness | Not trained on unless the run sets `train_on_truncation`; a positive reward counts as 0 | Positive reward counts as 0 |
| `MISSING_SUBMIT` | Your harness | Not trained on | Scored with any logged or automatic grade, otherwise failed |
| `LIMIT_REACHED` | The platform, when a run's token or context limit stops the model | Trained on; a positive reward counts as 0 | Positive reward counts as 0 |
| `MAX_STEPS` | The platform, when a run's turn limit is reached | Trained on; a positive reward counts as 0 | Positive reward counts as 0 |
| `STALE` | The platform, when training no longer needs the rollout | Not trained on | Not normally applicable |
| `GOLDEN_PATCH` | Reserved; do not send | Not trained on | — |

A completed rollout with no reward is counted as failed unless the run sets
`train_on_ungraded`; `TRUNCATION`, `LIMIT_REACHED` and `MAX_STEPS` count a missing reward as 0.
Training stops with `rollout_failure_limit_exceeded` only when every recent rollout failed.
Zero rewards and capped rollouts do not count as failures.

The platform also ends rollouts as `CANCELLED` (run cancelled or cleaned up) and
`SANDBOX_LOST`, `SANDBOX_CRASH` or `SANDBOX_TIMEOUT` (the environment failed). Your harness
cannot send these.

## Detect a cut-off final reply

Most agent loops stop when a reply contains no tool call, including a reply that hit its
output-token limit mid-thought. The harness then grades unfinished work. Check the final reply
before completing:

```python
import json

steps = list(client.trajectories.steps.list(tid))
final = steps[-1].messages[-1] if steps and steps[-1].messages else None
cut_off = (
    final is not None
    and final.role == "assistant"
    and final.finish_reason in ("length", "max_tokens")
    and not json.loads(final.tool_calls or "[]")
)
client.trajectories.complete(tid, termination_reason="TRUNCATION" if cut_off else "ENV_DONE")
```

If your harness reads the model's responses directly, use their `finish_reason` instead of
listing steps. A cut-off reply in the middle of a trajectory is fine when the agent continued.

## Platform limits and model request errors

When a run's turn, token or context limit is reached, the platform ends the trajectory itself.
Later model requests return an empty reply with `finish_reason: "stop"` and
`metadata.trajectory_termination_reason: "trajectory_cancelled"`. Stop calling the model when
you see it. The platform keeps its own reason (`LIMIT_REACHED` or `MAX_STEPS`) even if you then
call `complete()`.

Model request errors carry a `Trajectory-Should-Retry` header; retry only when it is `true`.
The Trajectory client already retries connection failures, timeouts, 408, 429 and 5xx twice.
These are the errors to handle:

| Status | Meaning | Action |
| --- | --- | --- |
| 413 `request_too_large` | The request body is over the size limit | Do not retry; send `ERROR` |
| 400 `context_length_exceeded` | The conversation no longer fits the model's context | Do not retry; send `ERROR` |
| 400 | Invalid request | Fix the request; send `ERROR` if the attempt cannot continue |
| 402, 403 | Insufficient credits or no access to the model | Stop; send `ERROR` |
| 409 `trajectory_busy` | Another request is in flight on this trajectory | Wait for it; do not send concurrent requests |
| 502, 503 | The model service is unavailable | Retried automatically; send `ERROR` if retries run out |
