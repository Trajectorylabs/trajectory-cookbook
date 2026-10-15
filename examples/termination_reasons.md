# Ending a trajectory

Log the reward, then call `client.trajectories.complete(tid, termination_reason=...)`. The
platform records the reason you send; it does not infer it from the model's replies. The
default, `ENV_DONE`, means the attempt finished normally.

## Choose the reason

Use the first reason in this list that matches:

1. `ERROR`: there is no valid grade. Model requests failed after retries, or the environment
   or grader failed. Do not log a reward, even if the grader returned one.
2. `TIMEOUT`: your harness stopped the agent at its own time limit, such as Harbor's
   `AgentTimeoutError`.
3. `TRUNCATION`: the model's final reply was cut off at its output-token limit.
4. `MISSING_SUBMIT`: the task needs a submission and the agent stopped without one.
5. `ENV_DONE`: anything else, including a graded wrong answer.

Training uses `ENV_DONE` rollouts with their logged reward. `ERROR`, `TIMEOUT` and
`MISSING_SUBMIT` rollouts are not trained on, and `ERROR` counts toward the run's failure
limit. `TRUNCATION` is trained on only when the run sets `train_on_truncation`, with positive
rewards counted as 0. The platform sets `LIMIT_REACHED` and `MAX_STEPS` when a run's token,
context or turn limit stops the model, and `STALE`, `CANCELLED` and `SANDBOX_*` on its own.
`GOLDEN_PATCH` is reserved.

## Detect a cut-off final reply

Most agent loops stop at the first reply without a tool call, including a reply cut off
mid-thought, and then grade unfinished work. Check the final reply before completing:

```python
import json

num_steps = client.trajectories.retrieve(tid).num_steps
page = client.trajectories.steps.list(tid, start_step_index=num_steps - 1, limit=1)
reply = page.items[0].messages[-1]
cut_off = reply.finish_reason in ("length", "max_tokens") and not json.loads(reply.tool_calls or "[]")
client.trajectories.complete(tid, termination_reason="TRUNCATION" if cut_off else "ENV_DONE")
```

If your harness reads the model's responses directly, check their `finish_reason` instead.

## When the platform ends the trajectory

After a run's turn, token or context limit is reached, model requests return an empty reply
with `metadata.trajectory_termination_reason: "trajectory_cancelled"`. Stop calling the
model. The platform keeps `LIMIT_REACHED` or `MAX_STEPS` even if you then call `complete()`.
For model request errors and retries, see [Connect the actor](../README.md#connect-the-actor).
