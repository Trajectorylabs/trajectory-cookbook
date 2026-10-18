# Ending a trajectory

Log the reward, then call `client.trajectories.complete(tid, termination_reason=...)`. The
platform records the reason you send; it does not infer it from the model's replies. The
default, `ENV_DONE`, means the attempt finished normally.

## Choose the reason

- `ENV_DONE`: the attempt finished normally, including a graded wrong answer.
- `ERROR`: a model, environment, or grader failure prevented a valid attempt. Do not log an
  invalid grade as a reward.
- `TIMEOUT`: your harness stopped the agent at its own time limit.
- `TRUNCATION`: the model ran out of room. Its final reply was cut off at its output-token
  limit, or the conversation no longer fit its context.
- `MISSING_SUBMIT`: the task needs a submission and the agent stopped without one.
- `MAX_STEPS`: your harness stopped the agent at its own turn or step limit.

The ending and the grade are separate. If the source evaluator grades a cut-off reply,
log that native grade before completing with `TRUNCATION`.

Training admits `ENV_DONE`, `LIMIT_REACHED` and `MAX_STEPS`; `TRUNCATION` requires
`train_on_truncation`. An ungraded `ENV_DONE` is excluded from training. In training and
evaluation, `LIMIT_REACHED`, `MAX_STEPS` and `TRUNCATION` use 0 if ungraded and cap positive
rewards at 0.

`ERROR` and timeouts are excluded by default. `train_on_error` admits `ERROR`;
`train_on_timeout` admits `TIMEOUT` and `SANDBOX_TIMEOUT`. These enabled failures use the
recorded reward, or 0 when none is recorded. `MISSING_SUBMIT` remains excluded. Other training
filters can still reject admitted samples. Check the selected model's `training.list_options`
before setting these options. `ERROR` counts toward the run's failure limit unless
`train_on_error` is enabled; it counts as failed in evaluation regardless of training options.
The failure limit counts failed rollouts, not completed rollouts rejected from training. Repeated
`TRUNCATION` does not trip it.

The platform also sets `MAX_STEPS` when its turn limit stops the model. It sets
`LIMIT_REACHED` for token or context limits, and `STALE`, `CANCELLED` and `SANDBOX_*` on its own.
`GOLDEN_PATCH` is reserved.

## Detect a cut-off reply

Read the finish reason from the model response:

```python
finish_reason = response.choices[0].finish_reason
```

If the harness hides the response, read it from the latest recorded step after the request
(after confirming the trajectory has at least one step):

```python
num_steps = client.trajectories.retrieve(tid).num_steps
page = client.trajectories.steps.list(tid, start_step_index=num_steps - 1, limit=1)
finish_reason = page.items[0].messages[-1].finish_reason
```

In either case, check before dispatching tool calls:

```python
cut_off = finish_reason in ("length", "max_tokens")
```

If cut off, stop the actor loop without dispatching that reply's tool calls. Grade any answer
the source evaluator accepts, then complete with `TRUNCATION`.

## When the platform ends the trajectory

After a run's turn, token or context limit is reached, model requests return an empty reply
with `metadata.trajectory_termination_reason: "trajectory_cancelled"`. Stop calling the
model. The platform keeps `LIMIT_REACHED` or `MAX_STEPS` even if you then call `complete()`.
If `max_total_tokens_per_trajectory` equals `max_output_tokens_per_step`, a full-length
reply can hit the total cap before the harness grades it. Leave room above the per-step cap
when the harness must grade a cut-off reply; `LIMIT_REACHED` may otherwise be ungraded.
For model request errors and retries, see [Connect the actor](../README.md#connect-the-actor).

A rollout's `CANCELLED` state or reason describes its execution lifecycle. It does not prove
the trajectory ended as `CANCELLED`: the trajectory may have reached `MAX_STEPS` first.
Use the recorded trajectory reason when available; otherwise report that distinction as
unverified rather than inferring it from the rollout or an empty model response.
