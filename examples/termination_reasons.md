# Detect a cut-off reply

The SDK's [trajectory endings guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/trajectory-endings.mdx)
defines the reasons your harness can send and their training and evaluation effects.
The platform records the reason you send; it does not infer one from model replies.

Read `finish_reason` before dispatching tool calls:

```python
finish_reason = response.choices[0].finish_reason
cut_off = finish_reason in ("length", "max_tokens")
```

If a wrapper hides the response, read the latest recorded step after the model call:

```python
num_steps = client.trajectories.retrieve(tid).num_steps
page = client.trajectories.steps.list(tid, start_step_index=num_steps - 1, limit=1)
finish_reason = page.items[0].messages[-1].finish_reason
cut_off = finish_reason in ("length", "max_tokens")
```

When `cut_off` is true, stop the actor loop without executing tool calls from that reply.
If the source grader accepts the partial answer, publish its native grade, then complete
with `TRUNCATION`. A clean harness exit does not turn that ending into `ENV_DONE`.

A platform limit can return an empty reply with
`metadata.trajectory_termination_reason: "trajectory_cancelled"`. Stop calling the model
and inspect the recorded trajectory reason; the platform may already have set
`LIMIT_REACHED` or `MAX_STEPS`. Leave room above the per-step output cap if the harness
must grade a reply that fills that allowance.
