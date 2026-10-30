# Detect a cut-off reply

The [termination reasons guide](https://docs.trajectory.ai/guides/termination-reasons)
defines the reasons your harness can send and their training and evaluation effects.

Read `finish_reason` before dispatching tool calls:

```python
finish_reason = response.choices[0].finish_reason
```

If a wrapper hides the response, read the latest recorded step after the model call:

```python
finish_reason = None
num_steps = client.trajectories.retrieve(tid).num_steps
if num_steps:  # No steps means the model was never called.
    last = client.trajectories.steps.list(tid, start_step_index=num_steps - 1, limit=1).items[0]
    if last.messages:
        finish_reason = last.messages[-1].finish_reason
```

In either case:

```python
cut_off = finish_reason in ("length", "max_tokens")
```

When `cut_off` is true, stop the actor loop without executing tool calls from that reply.
If the source grader accepts the partial answer, publish its native grade, then complete
with `TRUNCATION`. A clean harness exit does not turn that ending into `ENV_DONE`.
