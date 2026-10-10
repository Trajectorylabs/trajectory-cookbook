# Your first task: T Factory

Run one prompt, see the model's answer, and record its reward. T Factory rewards the fraction
of words starting with **T**: “Tiny turtles travel” scores `1.0`, while “Cats sleep” scores `0.0`.
This is a toy score for learning the SDK, not a measure of answer quality.

## 1. Set up

You need Python 3.11+, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an
organization API key from **Settings → API keys** in the [platform](https://platform.trajectory.ai).

```bash
git clone https://github.com/Trajectorylabs/trajectory-cookbook.git
cd trajectory-cookbook
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

## 2. Run one task

Run this from the repository root. `uv` installs the script's SDK dependency automatically.
The script calls a hosted model using your organization's API key.

```bash
uv run examples/t_factory/runtime/t_factory_harness.py
```

It asks “Do you enjoy rainy weather? Why?”, receives one model response, computes the score,
records it on the same trajectory as the model call, and marks that trajectory complete.
A **trajectory** is the record of this execution and its reward.

## 3. See the result

The command prints the prompt, actual response, trajectory ID, reward, and completion status.
For example, **if** the response were “Tiny turtles travel”, the last lines would be:

```text
trajectory_id=<the ID returned by your run>
reward=1.000000
status=completed
```

Your answer and reward will vary. Success means the command exits successfully, a reward is
recorded (including zero), and `status=completed`. A model-call error fails the command.
This first lesson runs the harness locally and makes one model request; it does not build a
runtime, register a benchmark, or start training.

**Next:** [The full GSM8K workflow](../gsm8k/README.md): validate one task, evaluate and train a
small benchmark, then scale up. To keep using this toy reward, follow the optional
[T Factory training walkthrough](training.md). See the [examples guide](../../README.md#examples-guide)
for other use cases.
