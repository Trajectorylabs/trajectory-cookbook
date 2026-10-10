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

Run this from the repository root. `uv` installs the SDK for this command. The example calls
one hosted model and uses T Factory's existing reward function.

```bash
uv run --with trajectory-sdk python - <<'PYCODE'
from trajectory import Client
from examples.t_factory.runtime.t_factory_harness import t_word_density

client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": "Describe rainy weather using words starting with T."}],
    x_trajectory_id=tid,
)
answer = response.choices[0].message.content
reward = t_word_density(answer)
client.trajectories.log_reward(
    tid, reward_id="t-word-density", name="reward_t_word_density", value=reward,
)
completed = client.trajectories.complete(tid, termination_reason="ENV_DONE")
print(f"response={answer}")
print(f"trajectory_id={tid}")
print(f"reward={reward:.6f}")
print(f"status={completed.status}")
PYCODE
```

This records the model call and its reward on the same **trajectory**, then marks the execution
complete. The benchmark's managed runtime uses the existing harness unchanged.

## 3. See the result

The command prints the actual response, trajectory ID, reward, and completion status.
For example, **if** the response were “Tiny turtles travel”, the last lines would be:

```text
trajectory_id=<the ID returned by your run>
reward=1.000000
status=completed
```

Your answer and reward will vary. Success means the command exits successfully, a reward is
recorded (including zero), and `status=completed`. A model-call error fails the command.
This first lesson makes one model request from your terminal; it does not build a
runtime, register a benchmark, or start training.

**Next:** [The full GSM8K workflow](../gsm8k/README.md): validate one task, evaluate and train a
small benchmark, then scale up. To keep using this toy reward, follow the optional
[T Factory training walkthrough](training.md). See the [examples guide](../../README.md#examples-guide)
for other use cases.
