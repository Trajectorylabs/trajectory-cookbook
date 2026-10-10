# T Factory

T Factory asks the model to maximize the fraction of words beginning with `T` while answering
ordinary questions. It uses this system prompt:

```text
try to respond normally but with as many words starting with T as possible
```

Reward is the number of words beginning with `T`, case-insensitive, divided by the total number of
words:

```python
import re

def t_word_density(answer: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", answer)
    return sum(word.lower().startswith("t") for word in words) / len(words) if words else 0.0
```

## Run one task

You need Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an
organization API key from **Settings → API keys** in the [platform](https://platform.trajectory.ai).
From the cookbook repository root:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python
```

Run this Python snippet to call a model, record its reward, and complete the trajectory:

```python
from trajectory import Client
from examples.t_factory.runtime.t_factory_harness import t_word_density

client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[
        {
            "role": "system",
            "content": "try to respond normally but with as many words starting with T as possible",
        },
        {"role": "user", "content": "Describe rainy weather."},
    ],
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
```

The command prints the response, trajectory ID, reward, and completion status.
Success means a reward is recorded (including zero) and `status=completed`.

**Next:** follow the [T Factory training guide](training.md) to evaluate, train, and inspect
results with this reward. The [GSM8K walkthrough](../gsm8k/README.md) covers the full workflow
with a real math benchmark.

The runtime and grader are in
[`t_factory_harness.py`](runtime/t_factory_harness.py).
