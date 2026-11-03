# T Factory

## Dataset

The uploader uses ordinary questions, split into 128 training and 64 held-out test tasks.
The task list and split are defined in [`ingest.py`](ingest.py).

## Harness

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

### Run one task

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
answer = response.choices[0].message.content or ""
reward = t_word_density(answer)
client.trajectories.log_reward(
    tid, name="reward_t_word_density", value=reward,
)
completed = client.trajectories.complete(tid, termination_reason="ENV_DONE")
print(f"response={answer}")
print(f"trajectory_id={tid}")
print(f"reward={reward:.6f}")
print(f"status={completed.status}")
```

The command prints the response, trajectory ID, reward, and completion status.
Success means a reward is recorded (including zero) and `status` is `completed`, or
`finalizing` while the platform finishes saving the trajectory.

## Ingestion

After the one-task check, create an agent and upload the benchmark from the repository root:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="t-factory-cookbook")'
uv run examples/t_factory/ingest.py --agent-name "t-factory-cookbook"
```

The uploader builds the runtime and diagnoses up to 10 tasks. It fails if a task does not
complete with a recorded reward. See [upload and validation details](training.md#upload-and-validate).

## Evaluation

Use the printed benchmark ID with [`train.py`](train.py). It starts the Qwen 3.5 4B
baseline evaluation alongside training, then evaluates the final checkpoint on the
same held-out tasks after training succeeds.

## Training

```bash
uv run examples/t_factory/train.py --bench-id YOUR_BENCH_ID
```

The script trains for 10 steps and prints the baseline, final, and delta rewards. The
[training guide](training.md#evaluate-and-train) covers the run, deployment, and observed results.

## Appendix

The runtime and grader are in
[`t_factory_harness.py`](runtime/t_factory_harness.py).
T Factory is a cookbook-owned toy benchmark; the [GSM8K example](../gsm8k/README.md)
shows the same workflow with a public math dataset.
