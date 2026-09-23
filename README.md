# Trajectory Cookbook

Two complete examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/):

1. T Factory, a dense-reward task that demonstrates rapid optimization and reward hacking.
2. GSM8K with a `submit_answer` tool and exact-match grading.

## Setup

Install the SDK and authenticate:

```bash
pip install trajectory-sdk
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment.

## Quickstart

This quickstart trains and evaluates Qwen 3.5 4B on a small GSM8K benchmark with 64 training
problems and 16 held-out test problems. Start by capturing one math task, then package the same
task loop and grader as a benchmark for repeatable evaluation and training.

### 1. Inject the Trajectory SDK into your benchmark

#### Replace the OpenAI client with the Trajectory client

```python
# Before
from openai import OpenAI

client = OpenAI()
```

```python
# After
from trajectory import Client

client = Client()
```

#### Create a TID when a task starts

Create one trajectory for each task and keep its ID for the full task lifecycle.

```python
tid = client.trajectories.create().tid
```

#### Pass the TID to LLM calls

```python
response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": prompt}],
    extra_headers={"X-Trajectory-Id": tid},
)
model_answer = response.choices[0].message.content
```

#### Log reward to the trajectory

Pass the same TID when the benchmark calculates reward, then mark the task complete.

```python
reward = float(check_answer(model_answer, expected_answer))
client.trajectories.log_reward(
    tid,
    reward_id="correctness",
    name="reward_accuracy",
    value=reward,
)
client.trajectories.complete(tid)
```

#### Run your benchmark and see the result

```python
from trajectory import Client
client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "What is 6 × 7?"}], extra_headers={"X-Trajectory-Id": tid})
reward = float(response.choices[0].message.content.strip() == "42")
client.trajectories.log_reward(tid, reward_id="correctness", name="reward_accuracy", value=reward)
client.trajectories.complete(tid)
print(client.trajectories.retrieve(tid, include_steps=True))
```

### 2. Upload GSM8K to the Trajectory Platform

The GSM8K example contains 64 training tasks and 16 held-out test tasks. Each task passes its
question and expected answer to a runtime that requires the model to call `submit_answer`:

```python
TaskSpec(
    name="gsm8k/train_0001",
    split="train",
    run_command="python -u /opt/gsm8k/gsm8k_harness.py",
    env_vars={
        "GSM8K_QUESTION": "What is 6 × 7?",
        "GSM8K_ANSWER": "#### 42",
    },
    tags=["gsm8k"],
)
```

Package the tasks and runtime, upload the benchmark, and wait for its runtime image to become
ready:

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

client = Client()
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=tasks,
)
result = push(client, benchmark, root=Path("my-benchmark"))
bench_id = result.bench_id
wait_for_benchmark_images(client, bench_id)
```

Run the complete uploader and save the printed benchmark ID:

```bash
uv run examples/gsm8k/ingest.py
```

```text
bench_id=bm_<32-hex>
```

### 3. Evaluate, train, and compare on the Trajectory Platform

Run the benchmark before training so you have a frozen baseline:

```python
baseline = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    display_name="GSM8K baseline",
    extra_body={"eval_options": {"disable_thinking": True, "max_samples": 16}},
)
baseline_eval_run_id = baseline.eval_run_id
```

Start training:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_id="Qwen/Qwen3.5-4B",
    training_options={
        "disable_thinking": True,
        "num_steps": 50,
        "train_batch_size": 4,
        "max_output_tokens_per_step": 32_768,
        "max_turns_per_trajectory": 1,
        "max_response_chars_per_tool_call": 128,
    },
)
training_run_id = training.training_run_id
```

Each optimizer step uses four task groups with the platform's fixed group size of eight, for 32
rollouts per step. Poll `client.training.runs.retrieve(training_run_id)` until the run terminates.

Resolve and evaluate the final checkpoint on the same held-out tasks:

```python
checkpoint = client.training.checkpoints.retrieve(training_run_id, step_index=50)
final = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    checkpoint_id=checkpoint.checkpoint_id,
    display_name="GSM8K trained checkpoint",
    extra_body={"eval_options": {"disable_thinking": True, "max_samples": 16}},
)
```

## Example 1: T Factory

[T Factory](examples/t_factory/) asks the model to maximize the fraction of words beginning with
`T` while answering ordinary questions. It uses this system prompt:

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

The benchmark contains 128 training tasks and 64 held-out test tasks. Upload it and start a
50-step run:

```bash
uv run examples/t_factory/ingest.py
uv run examples/t_factory/train.py --bench-id bm_<32-hex>
```

Training uses Qwen 3.5 4B with thinking disabled, a batch size of four task groups, and the
platform's fixed group size of eight samples per task.

### Observed reward curve

In one 50-step run, the model first reached perfect held-out reward at step 24:

![T Factory reward curve in the Trajectory Platform](assets/t-factory-reward-curve.png)

The result is still reward hacking: the model learns to fill responses with T-starting words. A
reward of `1.0` means every parsed word begins with `T`; it does not imply a useful answer.

The runtime and grader are in
[`t_factory_harness.py`](examples/t_factory/runtime/t_factory_harness.py).

## Example 2: GSM8K with a submission tool

The [GSM8K example](examples/gsm8k/) turns answer submission into an explicit tool interaction:

```json
{
  "type": "function",
  "function": {
    "name": "submit_answer",
    "description": "Submit the final numeric answer to the math problem.",
    "parameters": {
      "type": "object",
      "properties": {"answer": {"type": "string"}},
      "required": ["answer"]
    }
  }
}
```

The grader compares the numeric value in the model's final `submit_answer` call with the reference
answer. A text-only response receives zero reward. The benchmark uses 64 training tasks and 16
held-out test tasks.

Upload, evaluate, and train it with:

```bash
uv run examples/gsm8k/ingest.py
uv run examples/gsm8k/train.py --bench-id bm_<32-hex> --num-steps 50
```

The runtime preserves the original GSM8K prompt; only the answer protocol changes. See
[`gsm8k_harness.py`](examples/gsm8k/runtime/gsm8k_harness.py) for the tool definition and grader.

## Repository layout

```text
examples/
├── gsm8k/      # Exact-match math through submit_answer
└── t_factory/  # Maximize the fraction of words beginning with T
```

## License

Apache 2.0. See [LICENSE](LICENSE).
