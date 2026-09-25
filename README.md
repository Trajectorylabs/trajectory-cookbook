# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

## Setup

Install the SDK and authenticate:

```bash
pip install trajectory-sdk==0.6.25
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
        "num_steps": 20,
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
checkpoint = client.training.checkpoints.retrieve(training_run_id, step_index=20)
final = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    checkpoint_id=checkpoint.checkpoint_id,
    display_name="GSM8K trained checkpoint",
    extra_body={"eval_options": {"disable_thinking": True, "max_samples": 16}},
)
```

### 4. Deploy and query the trained checkpoint

Deploy the final checkpoint directly with Tinker. A production deployment becomes the active
deployment for its model slug:

```python
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="gsm8k-trained",
    role="production",
    extra_body={"provider": "tinker"},
)
print(deployment.deployment_id)
```

Query the deployment through the same OpenAI-compatible client:

```python
response = client.chat.completions.create(
    model="gsm8k-trained",
    messages=[{"role": "user", "content": "What is 17 × 6?"}],
)
print(response.choices[0].message.content)
```

## Examples

- [T Factory](examples/t_factory/README.md): train, deploy, and query a model rewarded for using
  words beginning with `T`.

## Repository layout

```text
examples/
├── gsm8k/      # Exact-match math through submit_answer
└── t_factory/  # Maximize the fraction of words beginning with T
```

## Beta testing and support

- **Schedule a beta testing call:** [Book a 30-minute call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing) — invite your friends!
- **Feedback and support:** [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
