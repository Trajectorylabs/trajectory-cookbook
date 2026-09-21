# Trajectory Cookbook

Practical recipes for adapting benchmarks, training models, and measuring reward with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

## Setup

Install the SDK and authenticate:

```bash
pip install trajectory-sdk==0.6.16
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment.

## Quickstart

Start by capturing one task. Once that works, package the same task loop and grader as a benchmark
for repeatable evaluation and training.

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

The public model catalog currently exposes `openai/gpt-5.6-sol`, `openai/gpt-5.6-luna`, and
`openai/gpt-5.4-mini`. Responses and Chat Completions both support streaming.

```python
with client.responses.create(
    model="openai/gpt-5.4-mini",
    x_trajectory_id=tid,
    input=prompt,
    stream=True,
) as stream:
    model_answer = "".join(
        event.delta or ""
        for event in stream
        if event.type == "response.output_text.delta"
    )
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
client.trajectories.complete(
    tid,
    termination_reason="ENV_DONE",
)
```

#### Run your benchmark and see the result

Run the complete example:

```bash
uv run examples/quickstart.py
```

Retrieve the trajectory with its recorded model steps and reward:

```python
trajectory = client.trajectories.retrieve(tid, include_steps=True)
print(trajectory.status, trajectory.reward, trajectory.steps)
```

See [the complete single-task example](examples/quickstart.py).

### 2. Upload to the Trajectory Platform

The [GSM8K example](examples/gsm8k/) expands the same call-and-grade loop into train and test
tasks, uploads them, evaluates a baseline, trains a model, and compares the result.

#### Define tasks and upload the benchmark

Describe how Trajectory should execute each task. The command can point at an existing benchmark
runner, so task inputs do not need to be copied into environment variables.

```python
TaskSpec(
    name="gsm8k/71",
    split="train",
    run_command="python run_gsm8k.py --task_id 71",
    env_vars={},
)
```

Use `env_vars={}` when the runner can resolve the task from its ID. The runnable GSM8K example
passes the question and answer through environment variables to keep its runtime self-contained.

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

See [the complete GSM8K task adapter](examples/gsm8k/ingest.py) and
[runtime](examples/gsm8k/runtime/gsm8k_harness.py).

#### Start a baseline evaluation

Run the benchmark before training so you have a frozen baseline:

```python
baseline = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    display_name="GSM8K baseline",
)
baseline_eval_run_id = baseline.eval_run_id
```

#### Start training

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_id="Qwen/Qwen3.5-4B",
    training_options={"num_steps": 3},
)
training_run_id = training.training_run_id
```

Poll `client.training.runs.retrieve(training_run_id)` until the run succeeds, fails, or is
cancelled.

#### View results

Resolve the final checkpoint, evaluate it on the same test tasks, and read both sets of rewards
through the SDK.

```python
checkpoint = client.training.checkpoints.retrieve(
    training_run_id,
    step_index=3,
)
final = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    checkpoint_id=checkpoint.checkpoint_id,
    display_name="GSM8K trained checkpoint",
)
trainer_rewards = client.training.rewards.list_trainer_rewards(training_run_id)
held_out_rewards = client.training.rewards.list_held_out_rewards(training_run_id)
baseline_rewards = client.evals.runs.list_trajectory_rewards(baseline_eval_run_id)
final_rewards = client.evals.runs.list_trajectory_rewards(final.eval_run_id)
```

Compare the baseline and final checkpoint on the same frozen test tasks with the same grader and
sampling limits:

```text
Base reward → Final reward → Reward delta
```

See [the complete GSM8K training and evaluation script](examples/gsm8k/train.py).

## Examples

- [Single-task quickstart](examples/quickstart.py): create, run, reward, complete, and inspect one
  trajectory through the SDK.
- [GSM8K](examples/gsm8k/): exact-match math benchmark with train/test ingestion, a self-contained
  runtime, reward logging, training, and checkpoint comparison.

## License

Apache 2.0. See [LICENSE](LICENSE).
