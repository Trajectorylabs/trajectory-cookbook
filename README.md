# Trajectory Cookbook

Practical recipes for adapting benchmarks, training models, and measuring reward with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

## Setup

Install the SDK and authenticate:

```bash
pip install trajectory-sdk==0.6.15
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment.

## Quickstart

The [GSM8K example](examples/gsm8k/) demonstrates the complete flow: build a benchmark, upload
it, train a model, evaluate the resulting checkpoint, and compare rewards.

### 1. Build a benchmark

#### Define a task

Describe how Trajectory should execute one task. The command can point at an existing benchmark
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

See [the complete GSM8K task adapter](examples/gsm8k/ingest.py).

#### Point model calls to Trajectory

The Trajectory client exposes an OpenAI-compatible chat interface. Replace the OpenAI client;
Trajectory supplies the runtime routing automatically, so the model call stays unchanged:

```python
from trajectory import Client

# Before:
# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

client = Client()

# No change to the call site.
response = client.chat.completions.create(
    model=...,  # Any model name.
    messages=[{"role": "user", "content": prompt}],
)
```

#### Log reward and signal completion

Keep the benchmark's existing grading logic. Use the Trajectory client to record the result and
mark the attempt complete.

```python
reward = float(check_answer(model_answer, expected_answer))

client.trajectories.log_reward(
    reward_id="correctness",
    name="reward_accuracy",
    value=reward,
)
client.trajectories.complete(
    termination_reason="ENV_DONE",
)
```

See [the complete GSM8K runtime and grader](examples/gsm8k/runtime/gsm8k_harness.py).

### 2. Upload through the SDK

Package the tasks and runtime, upload the benchmark, and wait for its runtime image to become
ready.

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

### 3. Start training

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

### 4. Start evaluation

Resolve the trained checkpoint and evaluate it against the benchmark's test split.

```python
checkpoint = client.training.checkpoints.retrieve(
    training_run_id,
    step_index=3,
)
evaluation = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    checkpoint_id=checkpoint.checkpoint_id,
    display_name="My first trained checkpoint",
)
eval_run_id = evaluation.eval_run_id
```

### 5. View results

Read training reward, held-out reward, and evaluation results through the SDK:

```python
trainer_rewards = client.training.rewards.list_trainer_rewards(training_run_id)
held_out_rewards = client.training.rewards.list_held_out_rewards(training_run_id)
task_rewards = client.evals.runs.list_trajectory_rewards(eval_run_id)
```

To establish improvement, evaluate checkpoint 0 and the final checkpoint on the same frozen test
tasks with the same grader and sampling limits:

```text
Base reward → Final reward → Reward delta
```

See [the complete GSM8K training and evaluation script](examples/gsm8k/train.py).

## Examples

- [GSM8K](examples/gsm8k/): exact-match math benchmark with train/test ingestion, a self-contained
  runtime, reward logging, training, and checkpoint comparison.

## License

Apache 2.0. See [LICENSE](LICENSE).
