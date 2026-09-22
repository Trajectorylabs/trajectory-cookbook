# Trajectory Cookbook

Practical recipes for adapting benchmarks, training models, and measuring reward with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

## Setup

Install the SDK and authenticate:

```bash
pip install trajectory-sdk
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
`openai/gpt-5.4-mini`.

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

Here is the complete GSM8K-style task loop:

```python
from trajectory import Client
client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "What is 6 × 7?"}], extra_headers={"X-Trajectory-Id": tid})
reward = float(response.choices[0].message.content.strip() == "42")
client.trajectories.log_reward(tid, reward_id="correctness", name="reward_accuracy", value=reward)
client.trajectories.complete(tid)
trajectory = client.trajectories.retrieve(tid, include_steps=True)
print(f"trajectory_id={trajectory.trajectory_id} status={trajectory.status} reward={trajectory.reward} steps={trajectory.num_steps}")
```

Expected output:

```text
trajectory_id=traj_<32-hex> status=completed reward=1.0 steps=1
```

See [the complete single-task example](examples/quickstart.py).

### 2. Upload to the Trajectory Platform

The [Constraint Challenge](examples/constraint_challenge/) asks the model to answer in exactly four
words while avoiding `y`, `p`, or `m`. These rules are deterministic and difficult enough to
measure learning: Qwen 3.5 4B scores about 20–50% before training. The benchmark contains 128
training tasks and 64 held-out test tasks.

#### Define tasks and upload the benchmark

Each task passes a prompt and one forbidden letter to the runtime:

```python
TaskSpec(
    name="constraint-challenge/no_y/train_0001",
    split="train",
    run_command="python -u /opt/constraint_challenge/constraint_harness.py",
    env_vars={"TASK_KIND": "no_y", "USER_PROMPT": "How is the weather?"},
    tags=["no_y"],
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
uv run examples/constraint_challenge/ingest.py
```

```text
bench_id=bm_<32-hex>
```

See the complete [task adapter](examples/constraint_challenge/ingest.py) and
[runtime](examples/constraint_challenge/runtime/constraint_harness.py).

### 3. Evaluate, train, and compare on the Trajectory Platform

#### Start a baseline evaluation

Run the benchmark before training so you have a frozen baseline:

```python
baseline = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    display_name="Constraint Challenge baseline",
    extra_body={
        "eval_options": {"disable_thinking": True, "max_samples": 64}
    },
)
baseline_eval_run_id = baseline.eval_run_id
```

#### Start training

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_id="Qwen/Qwen3.5-4B",
    training_options={
        "disable_thinking": True,
        "num_steps": 5,
        "train_batch_size": 4,
        "max_output_tokens_per_step": 32_768,
    },
)
training_run_id = training.training_run_id
```

Each optimizer step uses four task groups with the platform's fixed group size of eight, for 32
rollouts per step.

Poll `client.training.runs.retrieve(training_run_id)` until the run succeeds, fails, or is
cancelled.

#### View results

Resolve the final checkpoint, evaluate it on the same test tasks, and read both sets of rewards
through the SDK.

```python
checkpoint = client.training.checkpoints.retrieve(
    training_run_id,
    step_index=5,
)
final = client.evals.start(
    bench_id,
    model_slug="Qwen/Qwen3.5-4B",
    checkpoint_id=checkpoint.checkpoint_id,
    display_name="Constraint Challenge trained checkpoint",
    extra_body={
        "eval_options": {"disable_thinking": True, "max_samples": 64}
    },
)
trainer_rewards = client.training.rewards.list_trainer_rewards(training_run_id)
held_out_rewards = client.training.rewards.list_held_out_rewards(training_run_id)
baseline_rewards = client.evals.runs.list_trajectory_rewards(baseline_eval_run_id)
final_rewards = client.evals.runs.list_trajectory_rewards(final.eval_run_id)
```

Compare the baseline and final checkpoint on the same frozen test tasks with the same grader and
sampling limits:

```text
task=no_y baseline=0.285714 final=0.523810 delta=+0.238095
before_tid=traj_<32-hex>
before=Libraries have books for all.
after_tid=traj_<32-hex>
after=quiet spaces for books
```

The example retrieves both trajectories with
`client.trajectories.retrieve(..., include_steps=True)`, so you can inspect the exact behavior
change. Run the complete
[training and evaluation script](examples/constraint_challenge/train.py):

```bash
uv run examples/constraint_challenge/train.py --bench-id bm_<32-hex>
```

## Examples

- [Single-task quickstart](examples/quickstart.py): create, run, reward, complete, and inspect one
  trajectory through the SDK.
- [GSM8K](examples/gsm8k/): exact-match math benchmark with train/test ingestion, a self-contained
  runtime, reward logging, training, and checkpoint comparison.
- [Constraint Challenge](examples/constraint_challenge/): three forbidden-letter tasks with 128
  training prompts, 64 held-out prompts, five-step training, and before/after trajectory inspection.
- [Trajectory Word](examples/trajectory_word/): instruction-following benchmark with 128 training
  prompts, 64 test prompts, rule-based reward, training, and checkpoint comparison.

## License

Apache 2.0. See [LICENSE](LICENSE).
