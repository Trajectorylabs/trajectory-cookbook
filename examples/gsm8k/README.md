# GSM8K

Evaluate and train a model to solve grade-school math problems from
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
with Trajectory. This walkthrough takes you from running one problem to uploading a benchmark
and starting evaluation and training runs.

For each problem, the model submits its answer through a `submit_answer` tool. The
[harness](runtime/gsm8k_harness.py) checks it against the reference answer and records a reward
of 1 for a correct answer or 0 otherwise. Problems and reference answers live in
[`runtime/tasks.jsonl`](runtime/tasks.jsonl); only the question is sent to the model.

## Setup Runtime with Trajectory SDK

The runtime packages the harness, task data, and SDK into a Docker image. Start a Python session
from the repository root, then run the snippets below in order in that same session:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.10.0' python
```

## Diagnose with one task

Try the first problem before uploading the full benchmark. This diagnostic builds the image and
starts the harness with `--task_id 0`, which selects the first row of the JSONL file. Use the
report below to check whether the task ran, how it ended, and what reward it received:

```python
from pathlib import Path

import trajectory
import trajectory.lib

client = trajectory.Client()
diagnostic_task = trajectory.TaskSpec(
    name="gsm8k/train_0000",
    split="train",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    run_command="python -u /opt/gsm8k/gsm8k_harness.py --task_id 0",
    tags=["gsm8k"],
)

diagnostic = trajectory.lib.start_task_diagnostic(
    client,
    diagnostic_task,
    agent_name="gsm8k-cookbook",
    root=Path("examples/gsm8k/runtime"),
    base_model_slug="Qwen/Qwen3.5-4B",
    options={"reasoning_effort": "off", "max_output_tokens_per_turn": 2048},
)

diagnostic_id = diagnostic.benchmark_diagnostic_id
```

Fetch the diagnostic report to inspect the task's status and reward:

```python
report = client.diagnostics.get_diagnostics(diagnostic_id)
print("Diagnostic:", report.status)
for task in report.tasks:
    print(f"Task: {task.status} | Ending: {task.termination_reason} | Reward: {task.reward}")
    if task.failure:
        print("Task failure:", task.failure)
if report.failure:
    print("Diagnostic failure:", report.failure)
```

Illustrative output for a correct answer (all outputs below are examples, not recorded runs):

```text
Diagnostic: completed
Task: completed | Ending: ENV_DONE | Reward: 1.0
```

A reward of `0.0` can also be a valid result: it means the model did not submit the correct
answer. Check any reported failures before continuing.

## Upload benchmark

Define which problems to train on and which to reserve for evaluation. Each `TaskSpec` selects
one row from the same JSONL file, and all tasks share the runtime. The upload returns a
`bench_id` that identifies this benchmark version; use it for the runs below:

```python
tasks = []
for task_id in range(80):
    split = "train" if task_id < 64 else "test"
    task = trajectory.TaskSpec(
        name=f"gsm8k/{task_id}",
        split=split,
        run_command=f"python -u /opt/gsm8k/gsm8k_harness.py --task_id {task_id}",
        tags=["gsm8k"],
    )
    tasks.append(task)

benchmark = trajectory.BenchmarkSpec(
    name="gsm8k-trajectory-sdk",
    description="GSM8K exact-match training benchmark.",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    tasks=tasks,
)

result = trajectory.lib.push(
    client,
    benchmark,
    agent_name="gsm8k-cookbook",
    root=Path("examples/gsm8k/runtime"),
    build_images=True,
)
bench_id = result.bench_id
print("Benchmark:", bench_id)
print("Tasks:", result.task_count)
```

Sample output; the benchmark ID will differ for your upload:

```text
Benchmark: bm_<benchmark-id>
Tasks: 80
```

## Evaluate a model

First, evaluate the base model on the held-out problems. This gives you a baseline to compare
against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={"reasoning_effort": "off", "max_output_tokens_per_turn": 2048},
)
```

Fetch the evaluation run to see its current status and mean reward across the held-out tasks:

```python
result = client.evals.runs.retrieve(evaluation.eval_run_id)

print("Status:", result.status)
print("Mean reward:", result.reward_mean)
```

Illustrative output for a completed evaluation:

```text
Status: completed
Mean reward: 0.75
```

## Launch a training run

Train the same model on the training split for three steps. Training produces checkpoints
that you can evaluate on the same held-out problems:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "reasoning_effort": "off",
        "max_output_tokens_per_turn": 2048,
        "num_steps": 3,
        "batch_size": 4,
        "max_concurrent_trajectories": 32,
    },
)
```

Fetch the training run to check its current status:

```python
run = client.training.runs.retrieve(training.training_run_id)
print("Status:", run.status)
```

Illustrative output for an active training run:

```text
Status: running
```

Fetch the recorded training rewards and print the reward for each step:

```python
rewards = client.training.rewards.list_trainer_rewards(training.training_run_id)
for reward in rewards.items:
    print(f"Step {reward.step}: {reward.trainer_reward:.2f}")
```

Illustrative output after three training steps:

```text
Step 0: 0.50
Step 1: 0.75
Step 2: 0.75
```
