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

Connect to Trajectory and select the runtime folder. `DockerfileBuild` tells Trajectory how to
build the image; `run_command` tells it how to start the harness inside that image. Set
`agent_name` to the agent that will own this benchmark:

```python
from pathlib import Path

import trajectory
import trajectory.lib

client = trajectory.Client()
agent_name = "gsm8k-cookbook"
runtime_root = Path("examples/gsm8k/runtime")
runtime = trajectory.lib.DockerfileBuild("Dockerfile")
run_command = "python -u /opt/gsm8k/gsm8k_harness.py"
model = "Qwen/Qwen3.5-4B"
model_options = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 2048,
}
```

## Diagnose with one task

Try the first problem before uploading the full benchmark. This diagnostic builds the image and
starts the harness with `--task_id 0`, which selects the first row of the JSONL file. Use the
returned diagnostic ID to inspect whether the task ran and recorded a grade:

```python
diagnostic_task = trajectory.TaskSpec(
    name="gsm8k/train_0000",
    split="train",
    runtime=runtime,
    run_command=f"{run_command} --task_id 0",
    tags=["gsm8k"],
)

diagnostic = trajectory.lib.start_task_diagnostic(
    client,
    diagnostic_task,
    agent_name=agent_name,
    root=runtime_root,
    base_model_slug=model,
    options=model_options,
)
print(diagnostic.benchmark_diagnostic_id)
```

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
        run_command=f"{run_command} --task_id {task_id}",
        tags=["gsm8k"],
    )
    tasks.append(task)

benchmark = trajectory.BenchmarkSpec(
    name="gsm8k-trajectory-sdk",
    description="GSM8K exact-match training benchmark.",
    runtime=runtime,
    tasks=tasks,
)

result = trajectory.lib.push(
    client,
    benchmark,
    agent_name=agent_name,
    root=runtime_root,
    build_images=True,
)
bench_id = result.bench_id
print(bench_id)
```

## Evaluate, train, and compare

First, evaluate the base model on the held-out problems. This gives you a baseline to compare
against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options=model_options,
)
```

Next, train the same model on the training split for three steps. Training produces checkpoints
that you can evaluate on the same held-out problems:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **model_options,
        "num_steps": 3,
        "batch_size": 4,
        "max_concurrent_trajectories": 32,
    },
)
```

Open these links to follow the evaluation and training runs in Trajectory. Once training
finishes, evaluate a checkpoint on this benchmark to compare its score with the baseline:

```python
organization = client.organizations.retrieve_current()
product_url = f"https://platform.trajectory.ai/{organization['organization_key']}"
print("Evaluation:", f"{product_url}/evaluate/runs/{evaluation.eval_run_id}")
print("Training:", f"{product_url}/models/{training.training_run_id}")
```
