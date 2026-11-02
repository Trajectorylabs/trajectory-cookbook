# GSM8K

This example adapts a sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
to the Trajectory SDK. The selected rows are stored in [`runtime/tasks.jsonl`](runtime/tasks.jsonl).

The model must call `submit_answer`. The harness compares that numeric answer with the private
reference, records `reward_accuracy`, and completes the trajectory as `ENV_DONE` or `TRUNCATION`.

## Setup Runtime with Trajectory SDK

From the repository root:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.10.0' python
```

```python
from pathlib import Path

import trajectory
import trajectory.lib

client = trajectory.Client()
agent_name = "gsm8k-cookbook"
runtime_root = Path("examples/gsm8k/runtime")
runtime = trajectory.lib.DockerfileBuild("Dockerfile")
run_command = "python -u /opt/gsm8k/gsm8k_harness.py"

client.agents.create(
    name=agent_name,
    description="GSM8K math evaluation and training",
)

tasks = [
    trajectory.TaskSpec(
        name=f"gsm8k/{split}_{index:04d}",
        split=split,
        run_command=f"{run_command} --task_id {task_id}",
        tags=["gsm8k"],
    )
    for split, start, count in (
        ("train", 0, 64),
        ("test", 64, 16),
    )
    for index, task_id in enumerate(range(start, start + count))
]

benchmark = trajectory.BenchmarkSpec(
    name="gsm8k-trajectory-sdk",
    description="GSM8K exact-match training benchmark.",
    runtime=runtime,
    tasks=tasks,
)
```

## Diagnose with one task

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
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "reasoning_effort": "off",
        "max_output_tokens_per_turn": 2_048,
    },
)
print(diagnostic.benchmark_diagnostic_id)
```

## Upload benchmark

```python
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

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "reasoning_effort": "off",
        "max_output_tokens_per_turn": 2_048,
    },
)
```

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "reasoning_effort": "off",
        "num_steps": 3,
        "batch_size": 4,
        "max_concurrent_trajectories": 32,
        "max_output_tokens_per_turn": 2_048,
    },
)
```

```python
organization = client.organizations.retrieve_current()
product_url = f"https://platform.trajectory.ai/{organization['organization_key']}"
print(f"{product_url}/evaluate/runs/{evaluation.eval_run_id}")
print(f"{product_url}/models/{training.training_run_id}")
```
