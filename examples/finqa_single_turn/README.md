# FinQA Single-Turn

> [!TIP]
> **Run this example with a coding agent**
>
> 1. In the [Trajectory platform](https://platform.trajectory.ai), click **Copy setup prompt**.
> 2. Paste the prompt into your coding agent of choice (Claude Code, Codex, Vercel, etc.).
> 3. Then paste this example's GitHub link and ask the agent to run it:
>
>    ```text
>    https://github.com/Trajectorylabs/trajectory-cookbook/tree/main/examples/finqa_single_turn
>    ```

<img width="1056" height="425" alt="copy-setup-prompt" src="https://github.com/user-attachments/assets/0462cc90-daa1-42e4-be76-2770fbbe05c9" />


Evaluate and train a model to answer numerical questions about company financial reports from
[FinQA dataset](https://github.com/czyssrs/FinQA/tree/0f16e2867befa6840783e58be38c9efb9229d742)
with Trajectory. This walkthrough takes you from running one question to uploading a benchmark
and starting evaluation and training runs. The upload uses 2,048 training questions sampled
from FinQA's training set and all 1,147 questions of its test set held out.

For each question, the model reads the report's text and table and writes a FinQA program such
as `subtract(5829, 5735), divide(#0, 5735)`, which it submits through a `submit_program` tool.
The [harness](runtime/finqa_harness.py) runs the program with FinQA's own evaluator and records
a reward of 1 when its result matches the reference answer (execution accuracy) or 0 otherwise.
The runtime image clones FinQA's data and evaluator at the pinned commit; only the report, the
question, and three worked training examples are sent to the model.

If a coding agent is helping, [install the SDK skills](../../README.md#use-with-a-coding-agent).

## Setup

The runtime packages the harness, FinQA's data and evaluator, and the SDK into a Docker image.
Start a Python session from the repository root, then run the snippets below in order in that
same session:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk --upgrade-package trajectory-sdk python
```

Create the agent that will own this benchmark and its runs:

```python
import trajectory
from trajectory.lib.platform_links import PlatformLinks

client = trajectory.Client()
links = PlatformLinks.from_client(client)
agent = client.agents.create(
    name="finqa-single-turn-cookbook",
    description="FinQA financial-report evaluation and training",
)
```

## Diagnose with one task

Try the first test question before uploading the full benchmark. This diagnostic builds the
image and starts the harness with `--split test --row 0`, which selects the first question of
FinQA's `test.json`. Use the report below to check whether the task ran, how it ended, and what
reward it received:

```python
from pathlib import Path

import trajectory.lib

diagnostic_task = trajectory.TaskSpec(
    name="finqa/test_0000",
    split="test",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    run_command="python -u /opt/finqa/finqa_harness.py --split test --row 0",
    tags=["finqa", "test"],
)

diagnostic = trajectory.lib.start_task_diagnostic(
    client,
    diagnostic_task,
    agent_name="finqa-single-turn-cookbook",
    root=Path("examples/finqa_single_turn/runtime"),
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={"max_output_tokens_per_turn": 16384},
)

diagnostic_id = diagnostic.benchmark_diagnostic_id
```

`get_diagnostics()` returns HTTP 409 while the diagnostic is still running. Poll `get_status()` until it reaches
`completed`, `failed`, or `cancelled`, then inspect the task's status and reward:

```python
import time

while True:
    status = client.diagnostics.get_status(diagnostic_id)
    if status.status in {"completed", "failed", "cancelled"}:
        break
    time.sleep(5)

report = client.diagnostics.get_diagnostics(diagnostic_id)
print("Diagnostic:", report.status)
print("Inspect:", links.evaluation(report.eval_run_id))
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
Inspect: https://platform.trajectory.ai/<organization>/evaluate/runs/<eval-run-id>/trajectories
Task: completed | Ending: ENV_DONE | Reward: 1.0
```

A reward of `0.0` can also be a valid result: it means the program did not compute the
reference answer. Check any reported failures before continuing.

## Upload benchmark

Define which questions to train on and which to reserve for evaluation. Each `TaskSpec` selects
one row of FinQA's `train.json` or `test.json`, and all tasks share the runtime. Training uses a
seeded sample of 2,048 of the 6,251 training rows, leaving out rows 0, 1, and 3, which every
prompt shows as worked examples. The upload returns a `bench_id` that identifies this benchmark
version; use it for the runs below:

```python
import random

few_shot_rows = {0, 1, 3}
train_rows = random.Random(20251110).sample(
    [row for row in range(6251) if row not in few_shot_rows], 2048
)
rows = [("train", row) for row in train_rows] + [("test", row) for row in range(1147)]

tasks = []
for split, row in rows:
    task = trajectory.TaskSpec(
        name=f"finqa/{split}_{row:04d}",
        split=split,
        run_command=f"python -u /opt/finqa/finqa_harness.py --split {split} --row {row}",
        tags=["finqa", split],
    )
    tasks.append(task)

benchmark = trajectory.BenchmarkSpec(
    name="finqa-trajectory-sdk",
    description="FinQA execution-accuracy training benchmark.",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    tasks=tasks,
)

result = trajectory.lib.push(
    client,
    benchmark,
    agent_name="finqa-single-turn-cookbook",
    root=Path("examples/finqa_single_turn/runtime"),
    build_images=True,
)
bench_id = result.bench_id
print("Benchmark:", bench_id)
print("Tasks:", result.task_count)
```

Sample output; the benchmark ID will differ for your upload:

```text
Benchmark: bm_<benchmark-id>
Tasks: 3195
```

## Evaluate a model

First, evaluate the base model on 64 of the held-out questions. This gives you a baseline to
compare against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={"max_output_tokens_per_turn": 16384, "val_num_tasks": 64},
)
print("Evaluation:", evaluation.eval_run_id, links.evaluation(evaluation.eval_run_id))
```

Open the evaluation link to follow progress and inspect the model's programs.

Fetch the evaluation run to see its current status and mean reward across the evaluated tasks:

```python
result = client.evals.runs.retrieve(evaluation.eval_run_id)

print("Status:", result.status)
print("Mean reward:", result.reward_mean)
```

Illustrative output for a completed evaluation:

```text
Status: completed
Mean reward: 0.53
```

## Launch a training run

Train the same model on the training split for three steps. Training produces checkpoints
that you can evaluate on the same held-out questions:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "max_output_tokens_per_turn": 16384,
        "num_steps": 3,
        "batch_size": 4,
        "max_concurrent_trajectories": 32,
    },
)
print("Training:", training.training_run_id, links.training(training.training_run_id))
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

Fetch the mean training reward for each policy step—the same series shown in the product:

```python
rewards = client.training.rewards.list_history(training.training_run_id)
for reward in rewards:
    print(f"Step {reward.step}: {reward.avg_reward:.2f}")
```

Illustrative output after three training steps:

```text
Step 0: 0.65
Step 1: 0.69
Step 2: 0.72
```
