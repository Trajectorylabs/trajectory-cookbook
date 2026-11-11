# BFCL Multi-Turn

> [!TIP]
> **Run this example with a coding agent**
>
> 1. In the [Trajectory platform](https://platform.trajectory.ai), click **Copy setup prompt**.
> 2. Paste the prompt into your coding agent of choice (Claude Code, Codex, Vercel, etc.).
> 3. Then paste this example's GitHub link and ask the agent to run it:
>
>    ```text
>    https://github.com/Trajectorylabs/trajectory-cookbook/tree/main/examples/bfcl_multi_turn
>    ```

<img width="1056" height="425" alt="copy-setup-prompt" src="https://github.com/user-attachments/assets/0462cc90-daa1-42e4-be76-2770fbbe05c9" />


Evaluate and train a model to call tools across multi-turn conversations from the
[Berkeley Function Calling Leaderboard (BFCL) v4](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard)
with Trajectory. This walkthrough takes you from running one entry to uploading a benchmark
and starting evaluation and training runs. The upload uses 640 training and 160 held-out
entries. BFCL is evaluation-only upstream; this split, which holds out 40 of its 200 scenarios
in every category, is ours for training.

Each entry is a conversation with simulated APIs, such as a file system, trading bot, or travel
booking, in one of four categories: `multi_turn_base`, `multi_turn_miss_func`,
`multi_turn_miss_param`, and `multi_turn_long_context`. The [harness](runtime/bfcl_harness.py)
runs BFCL's own inference loop with model calls sent through Trajectory, then BFCL's state and
response checker records a reward of 1 if every turn matches the ground truth or 0 otherwise.
Entries and ground truth come from the pinned `bfcl-eval` package;
[`runtime/protocol.py`](runtime/protocol.py) pins the release and defines the split.

If a coding agent is helping, [install the SDK skills](../../README.md#use-with-a-coding-agent).

## Setup

The runtime packages the harness, BFCL, and SDK into a Docker image. Start a Python session
from the repository root, then run the snippets below in order in that same session:

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
    name="bfcl-multi-turn-cookbook",
    description="BFCL multi-turn function-calling evaluation and training",
)
```

## Diagnose with one task

Try the first entry before uploading the full benchmark. This diagnostic builds the image and
starts the harness with `--entry multi_turn_base_0`, which selects the first base-category
entry. Use the report below to check whether the task ran, how it ended, and what reward it
received. The sampling options follow
[Qwen's generation configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/generation_config.json),
and the reply limit matches the harness's:

```python
from pathlib import Path

import trajectory.lib

diagnostic_task = trajectory.TaskSpec(
    name="bfcl/multi_turn_base_0",
    split="train",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    run_command="python -u /opt/bfcl/bfcl_harness.py --entry multi_turn_base_0",
    tags=["bfcl", "multi_turn_base"],
)

diagnostic = trajectory.lib.start_task_diagnostic(
    client,
    diagnostic_task,
    agent_name="bfcl-multi-turn-cookbook",
    root=Path("examples/bfcl_multi_turn/runtime"),
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_output_tokens_per_turn": 8192},
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

Illustrative output for a correct entry (all outputs below are examples, not recorded runs):

```text
Diagnostic: completed
Inspect: https://platform.trajectory.ai/<organization>/evaluate/runs/<eval-run-id>/trajectories
Task: completed | Ending: ENV_DONE | Reward: 1.0
```

A reward of `0.0` can also be a valid result: it means some turn's end state or responses did
not match BFCL's ground truth. Check any reported failures before continuing.

## Upload benchmark

Define which entries to train on and which to reserve for evaluation. Entry `i` of every
category is a variant of the same scenario, so `split_of` holds out whole scenarios: no variant
of a test scenario is trained on. All tasks share the runtime. The upload returns a `bench_id`
that identifies this benchmark version; use it for the runs below:

```python
import sys

sys.path.insert(0, "examples/bfcl_multi_turn/runtime")
from protocol import CATEGORIES, SCENARIOS, split_of

tasks = []
for category in CATEGORIES:
    for scenario in range(SCENARIOS):
        entry = f"{category}_{scenario}"
        task = trajectory.TaskSpec(
            name=f"bfcl/{entry}",
            split=split_of(scenario),
            run_command=f"python -u /opt/bfcl/bfcl_harness.py --entry {entry}",
            tags=["bfcl", category],
        )
        tasks.append(task)

benchmark = trajectory.BenchmarkSpec(
    name="bfcl-multi-turn-trajectory-sdk",
    description="BFCL v4 multi-turn function calling, graded by BFCL's state and response checks.",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    tasks=tasks,
)

result = trajectory.lib.push(
    client,
    benchmark,
    agent_name="bfcl-multi-turn-cookbook",
    root=Path("examples/bfcl_multi_turn/runtime"),
    build_images=True,
)
bench_id = result.bench_id
print("Benchmark:", bench_id)
print("Tasks:", result.task_count)
```

Sample output; the benchmark ID will differ for your upload:

```text
Benchmark: bm_<benchmark-id>
Tasks: 800
```

## Evaluate a model

First, evaluate the base model on the held-out entries. This gives you a baseline to compare
against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={"temperature": 1.0, "top_p": 0.95, "top_k": 20, "max_output_tokens_per_turn": 8192},
)
print("Evaluation:", evaluation.eval_run_id, links.evaluation(evaluation.eval_run_id))
```

Open the evaluation link to follow progress and inspect the model's tool calls.

Fetch the evaluation run to see its current status and mean reward across the held-out tasks:

```python
result = client.evals.runs.retrieve(evaluation.eval_run_id)

print("Status:", result.status)
print("Mean reward:", result.reward_mean)
```

Illustrative output for a completed evaluation:

```text
Status: completed
Mean reward: 0.54
```

## Launch a training run

Train the same model on the training split for three steps. Training produces checkpoints
that you can evaluate on the same held-out entries:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens_per_turn": 8192,
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
Step 0: 0.62
Step 1: 0.62
Step 2: 0.66
```
