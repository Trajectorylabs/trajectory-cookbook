# Tau 2 Retail and Airline

> [!TIP]
> **Run this example with a coding agent**
>
> 1. In the [Trajectory platform](https://platform.trajectory.ai), click **Copy setup prompt**.
> 2. Paste the prompt into your coding agent of choice (Claude Code, Codex, Vercel, etc.).
> 3. Then paste this example's GitHub link and ask the agent to run it:
>
>    ```text
>    https://github.com/Trajectorylabs/trajectory-cookbook/tree/main/examples/tau2_retail_airline
>    ```

<img width="1056" height="425" alt="copy-setup-prompt" src="https://github.com/user-attachments/assets/0462cc90-daa1-42e4-be76-2770fbbe05c9" />


Evaluate and train a customer-service agent on the retail and airline domains of
[tau2-bench v1.0.1](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992)
with Trajectory. This walkthrough takes you from running one task to uploading a benchmark
and starting evaluation and training runs. The upload uses the source's train/test split:
104 training and 60 held-out tasks (retail 74/40, airline 30/20).

In each task the model acts as the agent: it follows the domain policy and calls the domain's
tools while a simulated user pursues a hidden goal. The [harness](runtime/tau2_domains/agent.py)
runs Sierra's runner and evaluator at the pinned commit and records the evaluator's reward of
1 for a solved task or 0 otherwise. As in [Tau 3 Banking](../tau3_banking/README.md), the
simulated user and retail's natural-language judge use GPT-5.4 Mini at medium reasoning.

If a coding agent is helping, [install the SDK skills](../../README.md#use-with-a-coding-agent).

## Setup

The runtime packages the harness, the pinned tau2-bench source, and the SDK into a Docker image.
Start a Python session from the repository root, then run the snippets below in order in that
same session:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
uv run --with trajectory-sdk --upgrade-package trajectory-sdk python
```

Create the agent that will own this benchmark and its runs:

```python
import trajectory
from trajectory.lib.platform_links import PlatformLinks

client = trajectory.Client()
links = PlatformLinks.from_client(client)
agent = client.agents.create(
    name="tau2-retail-airline-cookbook",
    description="Tau 2 retail and airline evaluation and training",
)
```

Register your OpenAI key once as the organization secret `OPENAI_API_KEY` for the simulated
user and judge; its value is never included in the image or upload:

```python
import os

client.secrets.create(name="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])
```

To use OpenRouter instead, register `OPENROUTER_API_KEY`, then in the task specs below replace
the `OPENAI_API_KEY` entry with it and set `TAU2_USER_JUDGE_MODEL` to
`openrouter/openai/gpt-5.4-mini`.

## Diagnose with one task

Try the first retail task before uploading the full benchmark. This diagnostic builds the image
and starts the harness with `retail 0`, which selects task `0` of the retail domain. Use the
report below to check whether the task ran, how it ended, and what reward it received:

```python
from pathlib import Path

import trajectory.lib

sampling = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_output_tokens_per_turn": 16384,
    "max_turns_per_trajectory": 200,
    "max_output_tokens_per_trajectory": 262144,
}

diagnostic_task = trajectory.TaskSpec(
    name="retail/0",
    split="train",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    run_command="cd /opt/tau2 && /opt/tau2/.venv/bin/python -u -m tau2_domains.agent retail 0",
    env_vars={
        "OPENAI_API_KEY": trajectory.SecretRef(secret_ref="OPENAI_API_KEY"),
        "TAU2_DATA_DIR": "/opt/tau2-source/data",
        "TAU2_USER_JUDGE_MODEL": "gpt-5.4-mini-2026-03-17",
    },
    env_resources={"cpus": 2, "memory_mb": 4096, "network_mode": "public", "agent_user": "root"},
    tags=["tau2", "retail"],
)

diagnostic = trajectory.lib.start_task_diagnostic(
    client,
    diagnostic_task,
    agent_name="tau2-retail-airline-cookbook",
    root=Path("examples/tau2_retail_airline/runtime"),
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options=sampling,
)

diagnostic_id = diagnostic.benchmark_diagnostic_id
```

The sampling values follow
[Qwen's thinking-mode configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/generation_config.json);
the 16,384-token reply limit leaves room for multi-turn reasoning.

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

Illustrative output for a solved task (all outputs below are examples, not recorded runs):

```text
Diagnostic: completed
Inspect: https://platform.trajectory.ai/<organization>/evaluate/runs/<eval-run-id>/trajectories
Task: completed | Ending: ENV_DONE | Reward: 1.0
```

A reward of `0.0` can also be a valid result: it means the evaluator found the task unsolved.
Check any reported failures before continuing.

## Upload benchmark

Read the source's train/test split for each domain at the pinned commit. Each `TaskSpec` selects
one task by domain and ID, and all tasks share the runtime. The upload returns a `bench_id`
that identifies this benchmark version; use it for the runs below:

```python
import json
from urllib.request import urlopen

source = "https://raw.githubusercontent.com/sierra-research/tau2-bench/fc0055dc4e0a316c3f83133267fbd6faaa770992"
tasks = []
for domain in ("retail", "airline"):
    with urlopen(f"{source}/data/tau2/domains/{domain}/split_tasks.json") as response:
        splits = json.load(response)
    for split in ("train", "test"):
        for task_id in splits[split]:
            task = trajectory.TaskSpec(
                name=f"{domain}/{task_id}",
                split=split,
                run_command=f"cd /opt/tau2 && /opt/tau2/.venv/bin/python -u -m tau2_domains.agent {domain} {task_id}",
                env_vars={
                    "OPENAI_API_KEY": trajectory.SecretRef(secret_ref="OPENAI_API_KEY"),
                    "TAU2_DATA_DIR": "/opt/tau2-source/data",
                    "TAU2_USER_JUDGE_MODEL": "gpt-5.4-mini-2026-03-17",
                },
                env_resources={"cpus": 2, "memory_mb": 4096, "network_mode": "public", "agent_user": "root"},
                tags=["tau2", domain],
            )
            tasks.append(task)

benchmark = trajectory.BenchmarkSpec(
    name="tau2-retail-airline-trajectory-sdk",
    description="Pinned tau2-bench retail and airline tasks with the upstream evaluator.",
    runtime=trajectory.lib.DockerfileBuild("Dockerfile"),
    tasks=tasks,
)

result = trajectory.lib.push(
    client,
    benchmark,
    agent_name="tau2-retail-airline-cookbook",
    root=Path("examples/tau2_retail_airline/runtime"),
    build_images=True,
)
bench_id = result.bench_id
print("Benchmark:", bench_id)
print("Tasks:", result.task_count)
```

Sample output; the benchmark ID will differ for your upload:

```text
Benchmark: bm_<benchmark-id>
Tasks: 164
```

## Evaluate a model

First, evaluate the base model on the held-out tasks. This gives you a baseline to compare
against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options=sampling,
)
print("Evaluation:", evaluation.eval_run_id, links.evaluation(evaluation.eval_run_id))
```

Open the evaluation link to follow progress and inspect the conversations.

Fetch the evaluation run to see its current status and mean reward across the held-out tasks:

```python
result = client.evals.runs.retrieve(evaluation.eval_run_id)

print("Status:", result.status)
print("Mean reward:", result.reward_mean)
```

Illustrative output for a completed evaluation:

```text
Status: completed
Mean reward: 0.65
```

## Launch a training run

Train the same model on the training split for three steps. Training produces checkpoints
that you can evaluate on the same held-out tasks:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        **sampling,
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
Step 0: 0.50
Step 1: 0.75
Step 2: 0.75
```

The tau2-bench source is MIT-licensed; its
[license](runtime/LICENSE-TAU2-BENCH) is included in the runtime image.
