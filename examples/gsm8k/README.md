# GSM8K

## Dataset

Evaluate and train a model to solve grade-school math problems from
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
with Trajectory. This walkthrough takes you from running one problem to uploading a benchmark
and starting evaluation and training runs. The upload uses 64 training and 16 held-out
problems.

## Harness

For each problem, the model submits its answer through a `submit_answer` tool. The
[harness](runtime/gsm8k_harness.py) checks it against the reference answer and records a reward
of 1 for a correct answer or 0 otherwise. Problems and reference answers live in
[`runtime/tasks.jsonl`](runtime/tasks.jsonl); only the question is sent to the model.

## Ingestion

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

### Diagnose one task

Try the first problem before uploading the full benchmark. This diagnostic builds the image and
starts the harness with `--task_id 0`, which selects the first row of the JSONL file. Use the
report below to check whether the task ran, how it ended, and what reward it received:

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
import time

diagnostic_id = diagnostic.benchmark_diagnostic_id
while client.diagnostics.get_status(diagnostic_id).status in {"pending", "running"}:
    time.sleep(5)

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

### Upload benchmark

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
print("Benchmark:", bench_id)
print("Tasks:", result.task_count)
```

Sample output; the benchmark ID will differ for your upload:

```text
Benchmark: bm_<benchmark-id>
Tasks: 80
```

## Evaluation

First, evaluate the base model on the held-out problems. This gives you a baseline to compare
against trained checkpoints:

```python
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options=model_options,
)
print("Evaluation:", evaluation.eval_run_id)
```

Sample output after the evaluation is accepted:

```text
Evaluation: evr_<evaluation-id>
```

## Training

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
print("Training:", training.training_run_id)
```

Sample output after training is accepted:

```text
Training: trn_<training-id>
```

Open these links to follow the evaluation and training runs in Trajectory. Once training
finishes, evaluate a checkpoint on this benchmark to compare its score with the baseline:

```python
organization = client.organizations.retrieve_current()
product_url = f"https://platform.trajectory.ai/{organization['organization_key']}"
print("Evaluation:", f"{product_url}/evaluate/runs/{evaluation.eval_run_id}")
print("Training:", f"{product_url}/models/{training.training_run_id}")
```

Sample links; your organization and run IDs appear in their place:

```text
Evaluation: https://platform.trajectory.ai/<organization>/evaluate/runs/evr_<evaluation-id>
Training: https://platform.trajectory.ai/<organization>/models/trn_<training-id>
```

## Appendix

The pinned GSM8K source is [MIT-licensed](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE);
the runtime includes its [license text](runtime/LICENSE-GSM8K).
