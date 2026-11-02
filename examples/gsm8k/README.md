# GSM8K

This example adapts a reproducible sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
to the Trajectory SDK. The selected rows are stored in [`runtime/tasks.jsonl`](runtime/tasks.jsonl).

The model must call `submit_answer`. The harness compares that numeric answer with the private
reference, records `reward_accuracy`, and completes the trajectory as `ENV_DONE` or `TRUNCATION`.

## Setup Runtime with Trajectory SDK

Run commands from the repository root with a Trajectory API key:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'import trajectory; trajectory.Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

## Diagnose with one task

Build the runtime and run one training problem before uploading the full benchmark:

```bash
uv run --with trajectory-sdk python -c \
  'from examples.gsm8k.ingest import task_diagnose; task_diagnose("gsm8k-cookbook")'
```

The command prints the diagnostic ID and fails if execution or grading does not complete.

## Upload benchmark

```bash
uv run examples/gsm8k/ingest.py \
  --agent-name "gsm8k-cookbook" \
  --name "gsm8k-trajectory-sdk"
```

The uploader registers the 80 tasks, waits for the runtime image, and runs a benchmark
diagnostic. Save the printed benchmark ID.

## Evaluate, train, and compare

Start a Python session with the SDK:

```bash
uv run --with trajectory-sdk python
```

Start an evaluation:

```python
import trajectory

client = trajectory.Client()
bench_id = "YOUR_BENCHMARK_ID"
evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "reasoning_effort": "off",
        "max_output_tokens_per_turn": 2_048,
    },
)
```

Start training:

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

Print links to both runs in the product UI:

```python
organization = client.organizations.retrieve_current()
product_url = f"https://platform.trajectory.ai/{organization['organization_key']}"
print(f"{product_url}/evaluate/runs/{evaluation.eval_run_id}")
print(f"{product_url}/models/{training.training_run_id}")
```
