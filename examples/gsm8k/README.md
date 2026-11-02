# GSM8K

This example adapts a reproducible sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
to the Trajectory SDK.

The model must call `submit_answer`. The harness compares that numeric answer with the private
reference, records `reward_accuracy`, and completes the trajectory as `ENV_DONE` or `TRUNCATION`.

## Set up

Run commands from the repository root with a Trajectory API key:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

If the agent already exists, reuse its name.

## Validate one task

Build the runtime and run one training problem before uploading the full benchmark:

```bash
uv run --with trajectory-sdk python -c \
  'from examples.gsm8k.ingest import task_diagnose; task_diagnose("gsm8k-cookbook")'
```

The command prints the diagnostic ID and fails if execution or grading does not complete.

## Upload the benchmark

```bash
uv run examples/gsm8k/ingest.py \
  --agent-name "gsm8k-cookbook" \
  --name "gsm8k-trajectory-sdk"
```

The uploader registers the 80 tasks, waits for the runtime image, and runs a benchmark
diagnostic. Save the printed benchmark ID.

## Evaluate, train, and compare

```bash
uv run examples/gsm8k/train.py \
  --agent-name "gsm8k-cookbook" \
  --bench-name "gsm8k-trajectory-sdk" \
  --num-steps 3
```

The script starts a baseline evaluation and training run, waits for both, resolves the final
checkpoint, and evaluates it on the same 16 held-out tasks. It prints all run IDs and the
baseline, final, and delta rewards. A short run validates the workflow; it is not evidence of
training improvement.

Interrupting the local process does not cancel accepted work. Use the printed IDs in the
platform or with `client.evals.runs.retrieve()` and `client.training.runs.retrieve()`.

## How it is organized

- [`ingest.py`](ingest.py) downloads the pinned rows and creates explicit train/test `TaskSpec`
  objects.
- [`runtime/gsm8k_harness.py`](runtime/gsm8k_harness.py) calls the model, grades the submitted
  answer, records reward, and completes the trajectory.
- [`train.py`](train.py) evaluates the base model, trains it, and compares the final checkpoint.

When adapting this example, keep private references out of model messages and tools. See the
[benchmark guide](https://docs.trajectory.ai/guides/benchmarks),
[evaluation guide](https://docs.trajectory.ai/guides/evaluations), and
[training guide](https://docs.trajectory.ai/guides/training) for the full API surface.
