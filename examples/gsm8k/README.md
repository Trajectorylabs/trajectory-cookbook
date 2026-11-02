# GSM8K

## Dataset

This example adapts a reproducible sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c)
to the Trajectory SDK. The selected rows are stored in [`runtime/tasks.jsonl`](runtime/tasks.jsonl).
The upload contains 64 training and 16 held-out test problems.

## Harness

The model must call `submit_answer`. The harness compares that numeric answer with the private
reference, records `reward_accuracy`, and completes the trajectory as `ENV_DONE` or `TRUNCATION`.

## Ingestion

Run commands from the repository root with a Trajectory API key:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'import trajectory; trajectory.Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

### Diagnose one task

Build the runtime and run one training problem before uploading the full benchmark:

```bash
uv run --with trajectory-sdk python -c \
  'from examples.gsm8k.ingest import task_diagnose; task_diagnose("gsm8k-cookbook")'
```

The command prints the diagnostic ID and fails if execution or grading does not complete.

### Upload the benchmark

```bash
uv run examples/gsm8k/ingest.py \
  --agent-name "gsm8k-cookbook" \
  --name "gsm8k-trajectory-sdk"
```

The uploader registers the 80 tasks, waits for the runtime image, and runs a benchmark
diagnostic. Save the printed benchmark ID.

## Evaluation

The training script first evaluates the base model on the held-out tasks. It later evaluates
the final checkpoint on those same tasks.

## Training

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

## Appendix

The SDK adapter owns the trajectory and model calls; the harness keeps the dataset's private
answer and exact-match grade. The pinned dataset revision and selected rows above make the
uploaded tasks reproducible.
The pinned GSM8K source is [MIT-licensed](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE);
the runtime includes its [license text](runtime/LICENSE-GSM8K).
