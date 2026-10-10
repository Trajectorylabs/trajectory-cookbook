# GSM8K: adapt a real dataset

After [your first task](../t_factory/README.md) and the [evaluation and training walkthrough](../t_factory/training.md),
use this example to learn dataset loading, explicit train/test splits, private answers, and tool-based grading.

This example adapts the public
[GSM8K dataset](https://github.com/openai/grade-school-math) to the Trajectory SDK. It demonstrates
the three benchmark integration points:

1. [`ingest.py`](ingest.py) writes source rows to local task files, maps them to train/test
   `TaskSpec` objects, and uploads them with the runtime.
2. [`runtime/gsm8k_harness.py`](runtime/gsm8k_harness.py) exposes a `submit_answer` tool, grades
   the numeric value submitted through its final tool call, logs reward, and completes the
   trajectory. Text-only answers receive zero reward.
3. [`train.py`](train.py) evaluates a baseline, trains a model, and evaluates the final checkpoint
   on held-out tasks.

## Prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A Trajectory API key

```bash
export TRAJECTORY_API_KEY="..."
```

The SDK defaults to `https://api.trajectory.ai`. Set `TRAJECTORY_BASE_URL` when using another
deployment.

[Choose or create an agent](../../README.md#choose-or-create-an-agent). For a new agent:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook")'
```

Run these commands from the cookbook repository root. For a new harness, use the
[one-task diagnostic recipe](../task_diagnostics.md) before uploading the full dataset.

## 1. Ingest the benchmark

Upload the example's 64 training tasks and 16 test tasks using its name:

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook"
```

The command prints the `agent_name` and `bench_id`, then waits for the runtime image to build. Keep
the benchmark ID for training.

## 2. Train and evaluate

```bash
uv run examples/gsm8k/train.py --bench-id YOUR_BENCH_ID --num-steps 3
```

The script:

1. Verifies that the benchmark has train and test splits.
2. Evaluates the base model on the benchmark's held-out tasks.
3. Starts a training run and waits for completion.
4. Resolves and evaluates the final checkpoint on the same held-out tasks.
5. Prints the baseline reward, final reward, and reward delta.

A single short run is an integration check, not statistical evidence that training improves the
model. Use repeated runs and a sufficiently large frozen test set for a reliable comparison.

## Adapt this example

To integrate another benchmark, preserve its original task data and grader, then replace:

- `_load_rows()` with the benchmark's dataset loader.
- The staged task JSON files with the inputs and private references needed by one task. Keep
  private references inaccessible to the model and its tools.
- The tool definition and `extract_submitted_answer()` with the benchmark's interaction protocol.
- The equality check with the benchmark's original grader.

The runtime calls the provided model endpoint, logs reward, and
completes the trajectory.
