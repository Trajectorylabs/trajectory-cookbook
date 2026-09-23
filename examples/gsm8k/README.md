# GSM8K

This example adapts the public
[GSM8K dataset](https://github.com/openai/grade-school-math) to the Trajectory SDK. It demonstrates
the three benchmark integration points:

1. [`ingest.py`](ingest.py) maps source rows to train/test `TaskSpec` objects and uploads them.
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

## 1. Ingest the benchmark

Upload the example's 64 training tasks and 16 test tasks:

```bash
uv run ingest.py
```

The command prints a `bench_id` and waits for the runtime image to build. Keep that ID for training.

For organizations with multiple agents, the [next example](../../README.md#example-2-maximize-t-density)
shows how to fetch the organization's default agent and pass `agent.agent_id` explicitly.

## 2. Train and evaluate

```bash
uv run train.py --bench-id YOUR_BENCH_ID --num-steps 3
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
- `TaskSpec.env_vars` with the inputs needed by one task.
- The tool definition and `extract_submitted_answer()` with the benchmark's interaction protocol.
- The equality check with the benchmark's original grader.

Keep the SDK boundary unchanged: the runtime resolves its existing trajectory with
`client.trajectories.create().tid`, calls the provided model endpoint, and passes that trajectory
ID to reward logging and completion. The model endpoint token preserves the benchmark's agent.
