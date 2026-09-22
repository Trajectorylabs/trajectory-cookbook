# Trajectory Word

This instruction-following example trains a model to obey the system prompt:

```text
Always include the word trajectory in your response.
```

Its deterministic synthetic dataset contains 128 training prompts and 64 held-out test prompts.
The prompts ask varied questions such as `How is the weather?` and `What is 1 + 1?`. A response
earns reward `1` when it contains `trajectory` as a standalone word, ignoring case, and `0`
otherwise.

## Prerequisites

- Python 3.11 or newer
- [`uv`](https://docs.astral.sh/uv/)
- A Trajectory API key

```bash
export TRAJECTORY_API_KEY="..."
```

## 1. Ingest the benchmark

```bash
uv run ingest.py
```

The command prints a `bench_id` and waits for the runtime image to build.

## 2. Train and evaluate

```bash
uv run train.py --bench-id YOUR_BENCH_ID --num-steps 3
```

The script evaluates the base model on the 64 test tasks, trains with batches of four task groups
and eight rollouts per group, then evaluates the final checkpoint on the same held-out tasks.
