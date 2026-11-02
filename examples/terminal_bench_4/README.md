# Terminal-Bench 4.0

[Terminal-Bench 4.0](https://github.com/harbor-framework/terminal-bench/tree/v4.0.0)
tests agents on terminal tasks. This example uses its official test tasks at commit
`452bf305c6daa62fc59061d22133a7cbc7c1572e` and keeps Harbor's task environment,
artifact collection, and verifier. The default upload contains one CPU task for a quick
check. It is an **evaluation-only** example: the release has no official training split.

## Try one task

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), Git, and a Trajectory organization
API key. The task runtime needs nested Docker, as in the
[SWE-Bench Pro example](../swe_bench_pro/README.md).

```sh
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk==0.9.37 python -c \
  'from trajectory import Client; Client().agents.create(name="terminal-bench-4", description="Terminal-Bench 4.0 evaluation")'
uv run examples/terminal_bench_4/ingest.py --agent-name terminal-bench-4 --agent-timeout 300
```

The command fetches the pinned upstream task, omits its `solution/` and `cheat/` files,
uploads it, waits for the image, and runs a diagnostic with Qwen 3.6 35B A3B. It prints
the benchmark and diagnostic IDs. A reward of zero is a valid grade; a missing grade or
failed task is not. The `--agent-timeout 300` flag caps the source's eight-hour agent
deadline for this smoke check, so its score is not comparable to Terminal-Bench's
leaderboard.

To evaluate the same uploaded version, use the printed `bench_id`:

```python
from trajectory import Client

run = Client().evals.create(
    bench_id="YOUR_BENCH_ID",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "reasoning_effort": "on",
        "max_turns_per_trajectory": 300,
        "max_output_tokens_per_step": 8192,
        "max_total_tokens_per_trajectory": 131072,
        "execution_timeout_seconds": 2100,
    },
)
print(run.eval_run_id)
```

Pass `--task NAME` more than once to select other names from
[`test_tasks.txt`](test_tasks.txt). Without `--agent-timeout`, tasks retain their original
agent deadline. Three official test tasks require GPU plus nested Docker and are rejected
by this CPU example. This recipe therefore does not claim a full 50-task leaderboard run.

## What changes from upstream

The task files and grades come from the pinned [Apache-2.0-licensed source](https://github.com/harbor-framework/terminal-bench/blob/v4.0.0/LICENSE).
The uploader excludes reference solutions and selects official test tasks. A small
Trajectory bash agent supplies model calls; Harbor still creates the original task
environment, transfers artifacts, and runs its separate verifier. The adapter sends
that verifier's reward to the Trajectory trajectory. The optional timeout flag changes
the agent budget only for the quick check above.
