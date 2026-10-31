# Big Finance Benchmark

This recipe adapts the public
[Big Finance Benchmark harness](https://github.com/Rogo-Technologies/big-finance-benchmark), an
existing ReAct research benchmark, while preserving its agent loop, web and SEC tools, and rubric
grader. It packages the public
[50-question subset](https://huggingface.co/datasets/RogoAI/big-finance-benchmark); the rest of
the benchmark stays held back.

Use this pattern when a benchmark already has an OpenAI-compatible model abstraction and its own
orchestrator. The SDK adapter connects the existing lifecycle to one Trajectory session:

1. Create one trajectory when a question starts.
2. Forward the solving agent's model requests with the same trajectory ID.
3. Run the benchmark's original rubric grader.
4. Record the normalized rubric score and complete the trajectory.

## What this example contains

The harness is Rogo Technologies' work under the Apache License 2.0, and the subset is
licensed CC BY 4.0. This example keeps only the Trajectory integration and pins the upstream
harness by commit:

- [`runtime/Dockerfile`](runtime/Dockerfile) downloads upstream commit
  `bafc769bc58032f659e4117c3d839258ce0fda40` as a tarball. It extracts the harness package, the
  orchestrator script, and `data/big_finance_subset.jsonl`, then copies the overlay on top.
- [`runtime/overlay/`](runtime/overlay/) holds only the files the integration adds or changes,
  at their upstream paths, with upstream's `LICENSE`:
  - `big_finance_harness/models/trajectory.py` is the new session adapter.
  - `scripts/run_eval_set.py` gains `--task-id` and the single-task `--trajectory` mode.
  - `big_finance_harness/tools/python_exec.py` runs agent-written code with an empty
    environment, so it cannot read the harness's API keys.
- [`ingest.py`](ingest.py) reads the subset's question IDs at the same commit and uploads one
  task per question.
- [`.dockerignore`](.dockerignore) limits the build context to the Dockerfile and overlay.
- [`test_overlay.py`](test_overlay.py) checks the adapter, the reward and ending, the
  `python_exec` environment, and the dataset removal inside the built image.

The pinned commit precedes upstream's change that returns model reasoning between turns; the
session adapter does not carry reasoning back into the history.

Upstream marks the benchmark evaluation-only, so every task uses the `test` split.

## Run it

Choose an existing agent or create one using the
[agent guide](https://docs.trajectory.ai/agent). Register three organization secrets. Their
values are never included in the image or upload:

- `OPENAI_API_KEY` for the rubric judge, `gpt-5.4-mini`.
- `TAVILY_API_KEY` for `web_search`.
- `SEC_EDGAR_USER_AGENT` for `edgar_search`, as SEC EDGAR requires a contact user agent.

```python
from trajectory import Client

client = Client()
for name in ["OPENAI_API_KEY", "TAVILY_API_KEY", "SEC_EDGAR_USER_AGENT"]:
    client.secrets.create(name=name, value="...")
```

Upload the benchmark from the repository root. Ingestion waits for the image build and prints
the benchmark ID:

```sh
uv run examples/big_finance_benchmark/ingest.py --agent-name YOUR_AGENT_NAME
```

Then run a benchmark diagnostic before a full evaluation, using the
[diagnostic guide](../task_diagnostics.md#check-an-uploaded-benchmark):

```python
diagnostic = client.diagnostics.start_benchmark(bench_id="YOUR_BENCH_ID")
print(diagnostic.benchmark_diagnostic_id)
```

Image build readiness alone does not verify policy inference, the web tools, grading, or
reward publication.

To run the overlay tests, build the image locally and run them inside it:

```sh
docker build -t big-finance-trajectory examples/big_finance_benchmark/runtime
docker run --rm \
  -v "$PWD/examples/big_finance_benchmark/test_overlay.py:/opt/big_finance/test_overlay.py:ro" \
  big-finance-trajectory sh -c "pip install -q pytest && python -m pytest -q test_overlay.py"
```

## Adapt the existing model client

The adapter implements the benchmark's existing `ModelClient` interface. It converts the
existing messages and tools, then forwards each request through the Trajectory session model:

```python
self.client = Client()
self.tid = self.client.trajectories.create().tid

response = await asyncio.to_thread(
    self.client.chat.completions.create,
    model=self.snapshot,  # "trajectory-session"
    messages=_to_oai_messages(system, messages),
    x_trajectory_id=self.tid,
    extra_body={"tools": _to_oai_tools(tools), "tool_choice": "auto"},
)
```

The adapter maps that response back into the benchmark's native response type, so its ReAct loop
and tool execution remain unchanged. The managed run's options set `max_tokens` and sampling, so
the adapter does not send them. It records when the run's per-step output limit cut a reply off.

## Reuse the original grader

In trajectory mode, the orchestrator loads its one question and deletes the dataset file, since
`python_exec` can read files. After the agent finishes, it calls the existing rubric grader and
normalizes its score. The adapter's trajectory receives the reward, then completes with the
reason the agent stopped. A harness error or grader failure completes `ERROR` without a reward;
a context overflow or a cut-off reply is still graded, as the native harness does, and completes
`TRUNCATION`:

```python
client, tid = adapter.client, adapter.tid
try:
    if run.stop_reason == "error":
        raise RuntimeError(f"agent run failed ({run.stop_reason}): {run.error}")
    graded = asyncio.run(grade(run=run, item=item, judge_model_id=judges[0]))
    earned, possible = graded.rubric_points_earned, graded.rubric_points_possible
    client.trajectories.log_reward(
        tid, name="reward_accuracy", value=earned / possible,
        explanation=f"{earned}/{possible} rubric points",
    )
except Exception:
    client.trajectories.complete(tid, termination_reason="ERROR")
    raise
endings = {
    "max_steps": "MAX_STEPS",
    "context_exceeded": "TRUNCATION",
    "token_budget": "TRUNCATION",
}
ending = "TRUNCATION" if adapter.truncated else endings.get(run.stop_reason, "ENV_DONE")
client.trajectories.complete(tid, termination_reason=ending)
```

## Package the evaluation set

The uploader creates one evaluation task for each subset question. `BFB_TASK_ID` selects exactly
one row inside the existing orchestrator, while credentials remain platform secret references:

```python
TaskSpec(
    name=f"big-finance/{task_id}",
    split="test",
    run_command=(
        "python -u /opt/big_finance/scripts/run_eval_set.py --trajectory "
        "--dataset /opt/big_finance/data/big_finance_subset.jsonl "
        "--run-id trajectory --max-steps 30 --judge openai:gpt-5.4-mini"
    ),
    env_vars={
        "BFB_TASK_ID": task_id,
        "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
        "TAVILY_API_KEY": SecretRef(secret_ref="TAVILY_API_KEY"),
        "SEC_EDGAR_USER_AGENT": SecretRef(secret_ref="SEC_EDGAR_USER_AGENT"),
    },
    env_resources=EnvResources(network_mode="public"),
)
```

The image contains only the harness, the orchestrator, the public subset, and the overlay. The
held-back benchmark data is not in the upstream repository and is never uploaded or copied into
the image.
