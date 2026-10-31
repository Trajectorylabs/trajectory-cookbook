# Big Finance Benchmark

This recipe adapts the public
[Big Finance Benchmark harness](https://github.com/Rogo-Technologies/big-finance-benchmark), an
existing ReAct research benchmark, while preserving its agent loop, web and SEC tools, and rubric
grader. It packages the public
[50-question subset](https://huggingface.co/datasets/RogoAI/big-finance-benchmark); the rest of
the benchmark stays held back.

Big Finance is an evaluation-only benchmark: Rogo marks every record `do_not_train`. This example
uploads the subset, runs a diagnostic, and evaluates a model or checkpoint on it. All 50 tasks
use the `test` split.

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
  [`runtime/checkout.Dockerfile`](runtime/checkout.Dockerfile) copies the same files from your
  checkout of that commit instead.
- [`runtime/overlay/`](runtime/overlay/) holds only the files the integration adds or changes,
  at their upstream paths, with upstream's `LICENSE`:
  - `big_finance_harness/models/trajectory.py` is the new session adapter.
  - `scripts/run_eval_set.py` gains `--task-id` and the single-task `--trajectory` mode, which
    must start as root (the task sets `agent_user="root"`).
  - `big_finance_harness/tools/python_exec.py` runs agent-written code with an empty
    environment, as `nobody` in a fresh temporary directory. The harness's API keys are neither
    in the child's environment nor readable from the root-owned `/proc/<parent>/environ`. The
    snippet runs in its own process group, which is killed when the call ends, and its output is
    capped as it is read.
- [`ingest.py`](ingest.py) reads the subset's question IDs at the same commit and uploads one
  task per question.
- [`.dockerignore`](.dockerignore) limits the build context to the Dockerfiles, the overlay,
  and the staged harness files.
- [`test_overlay.py`](test_overlay.py) checks the adapter, the reward and ending, the
  `python_exec` isolation, and the dataset removal inside the built image.
  [`test_ingest.py`](test_ingest.py) checks how `--upstream` stages a checkout.

The pinned commit precedes upstream's change that returns model reasoning between turns; the
session adapter does not carry reasoning back into the history.

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

Upload the benchmark from this repository's root. Ingestion downloads the pinned harness,
waits for the image build, and prints the benchmark ID:

```sh
uv run examples/big_finance_benchmark/ingest.py --agent-name YOUR_AGENT_NAME
```

If you are already in a checkout of Rogo's harness, package that checkout instead. It must be
at commit `bafc769bc58032f659e4117c3d839258ce0fda40`; ingestion stops if `HEAD` differs.
Ingestion copies only the committed harness package, orchestrator script, subset, `LICENSE`,
`README.md`, and `pyproject.toml` at that commit, and leaves out local changes, other data, and
grades. From your checkout:

```sh
git checkout bafc769bc58032f659e4117c3d839258ce0fda40
uv run PATH_TO_COOKBOOK/examples/big_finance_benchmark/ingest.py --upstream . \
  --agent-name YOUR_AGENT_NAME
```

The judge defaults to `openai:gpt-5.4-mini`. `--judge` takes any `provider:model` the harness
accepts, and the `openai` provider passes the model to LiteLLM unchanged, so any LiteLLM model
string works. `--judge-secret` names the organization secret holding that provider's key; it
reaches the task under the same environment variable name. For example, to grade through
OpenRouter, register `OPENROUTER_API_KEY` in place of `OPENAI_API_KEY`:

```sh
uv run examples/big_finance_benchmark/ingest.py --agent-name YOUR_AGENT_NAME \
  --judge openai:openrouter/openai/gpt-6-luna --judge-secret OPENROUTER_API_KEY
```

Then run a benchmark diagnostic before a full evaluation, using the
[diagnostic guide](../task_diagnostics.md#check-an-uploaded-benchmark):

```python
diagnostic = client.diagnostics.start_benchmark(bench_id="YOUR_BENCH_ID")
print(diagnostic.benchmark_diagnostic_id)
```

Image build readiness alone does not verify policy inference, the web tools, grading, or
reward publication.

To run the tests, run `test_ingest.py` locally, then build the image and run the overlay tests
inside it:

```sh
uv run --with trajectory-sdk --with pytest pytest examples/big_finance_benchmark/test_ingest.py
docker build -t big-finance-trajectory examples/big_finance_benchmark/runtime
docker run --rm \
  -v "$PWD/examples/big_finance_benchmark/test_overlay.py:/opt/big_finance/test_overlay.py:ro" \
  big-finance-trajectory sh -c "pip install -q pytest && python -m pytest -q test_overlay.py"
```

## Evaluate

The harness makes up to 30 model calls per question, each followed by web, SEC, or Python
tools, so give each attempt room for long reasoning replies and the whole loop:

```python
run = client.evals.create(
    bench_id="YOUR_BENCH_ID",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "reasoning_effort": "on",
        "max_output_tokens_per_step": 16384,
        "max_total_tokens_per_trajectory": 491520,  # 30 calls x 16,384 tokens
        "max_turns_per_trajectory": 30,  # matches the harness's --max-steps 30
        "execution_timeout_seconds": 3600,
    },
    display_name="Big Finance baseline",
)
print(run.eval_run_id)
```

With these settings, a 10-question Qwen 3.6 35B A3B evaluation completed every attempt with
`ENV_DONE` and no cut-off replies. The longest attempt used all 30 calls and still ended on
`final_answer`, and the run took about 25 minutes. Much lower per-step limits cut replies off:
in a 50-question Qwen 3.5 4B run with 4,096 tokens per step, 13 attempts ended `TRUNCATION`. The
snippet runs all 50 questions; add `"evaluation_max_samples": 10` to the options for a smaller
check.

To evaluate a checkpoint, pass its ready deployment's `deployment_id=` in place of
`base_model_slug`. A deployment brings its own sampling, token, and turn settings, so those
options return 422 with it; see the [evaluation guide](https://docs.trajectory.ai/guides/evaluations).

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
one row inside the existing orchestrator, while credentials remain platform secret references.
With the default judge, each task is:

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
    env_resources=EnvResources(network_mode="public", agent_user="root"),
)
```

The image contains only the harness, the orchestrator, the public subset, and the overlay. The
held-back benchmark data is not in the upstream repository and is never uploaded or copied into
the image.
