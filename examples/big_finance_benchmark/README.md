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
- [`runtime/overlay/`](runtime/overlay/) holds only the files the integration adds or changes,
  at their upstream paths, with upstream's `LICENSE`:
  - `big_finance_harness/models/trajectory.py` is the new session adapter.
  - `scripts/run_eval_set.py` gains `--task-id` and the single-task `--trajectory` mode, which
    must start as root (the task sets `agent_user="root"`).
  - `big_finance_harness/tools/python_exec.py` runs agent-written code as `nobody` in a fresh
    temporary directory, passing only a non-secret OpenBLAS thread limit. The harness's API keys
    are neither in the child's environment nor readable from the root-owned
    `/proc/<parent>/environ`. The snippet cannot start child processes, and its process group is
    killed when the call ends.
    Its output is capped as it is read.
- [`ingest.py`](ingest.py) reads the subset's question IDs at the same commit and uploads one
  task per question.
- [`.dockerignore`](.dockerignore) limits the build context to the Dockerfile and the overlay.
- [`test_overlay.py`](test_overlay.py) checks the adapter, the reward and ending, the
  `python_exec` isolation, and the dataset removal inside the built image.
- [`test_ingest.py`](test_ingest.py) checks the default, OpenRouter, and custom-secret
  credential mapping without contacting a judge.

The pinned commit precedes upstream's change that returns model reasoning between turns; the
session adapter does not carry reasoning back into the history.

## Run it

Choose an existing agent or create one using the
[agent guide](https://docs.trajectory.ai/agent). For the default judge, register these three
organization secrets. With a judge override, replace `VERTEXAI_CREDENTIALS` with the override's
credential; the two tool secrets are still needed. Secret values never enter the image or upload:

- `VERTEXAI_CREDENTIALS` for the rubric judge, upstream's default Gemini 3.1 Pro on Vertex AI
  (`vertex:gemini-3.1-pro-preview`). Its value is the JSON key of a service account that can
  call Vertex AI. LiteLLM reads this variable and takes the project from the key. To grade with
  another judge, register its provider's key instead; see [Choose the judge](#choose-the-judge).
- `TAVILY_API_KEY` for `web_search`.
- `SEC_EDGAR_USER_AGENT` for `edgar_search`, as SEC EDGAR requires a contact user agent.

```python
from pathlib import Path
from trajectory import Client

client = Client()
client.secrets.create(name="VERTEXAI_CREDENTIALS", value=Path("key.json").read_text())
for name in ["TAVILY_API_KEY", "SEC_EDGAR_USER_AGENT"]:
    client.secrets.create(name=name, value="...")
```

Upload the benchmark from this repository's root. Ingestion waits for the image build and
prints the benchmark ID:

```sh
uv run examples/big_finance_benchmark/ingest.py --agent-name YOUR_AGENT_NAME
```

Tasks set `VERTEX_DISABLE_DEDICATED=1`, so Vertex calls may use pay-as-you-go capacity; upstream
otherwise sends them to Provisioned Throughput only.

### Choose the judge

`--judge` takes any `provider:model` the harness accepts. The uploader maps the judge's
credential variable to the matching organization secret; use `--judge-secret` if that secret
has a different name. The harness has no `openrouter` provider, but its `openai` provider
passes the model name to LiteLLM unchanged, so an `openrouter/` model grades through OpenRouter.
Register `OPENROUTER_API_KEY` and run:

```sh
uv run examples/big_finance_benchmark/ingest.py --agent-name YOUR_AGENT_NAME \
  --judge openai:openrouter/openai/gpt-6-luna
```

### Run a diagnostic

Run a benchmark diagnostic before a full evaluation, using the
[diagnostic guide](../task_diagnostics.md#check-an-uploaded-benchmark). The harness makes up
to 50 model calls per question, each followed by web, SEC, or Python tools, so give each
attempt room for long reasoning replies and the whole loop. The evaluation below uses the
same model and settings:

```python
model = "Qwen/Qwen3.6-35B-A3B"
options = {
    "reasoning_effort": "on",
    "max_output_tokens_per_step": 16384,
    "max_total_tokens_per_trajectory": 819200,  # 50 calls x 16,384 tokens
    "max_turns_per_trajectory": 50,  # the harness's default max steps
    "execution_timeout_seconds": 3600,
}
diagnostic = client.diagnostics.start_benchmark(
    bench_id="YOUR_BENCH_ID", base_model_slug=model, options=options
)
print(diagnostic.benchmark_diagnostic_id)
```

Image build readiness alone does not verify policy inference, the web tools, grading, or
reward publication.

To run the tests, build the image and run the overlay tests inside it:

```sh
docker build -t big-finance-trajectory examples/big_finance_benchmark/runtime
docker run --rm \
  -v "$PWD/examples/big_finance_benchmark/test_overlay.py:/opt/big_finance/test_overlay.py:ro" \
  big-finance-trajectory sh -c "pip install -q pytest && python -m pytest -q test_overlay.py"
```

Check the uploader's judge-secret mapping from the cookbook root:

```sh
uv run --with trajectory-sdk --with pytest pytest examples/big_finance_benchmark/test_ingest.py
```

## Evaluate

```python
run = client.evals.create(
    bench_id="YOUR_BENCH_ID",
    base_model_slug=model,
    options=options,
    display_name="Big Finance baseline",
)
print(run.eval_run_id)
```

With these options, a 50-question Qwen 3.6 35B A3B evaluation completed every attempt with
`ENV_DONE` and no cut-off replies in about 32 minutes. Its longest attempt made 41 of its 50
model calls before calling `final_answer`. Much lower per-step limits
cut replies off: in a 50-question Qwen 3.5 4B run with 4,096 tokens per step, 13 attempts ended
`TRUNCATION`. The snippet runs all 50 questions; add `"evaluation_max_samples": 10` to the
options for a smaller check.

To evaluate a checkpoint, pass its ready deployment's `deployment_id=` in place of
`base_model_slug`. A deployment brings its own sampling, token, and turn settings, so those
options return 422 with it; see the [evaluation guide](https://docs.trajectory.ai/guides/evaluations).

## How the integration works

- [`trajectory.py`](runtime/overlay/big_finance_harness/models/trajectory.py) implements the
  harness's `ModelClient`. It creates one trajectory per attempt and sends each request to
  `trajectory-session` with that trajectory ID, leaving `max_tokens` and sampling to the run's
  options. A reply cut off by the per-step limit comes back without its tool calls, so the
  harness's loop stops and grades the partial answer.
- [`run_eval_set.py`](runtime/overlay/scripts/run_eval_set.py)'s `--trajectory` mode loads one
  question, deletes the dataset file because `python_exec` can read files, and runs the
  harness's own loop and rubric grader. It logs the normalized rubric score as
  `reward_accuracy` and completes with `TRUNCATION` for a cut-off reply, context overflow, or
  token budget, `MAX_STEPS` at the step limit, and `ENV_DONE` otherwise. A harness or grader
  error completes `ERROR` without a reward.
- [`ingest.py`](ingest.py)'s `build_benchmark` creates one `test` task per question. Each task
  selects its row with `BFB_TASK_ID` and receives credentials as secret references.

The image contains only the harness, the orchestrator, the public subset, and the overlay. The
held-back benchmark data is not in the upstream repository and is never uploaded or copied into
the image.
