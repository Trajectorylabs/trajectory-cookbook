# Big Finance Benchmark

This example adapts the public
[Big Finance Benchmark harness](https://github.com/Rogo-Technologies/big-finance-benchmark),
preserving its agent loop, web and SEC tools, and rubric grader. It packages the public
[50-question subset](https://huggingface.co/datasets/RogoAI/big-finance-benchmark); the rest of
the benchmark stays held back.

Big Finance is an evaluation-only benchmark: Rogo marks every record `do_not_train`. This example
uploads the subset, runs a diagnostic, and evaluates a model or checkpoint on it. All 50 tasks
use the `test` split.

The adapter sends the harness's model requests through one Trajectory session per question,
then records the native rubric score as `reward_accuracy`. The pinned upstream commit
predates reasoning being returned between turns, so the adapter does not carry reasoning
into the next request.

## What this example contains

The harness is Rogo Technologies' work under the Apache License 2.0, and the subset is
licensed CC BY 4.0. The runtime pins the upstream harness by commit:

- [`runtime/Dockerfile`](runtime/Dockerfile) downloads upstream commit
  `bafc769bc58032f659e4117c3d839258ce0fda40` as a tarball. It extracts the harness package, the
  orchestrator script, and `data/big_finance_subset.jsonl`, then copies the overlay on top.
- [`runtime/overlay/`](runtime/overlay/) adds the session adapter and single-task mode at
  their upstream paths. Its `python_exec` change runs agent-written code as `nobody`
  without harness credentials, with process, resource, output and time limits.
- [`ingest.py`](ingest.py) reads the subset's question IDs at the same commit and uploads one
  task per question.
- [`.dockerignore`](.dockerignore) limits the build context to the Dockerfile and the overlay.

## Run it

Choose an existing agent or create one using the
[agent guide](https://docs.trajectory.ai/agent). Export your organization API key before
running the commands below:

```sh
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

For the default judge, register these three
organization secrets. With a judge override, replace `VERTEXAI_CREDENTIALS` with the override's
credential; the two tool secrets are still needed:

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

### Optional judge override

`--judge` takes any `provider:model` the harness accepts. The uploader maps the judge's
credential variable to the matching organization secret; use `--judge-secret` if that secret
has a different name. The harness has no `openrouter` provider, but its `openai` provider
passes the model name to LiteLLM unchanged, so an `openrouter/` model grades through OpenRouter.
Register `OPENROUTER_API_KEY` and use this upload command instead of the default:

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

With these options and the OpenRouter `openai/gpt-6-luna` judge override, a 50-question
Qwen 3.6 35B A3B evaluation (`evr_06ae69b90a4776e08000776ed3b1e34f`) finished in 28
minutes. All 50 tasks recorded rewards: 49 ended `ENV_DONE` and one graded attempt ended
`TRUNCATION`. The platform's run mean was 0.479; it counts the truncated attempt as zero
despite its partial rubric reward. Add `"evaluation_max_samples": 10` to the options for a
smaller check.

To evaluate a checkpoint, pass its ready deployment's `deployment_id=` in place of
`base_model_slug`. A deployment brings its own sampling, token, and turn settings, so those
options return 422 with it; see the [evaluation guide](https://docs.trajectory.ai/guides/evaluations).

The image contains the public subset only. The held-back benchmark data is not in the
upstream repository and is never uploaded or copied into the image.

## Test the overlay

From the cookbook root:

```sh
docker build -t big-finance-trajectory examples/big_finance_benchmark/runtime
docker run --rm \
  -v "$PWD/examples/big_finance_benchmark/test_overlay.py:/opt/big_finance/test_overlay.py:ro" \
  big-finance-trajectory sh -c "pip install -q pytest && python -m pytest -q test_overlay.py"
uv run --with trajectory-sdk --with pytest pytest examples/big_finance_benchmark/test_ingest.py
```
