# Public Tau 3 Banking

This example packages Sierra's public 97-task Banking benchmark from
[tau2-bench v1.0.1](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992). It keeps the source task bytes,
`bm25_grep` retrieval, simulated-user prompts, and evaluator. Following the
[Artificial Analysis methodology](https://artificialanalysis.ai/methodology/intelligence-benchmarking),
the simulated user and `task_102` natural-language judge use GPT-5.4 Mini at medium reasoning
instead of the source release's GPT-4.1 defaults. The cookbook sorts task IDs into 80 training
and 17 held-out test tasks; this split is for training, not the full 97-task, five-repeat
leaderboard evaluation.

The SDK supplies agent inference. Sierra's runner continues after an output-limited model reply,
so this example records its final evaluator grade. Sierra grades attempts that stop after too
many agent errors as zero; the SDK records these as graded failed attempts and retains the
source ending in a trace event. Model, environment, and grader failures remain ungraded. The
simulator makes up to three attempts to get a nonempty user response without adding empty
replies to the conversation; exhaustion still fails the task.

Choose an existing agent and organization API key using the
[agent guide](https://docs.trajectory.ai/agent). Register your OpenAI key once as the
organization secret `OPENAI_API_KEY` for the upstream user simulator and judge; its value is
never included in the image or upload.

```sh
uv run --with trajectory-sdk python -c \
  'import os; from trajectory import Client; Client().secrets.create(name="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])'
uv run examples/tau3_banking/ingest.py --agent-name YOUR_AGENT_NAME
```

To use an OpenRouter customer and judge model, register `OPENROUTER_API_KEY` instead and pass
`--user-judge-model openrouter/MODEL` when ingesting. The native GPT-5.4 Mini medium setting
remains the default.

Ingestion waits for a new runtime build. Follow the
[benchmark diagnostic guidance](../../task_diagnostics.md) to execute a subset before training,
with the model and options you will evaluate with, such as
`base_model_slug="Qwen/Qwen3.6-35B-A3B"` and the sampling options below. Without them, a
diagnostic runs on the API's default diagnostic model. Image build readiness alone does not
verify policy inference, grading, or reward publication.

Managed runs take sampling from run options, not the upstream harness. The following settings
are for the tested Qwen 3.6 35B A3B evaluation, not a requirement of Tau 3 Banking. The
16,384-token reply limit let the multi-turn diagnostic finish; the smaller 2,048-token limit
cut off a model reply. The sampling values follow
[Qwen's thinking-mode configuration](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/generation_config.json).

```python
from trajectory import Client

Client().evals.create(
    bench_id="bm_<your-benchmark-id>",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "max_output_tokens_per_turn": 16384,
        "max_turns_per_trajectory": 200,
        "max_output_tokens_per_trajectory": 262144,
    },
)
```

For evaluations during training, set `val_top_p` and `val_top_k` in the training
options if you need the same sampling. Sierra's source is MIT-licensed; the pinned source and
its [license](https://github.com/sierra-research/tau2-bench/blob/fc0055dc4e0a316c3f83133267fbd6faaa770992/LICENSE)
are included in the runtime image.
