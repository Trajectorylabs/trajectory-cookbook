# Tau 3 Retail

This example packages Sierra's public 114-task Retail benchmark from
[tau2-bench v1.0.1](https://github.com/sierra-research/tau2-bench/tree/fc0055dc4e0a316c3f83133267fbd6faaa770992) with upstream's official 74 train / 40 test
split, and trains `Qwen/Qwen3.5-4B` on it. The agent is a customer-service representative with
17 tools; upstream's simulated customer drives each conversation from private instructions, and
upstream's evaluator grades the final database and the task's natural-language assertions.

The adapter uses upstream's simulator and evaluator, including their final grade when a
conversation continues after an output-limited model reply. Its one grading exception is a
malformed agent tool call: upstream raises, while this example records zero and completes the
attempt so one invalid reply does not fail the evaluation. Sierra's too-many-errors stop keeps
its source grade; unexpected errors are ungraded.

Choose an agent using the [GSM8K setup guide](../gsm8k/README.md#2-choose-or-create-an-agent).
Register the organization secret `OPENAI_API_KEY` for the simulated customer and judge. The
default is upstream's `gpt-4.1-2025-04-14` at temperature 0. To use another model, pass
`--user-judge-model MODEL` to both ingestion commands; an `openrouter/` model uses the
`OPENROUTER_API_KEY` organization secret instead. Secret values are never included in the image
or upload.

```sh
uv run examples/tau3_retail/ingest.py --agent-name YOUR_AGENT_NAME --task-diagnostic
uv run examples/tau3_retail/ingest.py --agent-name YOUR_AGENT_NAME
uv run examples/tau3_retail/train.py --agent-name YOUR_AGENT_NAME \
  --bench-name tau3-retail-trajectory-sdk --num-steps 20
```

The first command runs one task without uploading. The second uploads all 114 tasks and runs a
benchmark diagnostic. The third evaluates the base model, trains, and evaluates the checkpoint,
running 160 test conversations (four per task) before and after training. It uses a 2,048-token output
cap, learning rate `5e-5`, and 3 warmup steps; the model default of 15 warmup steps leaves a
20-step run barely trained. These agent and sampling settings are cookbook choices, so scores
here are not upstream leaderboard scores.
