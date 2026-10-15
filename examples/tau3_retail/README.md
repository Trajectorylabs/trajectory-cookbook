# Tau 3 Retail

This example packages Sierra's public 114-task Retail benchmark from Tau 2 commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992` (v1.0.1) with upstream's official 74 train / 40 test
split, and trains `Qwen/Qwen3.5-4B` on it. The agent is a customer-service representative with
17 tools; upstream's simulated customer drives each conversation from private instructions, and
upstream's evaluator grades the final database and the task's natural-language assertions.

Two agent failures also score `0.0`. A reply containing `</think>`: `disable_thinking` does not
stop the model from emitting the tag, and in an ungated run at learning rate `1e-4` training
rewarded leaky replies until 88% of rollouts leaked. And tool-call arguments that are not valid
JSON: upstream raises on them, and one crashed rollout fails a whole evaluation. Each
trajectory's `tau3_retail_protocol` event records both counts and the ungated upstream reward.

Choose an agent using the [Cookbook guidance](../../README.md#choose-or-create-an-agent).
Register the organization secret `OPENAI_API_KEY` for the simulated customer and judge
(`gpt-5.4-mini-2026-03-17`); its value is never included in the image or upload.

```sh
uv run examples/tau3_retail/ingest.py --agent-name YOUR_AGENT_NAME --task-diagnostic
uv run examples/tau3_retail/ingest.py --agent-name YOUR_AGENT_NAME
uv run examples/tau3_retail/train.py --agent-name YOUR_AGENT_NAME \
  --bench-name tau3-retail-trajectory-sdk --num-steps 20
```

The first command runs one task without uploading. The second uploads all 114 tasks and runs a
benchmark diagnostic. The third evaluates the base model, trains, and evaluates the checkpoint,
running each test task 4 times; with one sample, two evaluations of nearly the same model
differed by `0.15`. It uses a 2,048-token output cap and learning rate `5e-5` with 3 warmup
steps; the model default of 15 warmup steps leaves a 20-step run barely trained. In a 20-step
run, the test reward went from `0.60` to about `0.66`.
