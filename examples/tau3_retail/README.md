# Tau 3 Retail: train a small model on customer-service tasks

This example packages Sierra's public Tau 3 Retail tasks from τ²-bench commit
`fc0055dc4e0a316c3f83133267fbd6faaa770992` (v1.0.1) and trains `Qwen/Qwen3.5-4B` on them.
It follows the [GSM8K workflow](../gsm8k/README.md) with a multi-turn benchmark: validate one
task, smoke-test, upload, measure a baseline, train, and evaluate the checkpoint.

The agent plays a retail customer-service representative. It follows a written policy and
uses 17 tools to look up users, orders, and products, then cancels, modifies, returns, or
exchanges orders. A simulated customer drives each conversation from private instructions. The
policy requires explicit customer confirmation before any database change, so the simulated
customer is part of the benchmark rather than a convenience.

The runtime preserves the upstream task bytes, policy, database, user simulator prompts,
and evaluator. Reward is upstream's, `1.0` only when the final database matches the expected
state and the task's natural-language assertions pass, with one gate: a conversation in which
any agent reply contains `</think>` scores `0.0` (see [reasoning leaks](#watch-for-reasoning-leaks)).

| | Tau 3 Banking | Tau 3 Retail |
| --- | --- | --- |
| Tasks | 97, sorted into 80 train / 17 test | 114, upstream's official 74 train / 40 test |
| Retrieval | `bm25_grep` over 698 documents | None; policy and tools only |
| Reward basis | Database (87), actions (9), judged assertion (1) | Database plus judged assertions; 40 tasks define assertions |

In upstream's published `gpt-4.1-mini` retail transcripts, conversations take 13 agent turns at
the median and 28 at most, and the model passes 66% of tasks. Expect a 4B model to start lower.

## How the pieces fit

- [`ingest.py`](ingest.py) creates one `TaskSpec` per upstream task ID with its official split
  and a `run_command` that runs that single task.
- [`runtime/tau3_retail/agent.py`](runtime/tau3_retail/agent.py) runs the task through
  upstream's `run_single_task`, routes the agent's model calls through the Trajectory session,
  logs upstream's reward (zeroed on leaked reasoning) and a protocol event, and completes the
  trajectory. If the runner
  raises, it completes the trajectory with `ERROR` before re-raising.
- [`runtime/tau3_retail/protocol.py`](runtime/tau3_retail/protocol.py) pins the source commit,
  the official split, and a SHA-256 over `tasks.json`, `split_tasks.json`, `db.json`, and
  `policy.md`. The image build and every task refuse to run if any of these drift.
- [`runtime/tau3_retail/user_simulator.py`](runtime/tau3_retail/user_simulator.py) retries
  simulated-customer responses with neither text nor tool calls, at most three times, without
  advancing conversation state. It does not fabricate customer messages.
- [`train.py`](train.py) evaluates, trains, monitors progress, and compares the checkpoint.

The agent's calls go to the model being evaluated or trained. The simulated customer and the
assertion judge use `gpt-5.4-mini-2026-03-17` through a separate LiteLLM client and the
organization's `OPENAI_API_KEY` secret, so those calls never reach the trained model. Each
rollout makes roughly one customer call per agent turn, so a training step of 8 tasks × 8
samples makes several hundred OpenAI calls in addition to training compute.

## 1. Set up

Start from the cookbook root with your API key:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

[Choose or create an agent](../../README.md#choose-or-create-an-agent). For a new one:

```bash
uv run --with trajectory-sdk==0.9.16 python -c \
  'from trajectory import Client; Client().agents.create(name="tau3-retail-cookbook", description="Tau 3 Retail customer-service evaluation and training")'
```

The simulated customer and judge need the organization secret `OPENAI_API_KEY`. Check whether
it already exists before uploading a value:

```python
import os

from trajectory import Client

client = Client()
if "OPENAI_API_KEY" not in {secret.name for secret in client.secrets.list()}:
    client.secrets.create(name="OPENAI_API_KEY", value=os.environ["OPENAI_API_KEY"])
```

## 2. Validate one task

```bash
uv run examples/tau3_retail/ingest.py --agent-name tau3-retail-cookbook --task-diagnostic
```

This builds the runtime and runs training task `0` on `Qwen/Qwen3.5-4B` without uploading a
benchmark. It prints the diagnostic ID, status, task report, and reward. Zero is a valid graded
outcome; a failure here points to packaging, model routing, the simulated customer, or grading.
See [task diagnostics](../task_diagnostics.md) to inspect a result again.

## 3. Smoke-test one held-out task

```bash
uv run examples/tau3_retail/ingest.py --agent-name tau3-retail-cookbook --smoketest
uv run examples/tau3_retail/train.py --agent-name tau3-retail-cookbook \
  --bench-name tau3-retail-smoketest --smoketest
```

The first command uploads `tau3-retail-smoketest` with one train and one test task and waits for
its image. The second evaluates the test task and prints `smoketest_reward`.

## 4. Upload the benchmark

```bash
uv run examples/tau3_retail/ingest.py --agent-name tau3-retail-cookbook
```

This uploads all 114 tasks as `tau3-retail-trajectory-sdk`, waits for the image, and runs a
benchmark diagnostic on up to 10 tasks.

## 5. Train and compare

```bash
uv run examples/tau3_retail/train.py --agent-name tau3-retail-cookbook \
  --bench-name tau3-retail-trajectory-sdk --num-steps 20
```

The script evaluates the base model on the 40 test tasks, trains, evaluates the final
checkpoint on the same tasks, and prints `baseline_reward`, `final_reward`, and
`reward_delta`. Evaluation and training share these rollout options:

| Option | Value | Reason |
| --- | --- | --- |
| `disable_thinking` | `True` | Reasoning on every one of 13–28 turns multiplies context and cost. |
| `max_output_tokens_per_step` | `2048` | Covers the 4B's normal replies (99.5th percentile ~5,500 characters) and stops runaway ones. Matches `max_tokens` in `protocol.py`. |
| `max_turns_per_trajectory` | `100` | Counts agent and environment turns; retail uses up to ~80. |
| `train_batch_size` | `8` | 8 tasks × 8 samples per step; about two passes over 74 tasks in 20 steps. |
| `learning_rate` | `5e-5` | Half the model default; `1e-4` drifted into reasoning leaks, see below. |
| `lr_warmup_steps` | `3` | The default of 15 keeps a 20-step run in warmup; at `3e-5` it did not learn. |

The model's other defaults apply, including GSPO with 8 samples per task.
Retail tool responses stay under 3,400 characters in upstream transcripts, within the
16,384-character training limit, so training and evaluation see the same conversations.
Discover the options and their bounds with
`client.training.list_options(bench_id=..., base_model_slug="Qwen/Qwen3.5-4B")`.

Both evaluations run each of the 40 test tasks 4 times (`evaluation_samples_per_task`). With
one sample per task, two evaluations of nearly the same model differed by `0.15`: 12 tasks
flipped to failing and 6 to passing. Treat a small `reward_delta` from one short run as
inconclusive.

### Watch for reasoning leaks

`disable_thinking` pre-fills an empty `<think></think>` block; it does not stop the model from
emitting `</think>` itself. Upstream's reward checks the final database, not how the agent
talks. In an ungated run at learning rate `1e-4` with 2 warmup steps, Qwen 3.5 4B learned to
end a draft reply with `</think>` and start again, repeating sentences. Leaky conversations
that still reached the right database state earned full reward, so training reinforced them.
Leaks went from 0% of sampled rollouts at steps 0–2 to 88% at steps 13–15, average reply
length grew from ~500 to ~7,200 characters, and the training solve rate on the same tasks fell
from 0.64 to 0.54.

The runtime therefore scores any conversation with a leaked `</think>` as `0.0`. The base
model leaked in none of 40 baseline conversations, so the gate does not change its score.
Each trajectory's `tau3_retail_protocol` event records `reasoning_leaks`, the number of agent
replies containing `</think>`, and `upstream_reward`, the ungated score.

## Inspect existing work

Interrupting a command does not cancel accepted work. Use the printed IDs:

```python
from trajectory import Client

client = Client()
print(client.diagnostics.get_status("YOUR_DIAGNOSTIC_ID"))
print(client.evals.runs.retrieve("YOUR_EVAL_RUN_ID").reward_mean)
run = client.training.runs.retrieve("YOUR_TRAINING_RUN_ID")
print(run.status, client.training.runs.progress("YOUR_TRAINING_RUN_ID"), run.failure)
```

Each trajectory carries a `tau3_retail_protocol` event with the source commit and digest, the
customer and judge model, the upstream termination reason, the number of policy calls, and
the number of reasoning leaks, and the ungated upstream reward.
The reward's explanation holds upstream's full `reward_info`, including the database check.

## Known upstream task issues

Upstream ships reports in `data/tau2/domains/retail/task_issues/` for tasks `4` and `7` (train)
and `5` (test). This example keeps them so the split matches upstream and published results.

## Run the offline tests

```bash
uv run --no-project --with trajectory-sdk==0.9.16 --with pytest \
  python -m pytest examples/tau3_retail/test_tau3_retail.py

cd examples/tau3_retail/runtime
uv sync --frozen
git clone --filter=blob:none https://github.com/sierra-research/tau2-bench /tmp/tau2-bench
git -C /tmp/tau2-bench checkout fc0055dc4e0a316c3f83133267fbd6faaa770992
TAU2_DATA_DIR=/tmp/tau2-bench/data uv run --with pytest python -m pytest tests
```

The runtime tests run one real retail task through upstream's orchestrator and evaluator with a
scripted customer and a mocked Trajectory API. Without `TAU2_DATA_DIR`, the data checks skip.
