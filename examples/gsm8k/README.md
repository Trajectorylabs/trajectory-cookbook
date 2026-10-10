# GSM8K: the full evaluation and training workflow

After [your first task with T Factory](../t_factory/README.md), use this walkthrough to take a
real benchmark through managed validation, evaluation, training, and checkpoint comparison.
Start with one training problem and one held-out test problem, then scale to the example's
64 training and 16 test problems. These 80 problems are a sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math), not the entire upstream dataset.

The harness asks the model to solve a math problem and call `submit_answer`. It records reward
`1.0` for the correct numeric answer and `0.0` for an incorrect or text-only answer. The model
receives the question, while the grader retains the reference answer.

## 1. Set up and confirm your organization

Use the checkout and API key from the first lesson. If you are starting here, you need Python
3.11+, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), and an organization API
key from **Settings → API keys** in the [platform](https://platform.trajectory.ai):

```bash
git clone https://github.com/Trajectorylabs/trajectory-cookbook.git
cd trajectory-cookbook
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

Run all remaining commands from the repository root. `uv` installs each script's dependencies.
Runtime images are built remotely; local Docker is not required. Confirm the organization
before starting managed runs:

```bash
uv run --with trajectory-sdk python - <<'PY'
from trajectory import Client

client = Client()
print(client.organizations.retrieve_current())
for agent in client.agents.list():
    print(agent.name, agent.description)
PY
```

The SDK uses `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` for another deployment.

## 2. Choose or create an agent

Follow the shared [agent selection guidance](../../README.md#choose-or-create-an-agent).
Use an explicitly selected agent, reuse one whose purpose fits, or create one for a new use case.
For a new GSM8K project:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

The remaining commands use `gsm8k-cookbook`; substitute your selected agent's name if different.
Reuse that agent on later runs without repeating the creation command.

## 3. Validate one task before uploading a benchmark

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook" --diagnose-only
```

This downloads one training problem, packages its task file with the GSM8K harness and grader,
builds the runtime, and runs a diagnostic on `Qwen/Qwen3.5-4B`. The command prints its
`diagnostic_id`, polls status, then prints the task report and `diagnostic_reward`.

Continue when it exits successfully: the diagnostic and its task completed without failure,
and a reward was recorded. Zero is a valid graded answer. A ready image only confirms the
build; this step checks that model execution and grading work too.

If it fails, inspect the printed run-wide and per-task failures, fix the reported runtime or
grading problem, and repeat the diagnostic. Keep its ID to [inspect it later](#inspect-existing-work).
For a different harness, use the [task upload validation recipe](../task_diagnostics.md).

## 4. Register the smallest training benchmark

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook" --small
```

The default managed training flow needs training and test splits, so this registers **one
training problem and one different problem from GSM8K's test split** under `gsm8k-small`.
This is a separate benchmark from the one-task diagnostic. The uploader prints `agent_name`
and `bench_id`, then waits for the runtime image to be ready. Save that ID and wait for
successful exit before continuing.

This small benchmark lets you train on one task and evaluate on one held-out task before
uploading the larger example. It checks the integration; one test problem cannot establish
whether training improves the model generally.

## 5. Evaluate a baseline, train, and evaluate the checkpoint

Replace `YOUR_SMALL_BENCH_ID` with the ID from step 4:

```bash
uv run examples/gsm8k/train.py --bench-id YOUR_SMALL_BENCH_ID --num-steps 3
```

The script runs the complete sequence:

1. Check the registered task list and its explicit train/test splits.
2. Evaluate the base `Qwen/Qwen3.5-4B` model on the one held-out task and record its mean reward.
3. Train for three optimizer steps using the one training task.
4. Wait for successful training, retrieve the final checkpoint, and evaluate it on the same test task.
5. Print `baseline_reward`, `final_reward`, and `reward_delta` (final minus baseline).

One training task group can contain multiple model samples according to the model's defaults.
The script uses one task group per step for this small benchmark. The training task is never
used as the held-out evaluation task.

### Watch actual progress

The command prints the baseline `eval_run_id` and completed/total rollouts, then the
`training_run_id`, lifecycle status, and `steps=completed/total`.

- A printed ID means the request was accepted.
- `status=pending` and zero completed steps means training is waiting.
- Increasing completed steps show training progress.
- `status=succeeded` means training finished successfully; checkpoint evaluation follows.

Failed or cancelled runs stop the script with an error. Keep the evaluation and training IDs;
use the platform's agent pages or the [inspection commands](#inspect-existing-work) to follow
an existing run after an interruption. Reward may stay unchanged or fall in a short run.

For an existing benchmark where you want only evaluation or only training, use the
[individual SDK calls](../../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform).
The `train.py` command above always performs baseline evaluation, training, and checkpoint evaluation.

## 6. Expand to the 80-task example

After the small run completes, upload the example's **64 training and 16 held-out test problems**:

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook"
```

This registers `gsm8k-trajectory-sdk` with the same harness and grader. Save the new `bench_id`
and wait for successful image readiness, then replace `YOUR_BENCH_ID` below:

```bash
uv run examples/gsm8k/train.py --bench-id YOUR_BENCH_ID --num-steps 3
```

The script now evaluates all 16 held-out tasks and uses four training task groups per step.
It prints the same progress and reward comparison as the small run. Increasing `--num-steps`
lets you run longer; use repeated runs and a sufficiently large frozen test set to assess improvement.

Uploading the same benchmark name under the same agent creates a new version with a new ID.
Use the ID from the upload you intend to evaluate or train; the small and larger benchmark IDs
are distinct. Both runs start from the base model unless you explicitly use the SDK's
`parent_checkpoint_id` option to continue from a checkpoint.

## Inspect existing work

Interrupting a local command does not cancel work already accepted by the service. Inspect
its saved ID before submitting another run. Replace the matching placeholder in each command.

For a diagnostic:

```bash
uv run --with trajectory-sdk python - <<'PY'
from trajectory import Client

client = Client()
diagnostic_id = "YOUR_DIAGNOSTIC_ID"
status = client.diagnostics.get_status(diagnostic_id)
print(status)
if status.status in {"completed", "failed", "cancelled"}:
    print(client.diagnostics.get_diagnostics(diagnostic_id).to_json())
PY
```

For an image build:

```bash
uv run --with trajectory-sdk python - <<'PY'
from trajectory import Client

print(Client().benchmarks.images.list("YOUR_BENCH_ID").to_json())
PY
```

For training:

```bash
uv run --with trajectory-sdk python - <<'PY'
from trajectory import Client

client = Client()
run_id = "YOUR_TRAINING_RUN_ID"
run = client.training.runs.retrieve(run_id)
progress = client.training.runs.progress(run_id)
print(run.status, progress.completed_steps, progress.total_steps)
print(run.failure)
PY
```

For other models and supported settings, inspect
`Client().training.list_options(bench_id="YOUR_BENCH_ID")`. Pass a supported model to `train.py`
with `--model`; its advertised settings must support the example's training options.

## Optional: deploy and query the trained checkpoint

After training succeeds, replace `YOUR_TRAINING_RUN_ID` with the run you want to deploy.
Use its actual final step count; the commands above use three steps.

```bash
uv run --with trajectory-sdk python - <<'PY'
from trajectory import Client

client = Client()
checkpoint = client.training.checkpoints.retrieve("YOUR_TRAINING_RUN_ID", step_index=3)
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="gsm8k-trained",
    role="production",
)
print(deployment.deployment_id)
response = client.chat.completions.create(
    model="gsm8k-trained",
    messages=[{"role": "user", "content": "What is 17 times 6?"}],
)
print(response.choices[0].message.content)
PY
```

Deployment activates the production `gsm8k-trained` model slug. This query demonstrates serving;
the checkpoint evaluation in step 5 uses the benchmark's `submit_answer` tool and grader to measure reward.

## Adapt this workflow to your benchmark

The integration has three parts:

- [`ingest.py`](ingest.py): load source rows, stage task files, and create `TaskSpec` objects with
  explicit splits and a per-task `run_command`.
- [`runtime/gsm8k_harness.py`](runtime/gsm8k_harness.py): call the selected model, grade its answer,
  record reward, and complete the same trajectory.
- [`train.py`](train.py): evaluate, train, monitor progress, and compare the final checkpoint.

When changing the dataset or harness, preserve the benchmark's native task inputs, tools,
private references, and grading rules. GSM8K exposes only `submit_answer`; a harness with file
or shell tools must keep reference answers outside the model's access.

Use [runtime packaging guidance](../../README.md#package-a-benchmark-runtime) and the
[Harvey](../harvey_labs.md) or [Inspect](../inspect.md) recipes for an existing harness.
Benchmark-specific source inspection may be necessary to understand those inputs and grading
rules. The public guides cover the SDK lifecycle.
