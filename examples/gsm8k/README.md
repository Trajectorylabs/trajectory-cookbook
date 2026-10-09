# SDK onboarding: validate a task and train on GSM8K

Follow these commands to authenticate, choose an agent, validate one task, upload a benchmark,
and observe training progress. The example uses the public
[GSM8K dataset](https://github.com/openai/grade-school-math): 64 training problems and 16 held-out
test problems. It includes the harness, grader, Dockerfile, uploader, and training script.

For your own benchmark, follow the same stages and use the
[adaptation guidance](#adapt-this-example) below. Your benchmark's harness and grading rules
still need to be understood and preserved.

## 1. Install and authenticate

You need Python 3.11 or newer, Git, [uv](https://docs.astral.sh/uv/getting-started/installation/),
and an organization API key from **Settings → API keys** in the
[Trajectory platform](https://platform.trajectory.ai). Runtime images are built remotely;
a local Docker installation is not needed. These commands run diagnostics, evaluation,
and training in your organization.

```bash
git clone https://github.com/Trajectorylabs/trajectory-cookbook.git
cd trajectory-cookbook
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with 'trajectory-sdk>=0.9.14' python -c \
  'from trajectory import Client; print(Client().organizations.retrieve_current())'
```

Run all remaining commands from this repository root. `uv run` installs the dependencies declared
by each script, including SDK 0.9.14 or later. Verify that the printed organization is the one you
intend to use. The SDK connects to `https://api.trajectory.ai` by default; set
`TRAJECTORY_BASE_URL` if you are using another deployment.

## 2. Choose or create an agent

Follow the shared [agent selection guidance](../../README.md#choose-or-create-an-agent).
Use an explicitly selected agent, or reuse one whose purpose matches this project. For a new
use case, create a descriptively named agent even if unrelated agents already exist.

To see the available names and descriptions:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python - <<'PY'
from trajectory import Client

for agent in Client().agents.list():
    print(agent.name, agent.description)
PY
```

For a new GSM8K project:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

The commands below use `gsm8k-cookbook`. If you selected an existing agent, use its name instead.
On subsequent runs, reuse the same agent without repeating the creation command.

## 3. Validate one task

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook" --diagnose-only
```

This packages one training problem with the same harness used by the full benchmark, builds
its runtime, and starts a task diagnostic. It prints a `diagnostic_id`, polls its status, and
prints the task report and `diagnostic_reward`.

Continue when the command exits successfully: the diagnostic and its task completed without
failure, and a reward was recorded. A reward of `0.0` is a valid graded wrong answer. A pending
or running diagnostic has not finished validating the task. Image readiness alone does not
establish that the harness can execute or grade it.

If the diagnostic fails, read the printed run-wide `failure` and each task's `failure`. Correct
the reported runtime, model-call, or grading problem and repeat this step before uploading the
full benchmark. Keep the diagnostic ID; after an interruption, inspect the existing run:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python - <<'PY'
from trajectory import Client

client = Client()
diagnostic_id = "YOUR_DIAGNOSTIC_ID"
status = client.diagnostics.get_status(diagnostic_id)
print(status)
if status.status in {"completed", "failed", "cancelled"}:
    print(client.diagnostics.get_diagnostics(diagnostic_id).to_json())
PY
```

Replace `YOUR_DIAGNOSTIC_ID` with the printed ID. See the
[task diagnostics recipe](../task_diagnostics.md) to apply this check to your own task.

## 4. Upload the benchmark and wait for its image

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook"
```

The uploader registers all 80 tasks, prints `agent_name` and `bench_id`, then waits for the
runtime image to become ready. Wait for the command to exit successfully before continuing.
Keep the printed benchmark ID. Uploading the same benchmark name under the same agent creates
a new version with a new ID; use that new ID for subsequent runs.

If image preparation fails or waiting is interrupted, inspect the registered benchmark:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python - <<'PY'
from trajectory import Client

client = Client()
print(client.benchmarks.images.list("YOUR_BENCH_ID").to_json())
PY
```

Replace `YOUR_BENCH_ID` with the uploader's ID. A registered benchmark is not necessarily ready
for execution. The uploader uses `wait_for_benchmark_images()` to request the build and wait
for readiness; `images.list()` only reads its status.

## 5. Evaluate, train, and monitor progress

Use the benchmark ID from step 4. This command runs baseline evaluation, training, and checkpoint
evaluation together. For an existing benchmark where you only want to start an evaluation or
training run, use the cookbook’s [individual SDK calls](../../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform)
and the monitoring commands below.

```bash
uv run examples/gsm8k/train.py --bench-id YOUR_BENCH_ID --num-steps 3
```

The script evaluates a baseline on the 16 held-out tasks, starts three training steps on
`Qwen/Qwen3.5-4B`, and evaluates the final checkpoint on those same held-out tasks. It prints:

- The baseline `eval_run_id` and completed/total rollouts.
- The `training_run_id`, lifecycle status, and completed/total training steps.
- The checkpoint evaluation's reward and the change from baseline.

A printed training ID means the service accepted the request. `pending` with zero completed
steps means training is waiting; increasing completed steps demonstrate optimizer progress.
Only `status=succeeded` establishes successful training completion. The script stops with an
error if evaluation or training fails or is cancelled.

Keep the printed training ID. You can monitor that run from another terminal or after the
script is interrupted, without submitting it again:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python - <<'PY'
from trajectory import Client

client = Client()
run_id = "YOUR_TRAINING_RUN_ID"
run = client.training.runs.retrieve(run_id)
progress = client.training.runs.progress(run_id)
print(run.status, progress.completed_steps, progress.total_steps)
print(run.failure)
PY
```

Replace `YOUR_TRAINING_RUN_ID` with the printed ID. Open the agent's training page in the
platform to inspect the run and its trajectories. Interrupting the local script does not
cancel a run already accepted by the service.

The example defaults to `Qwen/Qwen3.5-4B`. To inspect available training models and their options:

```bash
uv run --with 'trajectory-sdk>=0.9.14' python - <<'PY'
from trajectory import Client

for model in Client().training.list_options(bench_id="YOUR_BENCH_ID").models:
    print(model.base_model_slug, model.options)
PY
```

Use `--model` to select a supported alternative whose advertised options support the example's
settings. A short run verifies the integration; use repeated runs and a sufficiently large
frozen test set to measure whether training improves the model.

## Adapt this example

The example has three integration points:

- [`ingest.py`](ingest.py) reads the source data, writes local task files, and maps them to
  `TaskSpec` objects with explicit train/test splits. Each task's `run_command` selects its file.
- [`runtime/gsm8k_harness.py`](runtime/gsm8k_harness.py) calls the model, grades the answer,
  logs the reward, and completes the trajectory. It uses the same trajectory ID for model
  calls, reward reporting, and completion. Managed runtimes supply credentials to `Client()`.
- [`train.py`](train.py) evaluates, trains, monitors progress, and compares the final checkpoint.

The GSM8K harness exposes only a `submit_answer` tool, with no file-reading tool. It grades the
submitted number against the private reference; text-only answers receive zero reward. When
adapting a harness with shell or file tools, keep private references outside the model's access.
Preserve your benchmark's native solving and grading logic, including its handling of errors
and genuine zero rewards.

Use the cookbook's [runtime packaging guidance](../../README.md#package-a-benchmark-runtime)
and [Harvey](../harvey_labs.md) or [Inspect](../inspect.md) recipes for existing harnesses.
The runnable example and public recipes define the SDK integration; benchmark-specific source
inspection may still be necessary to identify that benchmark's inputs, tools, and grader.
