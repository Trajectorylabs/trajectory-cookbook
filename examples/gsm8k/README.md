# GSM8K: the full evaluation and training workflow

After [your first task with T Factory](../t_factory/README.md), use this walkthrough to take a
real benchmark through managed validation, evaluation, training, and checkpoint comparison.
Use smoke-test helpers with one training problem and one held-out test problem, then scale to the example's
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
before starting managed runs. For the Python examples, start a session from the repository root:

```bash
uv run --with trajectory-sdk --with httpx python
```

```python
from trajectory import Client

client = Client()
print(client.organizations.retrieve_current())
for agent in client.agents.list():
    print(agent.name, agent.description)
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

## 3. Validate one task and upload a smoke-test benchmark

In your Python session:

```python
from examples.gsm8k.ingest import ingest_smoketest

bench_id = ingest_smoketest(agent_name="gsm8k-cookbook")
print(bench_id)
```

`ingest_smoketest` runs these checks by default:

1. `task_diagnose` downloads one training problem, packages its task file with the harness and
   grader, builds the runtime, and runs that task on `Qwen/Qwen3.5-4B`. This catches packaging,
   execution, and grading errors before uploading the benchmark.
2. Upload `gsm8k-smoketest` with one training problem and one different problem from GSM8K's
   test split, then wait for its runtime image to be ready.
3. `benchmark_diagnostic` checks the uploaded tasks and their runtime configuration. Benchmark
   diagnostics select up to 10 tasks, prioritizing distinct runtimes.

Both diagnostics print their ID, status, task report, and reward. The helper returns a
`bench_id` only after both checks complete successfully and record a reward. Zero is a valid
graded answer; image readiness alone does not prove execution or grading worked.

If a check fails, inspect its printed run-wide and per-task failures, fix the reported runtime
or grading problem, and [repeat the relevant diagnostic](#inspect-existing-work).
For a different harness, use the [task upload validation recipe](../task_diagnostics.md).

## 4. Evaluate one held-out task and confirm a grade

After the uploaded benchmark passes its diagnostic, run a separate one-task evaluation to see
that the evaluation path returns a grade before training:

```python
from trajectory import Client
from examples.gsm8k.train import eval_smoketest

client = Client()
bench_id = "YOUR_SMOKETEST_BENCH_ID"
baseline_reward = eval_smoketest(client, bench_id)
print(f"baseline_reward={baseline_reward:.6f}")
```

Replace `YOUR_SMOKETEST_BENCH_ID` with the `bench_id` returned by `ingest_smoketest`.
`eval_smoketest` evaluates one held-out task, prints its evaluation ID and progress, and returns
the reward. It stops on
failure or cancellation; zero is a valid graded answer. This catches basic execution and
grading errors before a larger run. One test problem cannot measure generalization.

## 5. Train on one task and evaluate the checkpoint

Continue in the same Python session. Use one task group per step for the smoke-test benchmark:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "disable_thinking": True,
        "num_steps": 3,
        "train_batch_size": 1,
        "max_output_tokens_per_step": 2048,
        "max_turns_per_trajectory": 1,
        "max_response_chars_per_tool_call": 128,
    },
)
run_id = training.training_run_id
print(f"training_run_id={run_id}")
```

One task group can contain multiple model samples according to the model's defaults.
The training task is separate from the held-out evaluation task.

### Watch actual progress

```python
import time

while True:
    run = client.training.runs.retrieve(run_id)
    progress = client.training.runs.progress(run_id)
    print(f"status={run.status} steps={progress.completed_steps}/{progress.total_steps}")
    if run.status in {"succeeded", "failed", "cancelled"}:
        break
    time.sleep(30)

if run.status != "succeeded":
    raise RuntimeError(f"training ended with status={run.status}: {run.failure}")
```

A printed ID means the request was accepted. `pending` with zero completed steps means
training is waiting; increasing step counts show progress. Continue only after `succeeded`.
Keep the evaluation and training IDs to [inspect existing work](#inspect-existing-work).

After successful training, evaluate the checkpoint on the same held-out task:

```python
checkpoint = client.training.checkpoints.retrieve(run_id, step_index=3)
final_reward = eval_smoketest(client, bench_id, parent_checkpoint_id=checkpoint.checkpoint_id)
print(f"baseline_reward={baseline_reward:.6f}")
print(f"final_reward={final_reward:.6f}")
print(f"reward_delta={final_reward - baseline_reward:+.6f}")
```

Reward may stay unchanged or fall in a short run. For an existing benchmark where you want
only evaluation or only training, use the [individual SDK calls](../../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform).

## 6. Expand to the 80-task example

After the smoke run completes, upload the example's **64 training and 16 held-out test problems**:

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook"
```

The full uploader follows the same sequence: diagnose one task, register `gsm8k-trajectory-sdk`,
wait for the runtime image, then diagnose the uploaded benchmark. Wait for successful exit
before training; a printed `bench_id` alone does not mean the diagnostics passed. Save the new
ID and replace `YOUR_BENCH_ID` below:

```bash
uv run examples/gsm8k/train.py --bench-id YOUR_BENCH_ID --num-steps 3
```

This command runs baseline evaluation, training, and checkpoint evaluation. It evaluates all
16 held-out tasks and uses the normal batch size of four training task groups per step.
It prints training progress and the baseline/final reward comparison. Increasing `--num-steps`
lets you run longer; use repeated runs and a sufficiently large frozen test set to assess improvement.

Uploading the same benchmark name under the same agent creates a new version with a new ID.
Use the ID from the upload you intend to evaluate or train; the smoke-test and larger benchmark IDs
are distinct. Both runs start from the base model unless you explicitly use the SDK's
`parent_checkpoint_id` option to continue from a checkpoint.

## Inspect existing work

Interrupting a local command does not cancel work already accepted by the service. Inspect
its saved ID before submitting another run. Replace the matching placeholder in each command.

For a diagnostic:

```python
from trajectory import Client

client = Client()
diagnostic_id = "YOUR_DIAGNOSTIC_ID"
status = client.diagnostics.get_status(diagnostic_id)
print(status)
if status.status in {"completed", "failed", "cancelled"}:
    print(client.diagnostics.get_diagnostics(diagnostic_id).to_json())
```

After fixing a reported problem, rerun only the relevant check. For a task built from local
runtime files:

```python
from examples.gsm8k.ingest import task_diagnose

task_diagnose(agent_name="gsm8k-cookbook")
```

For a benchmark that is already uploaded and whose images are ready:

```python
from examples.gsm8k.ingest import benchmark_diagnostic

benchmark_diagnostic(bench_id="YOUR_BENCH_ID")
```

For an image build:

```python
from trajectory import Client

print(Client().benchmarks.images.list("YOUR_BENCH_ID").to_json())
```

For training:

```python
from trajectory import Client

client = Client()
run_id = "YOUR_TRAINING_RUN_ID"
run = client.training.runs.retrieve(run_id)
progress = client.training.runs.progress(run_id)
print(run.status, progress.completed_steps, progress.total_steps)
print(run.failure)
```

For other models and supported settings, inspect
`Client().training.list_options(bench_id="YOUR_BENCH_ID")`. Pass a supported model to `train.py`
with `--model`; its advertised settings must support the example's training options.

## Optional: deploy and query the trained checkpoint

After training succeeds, replace `YOUR_TRAINING_RUN_ID` with the run you want to deploy.
Use its actual final step count; the commands above use three steps.

```python
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
