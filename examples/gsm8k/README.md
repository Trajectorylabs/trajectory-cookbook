# GSM8K: the full evaluation and training workflow

After [your first task with T Factory](../t_factory/README.md), use this walkthrough to take a
real benchmark through managed validation, evaluation, training, and checkpoint comparison.
For quick debugging, use one training problem and one held-out test problem, then scale to the example's
64 training and 16 test problems. These 80 problems are a sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math), not the entire upstream dataset.

The harness asks the model to solve a math problem and call `submit_answer`. It records reward
`1.0` for the correct numeric answer and `0.0` otherwise. No usable `submit_answer` call
means `MISSING_SUBMIT`; a length-capped reply means `TRUNCATION`. The model receives the
question, while the grader retains the reference answer.

## 1. Set up and confirm your organization

Use your API key from the first lesson. Start Python from the cookbook repository root:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
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

Follow the SDK's [agent selection guidance](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/agent.mdx).
For a new GSM8K project:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

Substitute your selected agent's name below if different.

## 3. Validate one task

In your Python session:

```python
from examples.gsm8k.ingest import task_diagnose

agent_name = "gsm8k-cookbook"
task_diagnose(agent_name=agent_name)
```

`task_diagnose` packages one training problem and its grader, builds the runtime, and runs
it on `Qwen/Qwen3.5-4B`. It prints the diagnostic ID, status, task report, and reward.
Zero is a valid grade. Fix packaging or grading failures and repeat the diagnostic.
For a different harness, use the [task upload validation recipe](../task_diagnostics.md).

## 4. Evaluate one held-out task and confirm a grade

Upload the smoke-test benchmark, then run a one-task evaluation to see that the evaluation
path returns a grade before training:

```python
from trajectory import Client
from examples.gsm8k.ingest import ingest_smoketest
from examples.gsm8k.train import eval_smoketest

client = Client()
agent_name = "gsm8k-cookbook"
bench_id = ingest_smoketest(agent_name=agent_name)
baseline_reward = eval_smoketest(client, bench_id)
print(f"baseline_reward={baseline_reward:.6f}")
```

`ingest_smoketest` uploads one training problem and one different problem from GSM8K's test
split, then waits for the runtime image. The full upload in step 6 runs a benchmark diagnostic.

`eval_smoketest` evaluates one held-out task and prints its evaluation ID, progress and grade
counts. It stops on failed rollouts or missing native grades. A graded wrong or cut-off answer is
still graded; the platform can score a cut-off answer as zero. Inspect its ending before
choosing training limits.
One test problem cannot measure generalization.

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
        "max_active_rollouts": 2,
        "max_output_tokens_per_step": 2048,
    },
)
run_id = training.training_run_id
print(f"training_run_id={run_id}")
```

One task group can contain multiple model samples according to the model's defaults.
`train_batch_size` does not cap launched attempts; `max_active_rollouts` bounds
concurrency, not total spend. Watch attempts during this smoke run.
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

Keep the evaluation and training IDs to [inspect a run](#inspect-a-run) if the local
process stops. Continue only after `succeeded`.

After successful training, evaluate the checkpoint on the same held-out task:

```python
checkpoint = client.training.checkpoints.retrieve(run_id, step_index=3)
final_reward = eval_smoketest(
    client, bench_id, parent_checkpoint_id=checkpoint.checkpoint_id
)
print(f"baseline_reward={baseline_reward:.6f}")
print(f"final_reward={final_reward:.6f}")
print(f"reward_delta={final_reward - baseline_reward:+.6f}")
```

Reward may stay unchanged or fall in a short run. For an existing benchmark, use the
[evaluation selectors](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/evaluations.mdx) and [training options](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/training.mdx) with the
corresponding SDK calls above.

## 6. Expand to the 80-task example

After the smoke run completes, upload the example's **64 training and 16 held-out test problems**
as a new version of the same benchmark:

```bash
uv run examples/gsm8k/ingest.py --agent-name "gsm8k-cookbook" --name "gsm8k-trajectory-sdk"
```

The command calls `ingest` to register `gsm8k-trajectory-sdk` and wait for the image,
then calls `benchmark_diagnostic` on a subset of the uploaded tasks and prints their outcomes.
It does not repeat the single-task diagnostic.

If the diagnostic fails, use the report to fix the task packaging or runtime and repeat this
upload. After successful exit, start training using the same agent and benchmark names:

```bash
uv run examples/gsm8k/train.py --agent-name "gsm8k-cookbook" --bench-name "gsm8k-trajectory-sdk" --num-steps 3
```

This command runs baseline evaluation, training, and checkpoint evaluation. It evaluates all
16 held-out tasks and uses the normal batch size of four training task groups per step.
It prints training progress and the baseline/final reward comparison. Increasing `--num-steps`
lets you run longer; use repeated runs and a sufficiently large frozen test set to assess improvement.

Uploading the same benchmark name under the same agent creates a new version. The comparison
script pins the baseline's `bench_id`; the manual smoke workflow uses the ID returned by
upload. Later versions cannot change either comparison's held-out tasks.
Training starts from the base model unless you set
`parent_checkpoint_id` to continue from a checkpoint.

## Inspect a run

Interrupting a local command does not cancel work already accepted by the service. Inspect
its printed ID before submitting another run. For an evaluation:

```python
from trajectory import Client

client = Client()
eval_run_id = "YOUR_EVAL_RUN_ID"
progress = client.evals.runs.progress(eval_run_id)
result = client.evals.runs.retrieve(eval_run_id)
print(result.status, progress.terminal_rollouts, progress.total_rollouts, result.reward_mean)
for attempt in client.trajectories.list(eval_run_id=eval_run_id):
    detail = client.trajectories.retrieve(
        attempt.trajectory_id, include_reward_components=True
    )
    print(attempt.trajectory_id, attempt.status, attempt.rollout, detail.reward_components)
```

An aggregate zero is not proof that the harness published a zero grade; inspect the
components and ending. For a diagnostic ID, use `client.diagnostics.get_diagnostics()`;
for a training ID, use `client.training.runs.retrieve()`. The SDK's
[evaluation](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/evaluations.mdx)
and [training](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/training.mdx)
guides cover run progress and recovery. The agent's **Train** and **Compare** tabs show
the same runs.

## Optional: deploy the trained checkpoint

After training succeeds, replace `YOUR_TRAINING_RUN_ID` with the run you want to deploy.
Use its actual final step count; the commands above use three steps.

```python
from trajectory import Client

client = Client()
checkpoint = client.training.checkpoints.retrieve("YOUR_TRAINING_RUN_ID", step_index=3)
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="gsm8k-trained",
    role="test",
)
print(deployment.deployment_id)
```

Follow the SDK's
[deployment guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/deployments.mdx)
to wait for `DEPLOYED`, test the exact deployment route, and promote it. Serving answers
do not replace the graded checkpoint evaluation in step 5.

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

Use the SDK's [runtime packaging guidance](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/benchmarks.mdx#package-the-harness-in-an-image) and the
[Harvey](../harvey_labs.md) or [Inspect](../inspect.md) recipes for an existing harness.
Benchmark-specific source inspection may be necessary to understand those inputs and grading
rules. The public guides cover the SDK lifecycle.
