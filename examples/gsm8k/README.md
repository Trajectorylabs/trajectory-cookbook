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
uv run --with trajectory-sdk==0.9.23 --with httpx python
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
uv run --with trajectory-sdk==0.9.23 python -c \
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
split, then waits for the runtime image to be ready. Both ingestion helpers only upload and
wait; diagnostics are separate calls. Benchmark-wide diagnostics are reserved for step 6.

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

A printed ID means the request was accepted. `pending` with zero completed steps means
training is waiting; increasing step counts show progress. Continue only after `succeeded`.
Keep the evaluation and training IDs to [inspect existing work](#inspect-existing-work).

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
then calls `benchmark_diagnostic` on a subset of the uploaded tasks. It selects up to 10 tasks,
prioritizing distinct runtimes, and prints detailed task outcomes and failures. It does not
repeat the single-task diagnostic from step 3.

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

## Inspect existing work

Interrupting a local command does not cancel work already accepted by the service. Inspect
its saved ID before submitting another run. Replace the matching placeholder in each command.

For an evaluation:

```python
from trajectory import Client

client = Client()
eval_run_id = "YOUR_EVAL_RUN_ID"
progress = client.evals.runs.progress(eval_run_id)
result = client.evals.runs.retrieve(eval_run_id)
print(result.status, progress.terminal_rollouts, progress.total_rollouts, result.reward_mean)
if progress.status == "failed":
    print(progress.failure)
for trajectory in client.trajectories.list(eval_run_id=eval_run_id):
    print(trajectory.trajectory_id, trajectory.status, trajectory.reward, trajectory.rollout)
```

`trajectories.list()` returns an iterable `SyncPage`. Use `list(...)` to count all returned
trajectories or `.items` for the current page; `len(page)` is not supported.

`completed`, `failed` and `cancelled` are terminal eval states. A completed run can contain
ungraded attempts; inspect individual trajectories before interpreting its mean as a benchmark
score. `client.evals.rewards.list_trajectory_rewards(eval_run_id)` returns run rewards, which
can include a platform zero for a cut-off attempt. Check the trajectory's logged reward before
calling that a native grade.

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

For a failed full-benchmark diagnostic, fix the reported problem and rerun the
[full upload](#6-expand-to-the-80-task-example). It publishes a new version and checks a subset
of its tasks before you retry training.

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

Use `client.training.rewards.list_trainer_rewards(run_id)` for training rewards and
`client.training.rewards.list_held_out_rewards(run_id)` for held-out rewards. For a partial
held-out round, inspect its sample counts and `round_completed` before comparing its mean.
To resume training, pass the chosen `checkpoint_id` as `parent_checkpoint_id` to
`client.training.create`, with that checkpoint's `base_model_slug`.

In the platform, the agent's **Train** and **Compare** tabs show these runs.
The SDK's [platform guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/platform.mdx)
gives the direct paths for saved IDs.

## Optional: deploy and query the trained checkpoint

After training succeeds, replace `YOUR_TRAINING_RUN_ID` with the run you want to deploy.
Use its actual final step count; the commands above use three steps. Use a new model slug
for this example; `role="production"` makes it active when deployment finishes.

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
```

Wait for `DEPLOYED`, then follow the SDK's
[deployment guide](https://github.com/Trajectorylabs/trajectory-platform/blob/main/docs/guides/deployments.mdx)
to query and complete a serving trajectory. For an existing production slug, create a
`role="test"` deployment and query its exact route before promoting it. Serving answers
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
