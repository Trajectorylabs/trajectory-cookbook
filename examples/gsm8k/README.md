# GSM8K: the full evaluation and training workflow

After [your first task with T Factory](../t_factory/README.md), use this walkthrough to take a
real benchmark through managed validation, evaluation, training, and checkpoint comparison.
For quick debugging, use one training problem and one held-out test problem, then scale to the example's
64 training and 16 test problems. These 80 problems are a sample of the public
[GSM8K dataset](https://github.com/openai/grade-school-math), not the entire upstream dataset.

The harness asks the model to solve a math problem and call `submit_answer`. It records reward
`1.0` for the correct numeric answer and `0.0` for an incorrect or text-only answer. The model
receives the question, while the grader retains the reference answer.

## 1. Set up and confirm your organization

Use your API key from the first lesson. Start Python from the cookbook repository root:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk==0.9.20 --with httpx python
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
uv run --with trajectory-sdk==0.9.20 python -c \
  'from trajectory import Client; Client().agents.create(name="gsm8k-cookbook", description="GSM8K math evaluation and training")'
```

The remaining commands use `gsm8k-cookbook`; substitute your selected agent's name if different.
Reuse that agent on later runs without repeating the creation command.

## 3. Validate one task

In your Python session:

```python
from examples.gsm8k.ingest import task_diagnose

agent_name = "gsm8k-cookbook"
task_diagnose(agent_name=agent_name)
```

`task_diagnose` downloads one training problem, packages its task file with the harness and
grader, builds the runtime, and runs that task on `Qwen/Qwen3.5-4B`. This catches packaging,
execution, and grading errors before uploading the benchmark, with more detailed error
reports and logs to help debug failures.

The task diagnostic prints its ID, status, task report, and reward. Zero is a valid graded
answer. Continue to the smoke upload and evaluation after this diagnostic passes.

If a check fails, inspect its printed run-wide and per-task failures, fix the reported runtime
or grading problem, and [repeat the relevant diagnostic](#inspect-existing-work).
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
bench_name = "gsm8k-smoketest"
ingest_smoketest(agent_name=agent_name)
baseline_reward = eval_smoketest(client, agent_name, bench_name)
print(f"baseline_reward={baseline_reward:.6f}")
```

Use your chosen `agent_name` and `bench_name="gsm8k-smoketest"`.
`ingest_smoketest` uploads one training problem and one different problem from GSM8K's test
split, then waits for the runtime image to be ready. Both ingestion helpers only upload and
wait; diagnostics are separate calls. Benchmark-wide diagnostics are reserved for step 6.

`eval_smoketest` evaluates one held-out task and prints its evaluation ID, progress and grade
counts. It returns the reward only when every scheduled attempt has a grade, and stops on
failure, cancellation or missing grades. A recorded zero is valid. This catches basic
execution and grading errors; one test problem cannot measure generalization.

## 5. Train on one task and evaluate the checkpoint

Continue in the same Python session. Use one task group per step for the smoke-test benchmark:

```python
training = client.training.create(
    agent_name=agent_name,
    benchmark_name=bench_name,
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
final_reward = eval_smoketest(
    client, agent_name, bench_name, parent_checkpoint_id=checkpoint.checkpoint_id
)
print(f"baseline_reward={baseline_reward:.6f}")
print(f"final_reward={final_reward:.6f}")
print(f"reward_delta={final_reward - baseline_reward:+.6f}")
```

Reward may stay unchanged or fall in a short run. For an existing benchmark where you want
only evaluation or only training, use the [individual SDK calls](../../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform).

## 6. Expand to the 80-task example

After the smoke run completes, upload the example's **64 training and 16 held-out test problems**:

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
script prints and pins the baseline's `bench_id` for training and checkpoint evaluation.
For the manual smoke-test calls above, selecting by agent and benchmark names resolves the
latest version each time; finish those calls before uploading another version, or pass the
original `bench_id` instead. Training starts from the base model unless you set
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

In the platform, the agent's **Training** and **Evaluation** sections show these runs.
For direct links, use the platform's origin and the organization's URL key from an existing
platform URL; the opaque organization ID is not its URL key.

| Page | Path |
| --- | --- |
| Benchmark version | `/<org-key>/agents/<agent-id>/benchmarks/<bench-id>` |
| Training run | `/<org-key>/models/<training-run-id>` |
| Eval trajectories | `/<org-key>/evaluate/runs/<eval-run-id>/trajectories` |

## Optional: deploy and query the trained checkpoint

After training succeeds, replace `YOUR_TRAINING_RUN_ID` with the run you want to deploy.
Use its actual final step count; the commands above use three steps. Deployments require a
committed checkpoint backed by Tinker training weights. Use a new model slug for this example:
`role="production"` makes it the active target when deployment finishes.
For `role="test"`, query the exact deployment route below; the model slug resolves only to
the active production deployment.

```python
import time

from trajectory import Client

client = Client()
checkpoint = client.training.checkpoints.retrieve("YOUR_TRAINING_RUN_ID", step_index=3)
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="gsm8k-trained",
    role="production",
)
print(deployment.deployment_id)
deadline = time.monotonic() + 600
while True:
    state = client.deployments.retrieve(deployment.deployment_id)
    if state.status == "DEPLOYED":
        break
    if state.status in {"FAILED", "CANCELLED"}:
        raise RuntimeError(state.model_dump())
    if time.monotonic() >= deadline:
        raise TimeoutError(f"Deployment still {state.status}: {deployment.deployment_id}")
    time.sleep(5)

if not state.is_active:
    raise RuntimeError("Deployment is ready but is not the active target for its slug")
tid = client.trajectories.create().tid
try:
    response = client.chat.completions.create(
        model=state.model_slug,
        messages=[{"role": "user", "content": "What is 17 times 6?"}],
        x_trajectory_id=tid,
    )
except Exception as error:
    try:
        client.trajectories.complete(tid, termination_reason="ERROR")
    except Exception as cleanup_error:
        error.add_note(f"Trace completion also failed: {type(cleanup_error).__name__}")
    raise
else:
    choice = response.choices[0]
    if choice.finish_reason in ("length", "max_tokens"):
        reason = "TRUNCATION"
    elif choice.finish_reason == "stop" and choice.message.content:
        reason = "ENV_DONE"
    else:
        reason = "ERROR"
    client.trajectories.complete(tid, termination_reason=reason)
    if reason != "ENV_DONE":
        raise RuntimeError(f"Serving check failed: {choice.finish_reason=}")
    print(choice.message.content)
```

A local polling timeout leaves the deployment running. Retrieve the same deployment ID before
retrying. To stop pending deployment work, use `POST /api/v1/deploy/<deployment-id>/cancel`
and wait for `CANCELLED`; a cancellation request alone does not confirm cleanup. To remove a
served deployment and its endpoint resources, call `client.deployments.delete(deployment_id)`.

For an existing production slug, create a `role="test"` deployment first. Once `DEPLOYED`,
query `POST /api/v1/deploy/<deployment-id>/chat/completions` with its model slug, organization
key and `X-Trajectory-Id`. Promote it from the platform's
**Deployment** page when ready; promotion requires a signed-in user. Deployment provisions
the model endpoint, so no separate model-endpoint creation is needed.
For reasoning models, `max_tokens` includes reasoning tokens; a cap sized for only the visible
answer can return no answer. Check the answer and complete its trajectory from `finish_reason`
as above; a length-capped HTTP response is `TRUNCATION`, not a completed-answer check.

An OpenAI client uses the API origin followed by `/v1` for active-slug inference, or
`/api/v1/deploy/<deployment-id>` for the exact-deployment check. Pass the trajectory header
through `extra_headers`. See [inference setup](../../README.md#connect-the-actor) for model
discovery and credentials, and [production traces](../production_traces.md) for application traffic.

This query demonstrates serving; the checkpoint evaluation in step 5 uses the benchmark's
`submit_answer` tool and grader to measure reward.

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
