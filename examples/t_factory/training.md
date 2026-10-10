# T Factory: from one task to a full benchmark

This is an alternative to the [GSM8K full workflow](../gsm8k/README.md) if you want to keep
using the first lesson's toy reward.

Start with [your first task](README.md). This walkthrough uses the same prompt and grader in a
managed runtime: validate one task, evaluate and train a small benchmark, then upload the full
192-task dataset. Run commands from the cookbook repository root with `TRAJECTORY_API_KEY` set.
The platform builds runtime images remotely; you do not need local Docker.

## 1. Confirm your organization and choose an agent

```bash
uv run --with trajectory-sdk python - <<'PYCODE'
from trajectory import Client

client = Client()
print(client.organizations.retrieve_current())
for agent in client.agents.list():
    print(agent.name, agent.description)
PYCODE
```

Check the printed organization. Follow the shared [agent selection guidance](../../README.md#choose-or-create-an-agent):
use the selected agent, reuse one whose purpose fits, or create one for a new use case.
For a new T Factory project:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="t-factory-cookbook", description="Learning task rewards and training with T Factory")'
```

Use your chosen agent's name in the following commands. Reuse it on subsequent runs.

## 2. Validate one task in the managed runtime

```bash
uv run examples/t_factory/ingest.py --agent-name "t-factory-cookbook" --diagnose-only
```

This builds the runtime and runs one task on `Qwen/Qwen3.5-4B`. It prints the diagnostic ID,
status, task report, and reward. Continue when it exits successfully: both the diagnostic and
the task completed without failure, and a reward was recorded. A score of zero is valid.
Image readiness alone does not prove execution or grading succeeded.

If it fails, inspect the printed run and task failures, correct the cause, and repeat this step.
For your own harness, use the [task diagnostics recipe](../task_diagnostics.md).

## 3. Upload one training task and one test task

The default managed training flow used here requires a training split and a test split. The small benchmark contains
**one training prompt and one different, held-out test prompt**. This lets you learn evaluation
and train on one task without uploading the full dataset. It is an integration check; one test
prompt cannot establish generalization.

```bash
uv run examples/t_factory/ingest.py --agent-name "t-factory-cookbook" --small
```

The command prints `agent_name` and `bench_id`, then waits for its image to be ready. Keep that
benchmark ID and wait for successful exit. This registers `t-factory-small`; it does not start
training. The diagnostic from step 2 is a separate check, not this training benchmark.

## 4. Evaluate, train, and compare

Replace `YOUR_SMALL_BENCH_ID` with the ID from step 3:

```bash
uv run examples/t_factory/train.py --bench-id YOUR_SMALL_BENCH_ID --num-steps 3
```

The script runs these stages in order:

1. Evaluate the base model on the single held-out task.
2. Train `Qwen/Qwen3.5-4B` for three steps using the single training task.
3. Evaluate the final checkpoint on that same held-out task.
4. Print baseline reward, final reward, reward delta, and before/after model responses.

The small run uses one task group per training step. Each group can contain multiple model
samples according to the model's training defaults; one task does not mean one model call.
A short run may leave reward unchanged or make it worse.

Watch `training_run_id`, `status`, and `steps=completed/total`. An ID means the request was
accepted; `pending` with zero steps is still waiting. Increasing step counts show training
progress, and `status=succeeded` means training completed. The script stops on failure or
cancellation and only evaluates a checkpoint after training succeeds.

For a benchmark you already selected, use the cookbook's [individual SDK calls](../../README.md#3-evaluate-train-and-compare-on-the-trajectory-platform)
to run only evaluation or only training. The command above always runs both.

## 5. Scale to the full benchmark

Once the small flow works, upload all **128 training tasks and 64 held-out test tasks**:

```bash
uv run examples/t_factory/ingest.py --agent-name "t-factory-cookbook"
uv run examples/t_factory/train.py --bench-id YOUR_FULL_BENCH_ID --num-steps 3
```

Replace `YOUR_FULL_BENCH_ID` with the new uploader output. The full benchmark is named
`t-factory`; it evaluates 64 test tasks and uses four training task groups per step.
It uses the same harness and grader as the first lesson. Uploading the same name under the
same agent creates a new version and benchmark ID, so always use the latest printed ID.

To run longer, increase `--num-steps`. Inspect the [observed reward behavior](results.md) for
examples of how this toy objective can change responses, including reward hacking.

## Inspect an existing run

Keep the diagnostic, benchmark, and training IDs. Interrupting a local command does not cancel
work already accepted by the service. Inspect that work before submitting a replacement.
Replace the corresponding placeholder in these snippets and run with
`uv run --with trajectory-sdk python` (or your SDK-enabled Python environment).

For a diagnostic:

```python
from trajectory import Client

client = Client()
diagnostic_id = "YOUR_DIAGNOSTIC_ID"
print(client.diagnostics.get_status(diagnostic_id))
print(client.diagnostics.get_diagnostics(diagnostic_id).to_json())
```

For a benchmark image that failed or is still building:

```python
from trajectory import Client

client = Client()
print(client.benchmarks.images.list("YOUR_BENCH_ID").to_json())
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

Open the agent's training page in the platform to inspect its runs and trajectories.
To choose another model, inspect `client.training.list_options(bench_id="YOUR_BENCH_ID")`
and pass a supported model with `--model`; its advertised options must support the script's settings.

## Optional: deploy and query

Resolve the final checkpoint and create a production Model Endpoint deployment. Use the actual
training run ID and final step count (the commands above use three steps):

```python
from trajectory import Client

client = Client()
checkpoint = client.training.checkpoints.retrieve(
    "tr_<training-run-id>",
    step_index=3,
)
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="t-factory-trained",
    role="production",
)
print(f"deployment_id={deployment.deployment_id}")
```

Creating a deployment restores the training checkpoint, starts its Model Endpoint, and
activates the production model slug before returning.

### Query the deployed model

Use the deployment's model slug with the SDK's OpenAI-compatible chat API:

```python
response = client.chat.completions.create(
    model="t-factory-trained",
    max_tokens=4096,
    messages=[
        {
            "role": "user",
            "content": "Describe playing music in a friendly way.",
        }
    ],
)
print(response.choices[0].message.content)
```

## Next: your benchmark

- [GSM8K](../gsm8k/README.md): load a real dataset, keep private answers separate, and grade tool submissions.
- [Number guessing](../number_guessing/README.md): introduce multiple turns and context compaction.
- [Harvey](../harvey_labs.md) and [Inspect](../inspect.md): preserve an existing harness's solver, tools, and grader.

Use the [runtime packaging reference](../../README.md#package-a-benchmark-runtime) when adapting your
own benchmark. Benchmark-specific source inspection may be necessary to understand its inputs,
tools, and grader; these recipes document the SDK integration.
