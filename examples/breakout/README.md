# Breakout: write the policy weights

Ask a model to write the weights of a small neural network that plays **MinAtar
Breakout**, then use the game's score as an RL reward. The model gets one reply and
no tools. The harness parses numeric assignments, constructs the network, and plays
the games on a CPU. No GPU sandbox or model-generated code execution is needed.

This is a starter example of [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/),
adapted from Experiment 5 by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab.
RL updates the language model that writes the weights; each submitted game-playing
network is evaluated without further training.

## The task and score

Every task sends the exact same [prompt](runtime/prompt.txt): the game rules and
source, observation layout, network dimensions (`600 → 32 → 32 → 6`, tanh), and
numeric assignment format. Only the private episode seeds change.

| Split | Task records | Episodes per generated policy | Seeds |
| --- | ---: | ---: | --- |
| Train | 32 | 64 | Disjoint bundles covering 1000–3047 |
| Test | 1 | 256 | 100000–100255 |

One task record produces **one policy**, which plays all its episode seeds. The
mean game score is that policy's reward. Seeds are stored in
[tasks.jsonl](runtime/tasks.jsonl); they are never sent to the model. There are no
actor tools that can read the task file. Held-out seeds measure performance on the
same game, not generalization to different games.

The scorer preserves MinAtar 1.0.15, 0.1 sticky-action probability, stochastic
policy actions, and a 10,000-step episode cap. Each episode has separate seeded
game and policy RNGs. Replaying the same weights and seeds gives the same score.

The harness logs the **raw mean game score** as `breakout_score`, with bounds
`[0, 10000]`. Trajectory therefore displays an aggregate reward of
`mean_game_score / 10000`. Invalid replies and token-truncated replies score zero;
truncations are marked `TRUNCATION`. Infrastructure failures are marked `ERROR`
without a fabricated game reward. As in the source parser, unsupported individual
assignments are skipped; a reply with no readable assignments is invalid.

## Setup

Use `trajectory-sdk>=0.10.1`. From the repository root, set your organization's
`TRAJECTORY_API_KEY` and start a Python session:

```bash
uv run --with 'trajectory-sdk>=0.10.1' python
```

Run the following snippets in that session. Reuse an existing agent, or create one
for this example:

```python
from trajectory import Client
from trajectory.lib import push, start_task_diagnostic
from examples.breakout.upload import RUNTIME_ROOT, build_spec

client = Client()
organization = client.organizations.retrieve_current()
print(organization["organization_id"])
agent = client.agents.create(name="breakout-starter")
spec = build_spec()

model = "Qwen/Qwen3.6-35B-A3B"
options = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_output_tokens_per_turn": 32768,
    "max_output_tokens_per_trajectory": 32768,
    "max_turns_per_trajectory": 1,
    "trajectory_timeout_seconds": 7200,
}
```

Managed runs apply these sampling and token settings to the harness's model call.
The harness allows 90 minutes for that HTTP request, with the SDK's default retries;
the outer two-hour trajectory limit also covers scoring. Model inference is paid;
the short training example below is not a dollar cap.

## Diagnose one task

Build the image and run the first training seed bundle before uploading the spec:

```python
import time

task = spec.tasks[0].model_copy(update={"runtime": spec.runtime})
diagnostic = start_task_diagnostic(
    client,
    task,
    agent_name=agent.name,
    root=RUNTIME_ROOT,
    base_model_slug=model,
    options=options,
)
diagnostic_id = diagnostic.benchmark_diagnostic_id
while client.diagnostics.get_status(diagnostic_id).status not in {
    "completed", "failed", "cancelled"
}:
    time.sleep(5)
report = client.diagnostics.get_diagnostics(diagnostic_id)
print(report.status, report.failure)
for task_result in report.tasks:
    print(task_result.termination_reason, task_result.reward, task_result.failure)
```

Check the ending and reward before continuing. A normally completed, zero-scoring
policy is a valid result; an execution error is not.

## Upload and evaluate

```python
uploaded = push(
    client, spec, agent_name=agent.name, root=RUNTIME_ROOT, build_images=True
)
bench_id = uploaded.bench_id
print(bench_id, uploaded.task_count)  # 33 task records

# Inspect model availability and supported options in your organization.
print(client.training.list_options(bench_id=bench_id, base_model_slug=model))

evaluation = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={**options, "val_trajectories_per_task": 1},
)
platform = f"https://platform.trajectory.ai/{organization['organization_key']}"
print(f"{platform}/evaluate/runs/{evaluation.eval_run_id}/trajectories")
```

The evaluation uses the single test task: one generated policy over 256 games.
To upload from the command line to an existing agent instead:

```bash
uv run --with 'trajectory-sdk>=0.10.1' python examples/breakout/upload.py --agent-name breakout-starter
```

Wait for evaluation to finish, then read the raw game score:

```python
while client.evals.runs.retrieve(evaluation.eval_run_id).status not in {
    "completed", "failed", "cancelled"
}:
    time.sleep(5)
result = client.evals.runs.retrieve(evaluation.eval_run_id)
print(result.status, result.reward_mean)
for item in client.trajectories.list(eval_run_id=evaluation.eval_run_id):
    detail = client.trajectories.retrieve(
        item.trajectory_id, include_steps=False, include_reward_components=True
    )
    print(detail.termination_reason, detail.reward_components)
```

Read `breakout_score` from the reward components for raw points. Its mean is over
game seeds, not independent model replies. Increase `val_trajectories_per_task`
when comparing several independently generated policies.

## Train for three updates

After the diagnostic and evaluation finish with recorded grades, start a small
run on the training seed bundles. Each update uses one group of four model replies;
each reply is evaluated over the same 64 seeds within that group.

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **options,
        "num_steps": 3,
        "batch_size": 1,
        "group_size": 4,
        "max_concurrent_trajectories": 8,
        "val_num_tasks": 1,
        "val_trajectories_per_task": 1,
        "val_max_concurrent_trajectories": 1,
    },
)
print(f"{platform}/models/{training.training_run_id}")
```

All other settings use the selected model's defaults. Group filtering and async
collection can generate more than 12 training replies. This short run demonstrates
the workflow; it does not promise a measurable improvement.

```python
progress = client.training.runs.progress(training.training_run_id)
print(progress.status, progress.completed_steps, progress.total_steps)
for point in client.training.rewards.list_history(training.training_run_id):
    print(point.step, point.avg_reward)
# To stop: client.training.runs.cancel(training.training_run_id)
```

Use the same benchmark version and evaluation options to
[compare a trained checkpoint](https://docs.trajectory.ai/guides/evaluations#3-score-a-checkpoint)
with the base model.

## Local checks and provenance

Install the runtime dependencies and run the offline checks:

```bash
uv venv .venv
uv pip install 'trajectory-sdk>=0.10.1' 'numpy==2.4.3'
uv pip install --no-deps 'MinAtar==1.0.15'
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s examples/breakout/tests -v
```

The tests cover numeric parsing without code execution, deterministic replay,
observation layout, disjoint seeds, the SDK reward contract, and normal, invalid,
truncated, and infrastructure-failure endings. MinAtar's visualization dependencies
are unnecessary for CPU scoring and are omitted from the runtime image.

The prompt is adapted from
[Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch)
(CC BY 4.0), and embeds MinAtar source (GPL-3.0). See
[ATTRIBUTION.txt](runtime/ATTRIBUTION.txt) and [LICENSE-MinAtar](runtime/LICENSE-MinAtar).
This adaptation uses fixed train/test seed bundles and 64/256 evaluation episodes
instead of the original experiment's 1,000 episodes per submission, and adds
bounded numeric parsing. No reference solution or trained weights are packaged.
