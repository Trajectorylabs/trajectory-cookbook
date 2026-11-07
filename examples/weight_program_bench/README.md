# WeightProgramBench

Train a language model to write the weights of small game-playing neural networks.
Each task supplies the game's rules, observation layout, source code, and network
format. The model answers once, without tools. A CPU grader parses numeric tensor
assignments and evaluates the resulting policy; it never executes model-written code
or optimizes the submitted network.

This packages the expanded weight-space programming experiment: **317 training games
and five held-out MinAtar games**. RL updates the language model between submissions.
The game-playing networks themselves receive no training.

## Tasks and origins

| Split | Games | Origin |
| --- | ---: | --- |
| Train | 51 | Atari-style synthetic conversions in Trajectory's `minatar-synth` |
| Train | 266 | Crisp-inspired synthetic conversions in the same source snapshot |
| Test | 5 | Original MinAtar: Asterix, Breakout, Freeway, Seaquest, Space Invaders |

The training games are independent Python conversions with `10 × 10 × 6` binary
observations. They are **not Atari ROMs, ALE environments, or the native Crisp games**.
The held-out games use MinAtar 1.0.15 with their channels mapped into the same six-channel
input. Held-out game names are absent from training; related mechanics can still occur
in training games. This split does not establish that every test mechanic is novel.

The task is inspired by [weight-space programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab, especially
[Experiment 5](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch).
The expanded training environments and task context come from
[Trajectory's minatar-synth](https://github.com/Trajectorylabs/minatar-synth/tree/4038830175c4201ca10e004ed4441985f2acddf2).
Crisp game designs are credited to ABA Games and contributors: 136 conversions cite
[crisp-game-lib-games](https://github.com/abagames/crisp-game-lib-games/tree/cfb39d2f988feb5918eb83697145b5a35adabf58),
eight cite [crisp-game-lib-11-games](https://github.com/abagames/crisp-game-lib-11-games/tree/6fea4eefe2deaa42706051d96a04ec9e36104696),
and 122 cite the [portable C/SDL port](https://github.com/joyrider3774/crisp-game-lib-portable-sdl/tree/04bc2c4cfc2853628eda25a46190bb7ac6fe216a)
by Joyrider3774 and contributors. The five test games come from
[MinAtar](https://github.com/kenjyoung/MinAtar) by Kenny Young and Tian Tian.

See [TASK_ORIGINS.json](TASK_ORIGINS.json) for each game's family and source link,
and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution and license scopes.
The synthetic source snapshot has no declared upstream license; its redistribution
terms must be settled before merging this example. The example is self-contained:
running it does not require access to the private upstream repository.

## Network and reward

Every task uses the same `600 → 32 → 32 → 6` tanh network: **20,486 parameters**.
Weights start at zero; the reply supplies numeric assignments in an `edits` block.
The scorer samples actions from the network's softmax output.

Each generated policy plays **1,000 episodes**, with environment seeds `1729–2728`
and independent action RNG seeds `7919–8918`. Sticky-action probability is `0.1`;
episodes stop at termination or 10,000 actions. Original MinAtar difficulty ramping
is disabled. Episode seeds are fixed across tasks; game identity defines the split.

The reward is `clip(mean_game_score, 0, game_score_cap) / game_score_cap`, logged as
`reward_normalized_minatar_score`. The original test-game caps are 1,000, except
Freeway's cap of 100; each synthetic game declares its own cap. These are fixed
scales, not estimates of equal difficulty across games.

Invalid submissions receive zero. As in the expanded experiment, a response cut off
by its token limit is still graded and marked `TRUNCATION`; the training run's
`train_on_truncation` option controls eligibility. Infrastructure failures are `ERROR`,
not fabricated game scores. One generated policy's 1,000 episodes are not 1,000
independent language-model samples.

## Upload and evaluate

Install `trajectory-sdk>=0.10.1`, configure `TRAJECTORY_API_KEY`, and run the following
from the repository root. Reuse an agent in your organization, or create one for this
project. Model inference is paid.

```python
from trajectory import Client
from trajectory.lib import push
from examples.weight_program_bench.upload import RUNTIME_ROOT, build_spec

client = Client()
organization = client.organizations.retrieve_current()
agent = client.agents.create(name="weight-program-bench")
spec = build_spec()
uploaded = push(client, spec, agent_name=agent.name, root=RUNTIME_ROOT, build_images=True)
bench_id = uploaded.bench_id
print(bench_id, uploaded.task_count)  # 322

model = "Qwen/Qwen3.6-35B-A3B"
print(client.training.list_options(bench_id=bench_id, base_model_slug=model))
options = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_output_tokens_per_turn": 32768,
    "max_output_tokens_per_trajectory": 32768,
    "max_turns_per_trajectory": 1,
    "trajectory_timeout_seconds": 7200,
}
evaluation = client.evals.create(
    bench_id=bench_id, base_model_slug=model,
    options={**options, "val_num_tasks": 5, "val_trajectories_per_task": 1},
)
platform = f"https://platform.trajectory.ai/{organization['organization_key']}"
print(f"{platform}/evaluate/runs/{evaluation.eval_run_id}/trajectories")
```

Managed runs apply their generation options to the harness's model call. The harness
preserves the source's 32,768-token request and SDK `max_retries=20`; transport retries
are distinct from independent evaluation samples. No training starts during upload.
For an existing agent, the upload-only command is:

```bash
uv run --with 'trajectory-sdk>=0.10.1' python examples/weight_program_bench/upload.py --agent-name weight-program-bench
```

Wait for the evaluation, then inspect each task's ending and grade:

```python
import time

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
    print(item.task_id, detail.termination_reason, detail.reward_components)
```

This produces one independent policy for each of the five test games. Increase
`val_trajectories_per_task` for repeated generations, and compare per-game scores
alongside the mean. A single best checkpoint or sample is not an unbiased estimate
of the model's expected performance.

## Try three training updates

After a managed evaluation completes with valid recorded grades, a small SDK run can
exercise the training path. This example uses one game group of four responses per
update; asynchronous collection and filtering can create additional attempts.

```python
training = client.training.create(
    bench_id=bench_id, base_model_slug=model,
    options={
        **options,
        "num_steps": 3,
        "batch_size": 1,
        "group_size": 4,
        "max_concurrent_trajectories": 8,
        "val_num_tasks": 5,
        "val_trajectories_per_task": 1,
    },
)
print(f"{platform}/models/{training.training_run_id}")
progress = client.training.runs.progress(training.training_run_id)
print(progress.status, progress.completed_steps, progress.total_steps)
# To stop: client.training.runs.cancel(training.training_run_id)
```

The short run demonstrates SDK usage; it is not the 100-update research experiment
and is not a dollar cap. Other settings use the selected model's platform defaults.

## Local validation

```bash
uv venv .venv
uv pip install 'trajectory-sdk==0.10.1' 'numpy==2.3.4'
uv pip install --no-deps 'MinAtar==1.0.15'
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s examples/weight_program_bench/tests -v
```

The CPU-only image contains the frozen task prompts and game sources. No reference
solution, trained weights, internal organization IDs, or runtime secrets are included.
