# Harvey LAB

This example adapts an existing multi-turn benchmark without replacing its agent loop, tools,
grader, or Podman sandbox. It runs the public
[Harvey LAB repository](https://github.com/harveyai/harvey-labs) at commit
`1dd81403b2fbb60596f7aea3fcecafad7bf73143` with a small Trajectory overlay.

Use this pattern when a benchmark already owns its task lifecycle and evaluation logic. The SDK
adapter only needs to connect that lifecycle to a Trajectory session:

1. Create one trajectory when a task starts.
2. Forward the solving agent's model requests with the same trajectory ID.
3. Run the benchmark's original judge after the agent loop.
4. Log the resulting reward and complete the trajectory.

## Files

- [`ingest.py`](ingest.py): assign splits and upload the benchmark.
- [`tasks.txt`](tasks.txt): the 2,010 task directories at the pinned commit.
- [`runtime/Dockerfile`](runtime/Dockerfile): download the pinned source and copy the overlay
  over it.
- [`runtime/lab_core/`](runtime/lab_core/): the overlay. It adds the
  [Trajectory adapter](runtime/lab_core/harness/adapters/trajectory.py) and changes the harness,
  judge, and sandbox files it hooks into.
  [`runtime/NOTICE`](runtime/NOTICE) carries Harvey LAB's MIT license.
- [`tests/`](tests/): unit tests for the adapter, judge routing and deliverable matching, the
  sandbox's cgroups switch, and the upload's judge settings.

## Run it

[Choose or create an agent](https://docs.trajectory.ai/agent). Harvey LAB grades each run with
two judges, Claude Sonnet 4.6 and GPT-5.5, and averages their scores. For that default pair,
register the organization secrets `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`. A single-judge
override needs only its provider's key. Secret values never enter the image or upload.

```python
from trajectory import Client

client = Client()
client.secrets.create(name="ANTHROPIC_API_KEY", value="YOUR_ANTHROPIC_API_KEY")
client.secrets.create(name="OPENAI_API_KEY", value="YOUR_OPENAI_API_KEY")
```

From the repository root, upload the benchmark:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME
```

`--judges` takes one or two models, as in Harvey LAB's own evaluation CLI. Claude models use
`ANTHROPIC_API_KEY`, Gemini models `GEMINI_API_KEY`, Mistral models `MISTRAL_API_KEY`, and the others `OPENAI_API_KEY`. For an OpenRouter judge, store its key as `OPENAI_API_KEY` and pass the
provider's `openai/` model ID; the uploader selects OpenRouter's endpoint:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME \
  --judges openai/gpt-6-luna
```

Use `--judge-base-url` to select another OpenAI-compatible endpoint.
One task uses one OpenAI-compatible endpoint, so do not mix an `openai/` OpenRouter ID
with a direct OpenAI judge ID in the same pair; the uploader rejects that combination.

The script prints the benchmark ID and waits for the runtime image, which downloads the
3 GB source archive. Image readiness alone doesn't verify model inference, grading, or reward
logging, so run a [benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark)
before an evaluation or training run, with the model and settings you plan to evaluate with.

Harvey LAB tasks are long: an attempt can take dozens of tool calls, and a reasoning model's
reply can run to thousands of tokens. These settings ran on Qwen 3.6 35B A3B, with a
three-step smoke training run. They leave each reply the model's full context and give each
attempt two hours:

```python
from trajectory import Client

client = Client()
bench_id = "YOUR_BENCH_ID"
model = "Qwen/Qwen3.6-35B-A3B"
limits = {
    "reasoning_effort": "on",
    "max_output_tokens_per_step": 262142,
    "max_total_tokens_per_trajectory": 262144,
    "execution_timeout_seconds": 7200,
}
evaluation = {**limits, "max_turns_per_trajectory": 200}

diagnostic = client.diagnostics.start_benchmark(
    bench_id=bench_id, base_model_slug=model, options=evaluation
)
print(diagnostic.benchmark_diagnostic_id)
```

The unit tests run in the runtime image:

```bash
docker build -t harvey-lab-trajectory examples/harvey_labs/runtime
docker run --rm -v "$PWD/examples/harvey_labs:/example:ro" harvey-lab-trajectory \
  sh -c 'pip install -q pytest && cd /tmp && python -m pytest -q -p no:cacheprovider /example/tests'
```

## Evaluate and train

Evaluate with the diagnostic's settings, then train with the same limits:

```python
baseline = client.evals.create(
    bench_id=bench_id, base_model_slug=model, options=evaluation
)
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **limits,
        "max_turns_per_trajectory": 125,
        "num_steps": 3,
        "train_batch_size": 1,
        "group_size": 8,
        "max_active_rollouts": 8,
        "evaluation_enabled": False,
    },
)
```

After training succeeds, evaluate its final checkpoint with the baseline's options, since a
checkpoint evaluation otherwise inherits the training settings:

```python
checkpoint = max(
    client.training.checkpoints.list(training.training_run_id),
    key=lambda c: c.step_index,
)
final = client.evals.create(
    bench_id=bench_id,
    base_model_slug=checkpoint.base_model_slug,
    parent_checkpoint_id=checkpoint.checkpoint_id,
    options=evaluation,
)
```

The reward is the fraction of rubric criteria passed, averaged over the two judges. Harvey LAB's
all-pass score rarely differs across a group of attempts at one task, so it gives training little
signal; it is logged as the zero-weight `all_pass` component, shown with each trajectory's reward. Check the step
summaries' `no_signal_groups` early (see the [training guide](https://docs.trajectory.ai/guides/training)).
An attempt that prints the whole document store in one command can overflow the context and
end as `TRUNCATION`; training skips those attempts by default.

In a Qwen 3.6 35B A3B smoke run with the settings above, training
`trn_06ae6450675677bf8000db5489ad3917` completed all three steps. The baseline
`evr_06ae63faa3c57aed800058760a953f7a` graded six of eight tasks; two attempts ended
`TRUNCATION` without a reward. Checkpoint evaluation
`evr_06ae655b8a36745d8000313d9a132baf` graded all eight tasks, with mean reward 0.61.
The diagnostic on the current overlay, `bmd_06ae673f68bd71b98000f63c85b243e0`,
graded all ten tasks with `ENV_DONE` endings. The baseline and checkpoint means are not
directly comparable because their graded task counts differ.

## Ingest the existing task tree

[`tasks.txt`](tasks.txt) lists every upstream directory with a `task.json`, and the uploader
assigns a deterministic held-out split. Hashing the task path keeps the same 120 test tasks
across repeated ingestions:

```python
ordered = sorted(
    tasks,
    key=lambda task: hashlib.sha256(f"harvey-test-v1:{task}".encode()).digest(),
)
test_tasks = set(ordered[:120])
```

Each task runs the original harness through the overlay's Trajectory adapter, which takes its
judges from `--judges`. Secrets come from the platform, not the benchmark source. In managed
runs, the run sets sampling and per-step output tokens, and its `reasoning_effort` takes
precedence over `--reasoning-effort`:

```python
from trajectory import SecretRef, TaskSpec
from trajectory.types.benchmarks.task_spec import EnvResources

TaskSpec(
    name=task,
    split="test" if task in test_tasks else "train",
    run_command=(
        "python -m lab_core.harness.run --model trajectory/session "
        f"--task {task} --run-id trajectory --max-turns 200 "
        "--reasoning-effort low --judges claude-sonnet-4-6 gpt-5.5"
    ),
    env_vars={
        "ANTHROPIC_API_KEY": SecretRef(secret_ref="ANTHROPIC_API_KEY"),
        "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
        "HARVEY_PODMAN_DISABLE_CGROUPS": "1",
    },
    env_resources=EnvResources(network_mode="public", docker_engine=True),
)
```

The public benchmark contains 2,010 tasks: 1,890 training tasks and 120 held-out test tasks.

## Preserve the original harness

The added model adapter creates one TID and sends the harness's existing Responses API calls
through `trajectory-session`. After the run, it grades with Harvey LAB's own
`evaluate_run_dual`, or `evaluate_run` for a single judge, records the fraction of criteria
passed as `criteria_fraction` and the all-pass rate as the zero-weight `all_pass`, and completes
the same trajectory. If the agent loop or the judge fails, it completes the trajectory with
`ERROR` and logs no reward. This keeps prompts, tools, output files, and grading behavior owned
by the upstream benchmark.

The [runtime image](runtime/Dockerfile) follows the same overlay approach: download a pinned
public source revision, then copy only the SDK adapter and the narrow harness changes into the
image. Pinning the source prevents upstream changes from silently changing an ingested
benchmark. The overlay's Python files keep upstream's formatting, so each one diffs cleanly
against the pinned commit.

## Run Podman inside the benchmark runtime

Harvey LAB already isolates document tools in Podman. The outer benchmark runtime therefore
requests a container engine with `docker_engine=True`, and the original harness launches its
inner Podman container as usual.
The flag starts `dockerd`, so the runtime image must include `dockerd` and the Docker CLI.

Some nested environments use `crun` without delegated cgroup controllers. When
`HARVEY_PODMAN_DISABLE_CGROUPS=1`, the overlay's sandbox change checks `podman info` for `crun`
and adds `--cgroups=disabled`, skipping the CPU, memory, and process limits that need cgroups.
Local runs leave the variable unset and keep their default cgroup behavior.
