# Harvey LAB

This example runs the public
[Harvey LAB repository](https://github.com/harveyai/harvey-labs) at commit
`1dd81403b2fbb60596f7aea3fcecafad7bf73143` with a Trajectory overlay. The
upstream agent loop, tools, judges and Podman sandbox remain in place. Each task uses one
Trajectory session for model calls and its final grade.

## Files

- [`ingest.py`](ingest.py): upload all 2,010 upstream tasks, with a stable split of
  1,890 training and 120 held-out test tasks.
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
override needs only its provider's key. Export your organization API key first:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

```python
from trajectory import Client

client = Client()
client.secrets.create(name="ANTHROPIC_API_KEY", value="YOUR_ANTHROPIC_API_KEY")
client.secrets.create(name="OPENAI_API_KEY", value="YOUR_OPENAI_API_KEY")
```

From the repository root, upload the benchmark:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME
```

For an override, use this upload command instead of the default. Register
`OPENAI_API_KEY` with your OpenRouter credential; the uploader selects OpenRouter's
endpoint for the `openai/` model ID:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME \
  --judges openai/gpt-6-luna
```

`--judges` takes one or two models, as in Harvey LAB's own evaluation CLI. Claude models use
`ANTHROPIC_API_KEY`, Gemini models `GEMINI_API_KEY`, and Mistral models `MISTRAL_API_KEY`.
GPT and O-series models use `OPENAI_API_KEY`; other IDs require `--judge-base-url` and use
that key too.

Use `--judge-base-url` to select another OpenAI-compatible endpoint for OpenAI-client judges.
Claude, Gemini, and Mistral IDs still use their native providers and keys when it is set.
One task uses one OpenAI-compatible endpoint, so do not mix an `openai/` OpenRouter ID
with a direct OpenAI judge ID in the same pair; the uploader rejects that combination.

The script prints the benchmark ID and waits for the runtime image, which downloads the
3 GB source archive. Image readiness alone doesn't verify model inference, grading, or reward
logging, so run a [benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark)
before an evaluation or training run, with the model and settings you plan to evaluate with.

Harvey LAB tasks are long: an attempt can take dozens of tool calls, and a reasoning model's
reply can run to thousands of tokens. These settings ran on Qwen 3.6 35B A3B, with a
three-step smoke training run. They leave each reply the model's full context and give each
attempt two hours. The diagnostic chooses its own sample; the eight-task evaluation sample
keeps the first run short. Remove `evaluation_max_samples` to evaluate all 120 held-out tasks:

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
diagnostic_options = {**limits, "max_turns_per_trajectory": 200}
evaluation = {**diagnostic_options, "evaluation_max_samples": 8}

diagnostic = client.diagnostics.start_benchmark(
    bench_id=bench_id, base_model_slug=model, options=diagnostic_options
)
print(diagnostic.benchmark_diagnostic_id)
```

## Evaluate and train

Evaluate with the diagnostic's model and runtime limits on eight held-out tasks, then train
with the same limits:

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

`criteria_fraction` is the fraction of rubric criteria passed, averaged over the two
judges. `all_pass` is logged with weight zero because it gives training little signal;
retrieve both with `include_reward_components=True`. Check `no_signal_groups` in the
[training guide](https://docs.trajectory.ai/guides/training). A context overflow ends an
attempt as `TRUNCATION`; training skips those attempts by default.

In a Qwen 3.6 35B A3B smoke run with the OpenRouter `openai/gpt-6-luna` judge
override, diagnostic `bmd_06ae6a05c31374648000e5553ab91809` graded all ten tasks with
`ENV_DONE` endings. Baseline `evr_06ae6b1b47dc71ad8000ef30b2fcf7cf` graded seven of eight
tasks, training `trn_06ae6b96664177f58000a38ed4c19a9e` succeeded for all three steps,
and checkpoint evaluation `evr_06ae6c84bb6272b5800076c31dc85487` graded seven of eight.
Both evaluations used the same task IDs and options. Each had one ungraded `TRUNCATION` on a
different task; the platform run means, 0.647 and 0.645, count these as zeros. Eight tasks
with missing grades cannot establish training lift.

The runtime downloads the pinned upstream source and applies the overlay. It requests
`docker_engine=True` so Harvey LAB can run its document tools inside Podman. When a
nested `crun` runtime lacks delegated cgroups, `HARVEY_PODMAN_DISABLE_CGROUPS=1`
disables those inner cgroup limits; local runs keep upstream's defaults.

## Test the overlay

From the cookbook root, run the tests in the runtime image:

```bash
docker build -t harvey-lab-trajectory examples/harvey_labs/runtime
docker run --rm -v "$PWD/examples/harvey_labs:/example:ro" harvey-lab-trajectory \
  sh -c 'pip install -q pytest && cd /tmp && python -m pytest -q -p no:cacheprovider /example/tests'
```
