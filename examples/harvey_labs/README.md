# Harvey LAB

For coding-agent setup, [install the SDK skills](../../README.md#use-with-a-coding-agent).

## Dataset

This example uses the public [Harvey LAB source](https://github.com/harveyai/harvey-labs)
at commit `1dd81403b2fbb60596f7aea3fcecafad7bf73143`. It uploads all 2,010
tasks in [`tasks.txt`](tasks.txt), with a stable split of 1,890 training and 120
held-out test tasks.

## Harness

The source agent loop, tools, judges, and Podman sandbox remain in place. The overlay
connects one Trajectory session to each task: it forwards the agent's model calls,
runs the original rubric judge, logs the grade, and completes the trajectory.
`criteria_fraction` is the training reward; `all_pass` is recorded with weight zero
for inspection.

Harvey LAB normally grades with Claude Sonnet 4.6 and GPT-5.5 and averages their
scores. Ambiguous deliverable filenames use the selected rubric judge instead of a
separate Claude call; a failed match raises a grading error rather than silently
counting a deliverable as missing.

## Ingestion

[Choose or create an agent](https://docs.trajectory.ai/agent). For the default judge
pair, register organization secrets `ANTHROPIC_API_KEY` and `OPENAI_API_KEY`.
A single-judge override needs only that provider's key. From the cookbook root:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
```

```python
from trajectory import Client

client = Client()
client.secrets.create(name="ANTHROPIC_API_KEY", value="YOUR_ANTHROPIC_API_KEY")
client.secrets.create(name="OPENAI_API_KEY", value="YOUR_OPENAI_API_KEY")
```

Then upload the benchmark:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME
```

The uploader prints the benchmark ID and waits for the runtime image, which downloads
the 3 GB pinned source archive. Image readiness does not establish that inference or
grading works, so run a [benchmark diagnostic](https://docs.trajectory.ai/guides/spec#upload-and-diagnose-the-full-spec)
before evaluation or training.

For a single-judge OpenRouter override, register `OPENAI_API_KEY` with that credential
and use this command instead:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME \
  --judges openai/gpt-6-luna
```

`--judges` accepts one or two models. Claude, Gemini, and Mistral IDs use their
native providers; GPT and O-series IDs use `OPENAI_API_KEY`. See
`uv run examples/harvey_labs/ingest.py --help` for endpoint overrides.

## Evaluation

Harvey LAB tasks can take dozens of tool calls. These Qwen 3.6 35B A3B limits were
used for the diagnostic, an eight-task baseline, and the checkpoint evaluation.
Remove `val_num_tasks` to evaluate all 120 held-out tasks.

```python
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

client = Client()
links = PlatformLinks.from_client(client)
bench_id = "YOUR_BENCH_ID"
model = "Qwen/Qwen3.6-35B-A3B"
limits = {
    "reasoning_effort": "on",
    "max_output_tokens_per_turn": 262142,
    "max_output_tokens_per_trajectory": 262144,
    "trajectory_timeout_seconds": 7200,
}
diagnostic_options = {**limits, "max_turns_per_trajectory": 200}
evaluation_options = {**diagnostic_options, "val_num_tasks": 8}

diagnostic = client.diagnostics.start_benchmark(
    bench_id=bench_id, base_model_slug=model, options=diagnostic_options
)
print(diagnostic.benchmark_diagnostic_id)
```

After the diagnostic completes with grades, continue in the same Python session:

```python
baseline = client.evals.create(
    bench_id=bench_id, base_model_slug=model, options=evaluation_options
)
print("Evaluation:", baseline.eval_run_id, links.evaluation(baseline.eval_run_id))
```

The source's `criteria_fraction` and `all_pass` components are available with
`include_reward_components=True`. A context overflow ends an attempt as
`TRUNCATION`; training skips those attempts by default.

## Training

Continue in the same Python session after the diagnostic and baseline complete:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={
        **limits,
        "max_turns_per_trajectory": 125,
        "num_steps": 3,
        "batch_size": 1,
        "group_size": 8,
        "max_concurrent_trajectories": 8,
        "val_enabled": False,
    },
)
print("Training:", training.training_run_id, links.training(training.training_run_id))
```

After training succeeds, select the highest-step checkpoint and evaluate it with
the same held-out tasks and options as the baseline:

```python
checkpoint = max(
    client.training.checkpoints.list(training.training_run_id),
    key=lambda c: c.step_index,
)
final = client.evals.create(
    bench_id=bench_id,
    base_model_slug=checkpoint.base_model_slug,
    parent_checkpoint_id=checkpoint.checkpoint_id,
    options=evaluation_options,
)
print("Evaluation:", final.eval_run_id, links.evaluation(final.eval_run_id))
```

A Qwen 3.6 35B A3B smoke run using the OpenRouter `openai/gpt-6-luna` judge
override graded all ten diagnostic tasks (`bmd_06ae6a05c31374648000e5553ab91809`).
The baseline (`evr_06ae6b1b47dc71ad8000ef30b2fcf7cf`) and checkpoint evaluation
(`evr_06ae6c84bb6272b5800076c31dc85487`) each graded seven of eight tasks;
training `trn_06ae6b96664177f58000a38ed4c19a9e` succeeded for all three steps.
The evaluations used the same task IDs and options. Their platform means, 0.647 and
0.645, count one different ungraded `TRUNCATION` each as zero. This sample does
not establish training lift.

## Appendix

The runtime [Dockerfile](runtime/Dockerfile) downloads the pinned source and copies
the narrow [overlay](runtime/lab_core/) into it. It requests `docker_engine=True`
for Harvey LAB's Podman document tools. When nested `crun` lacks delegated cgroups,
`HARVEY_PODMAN_DISABLE_CGROUPS=1` disables those inner limits; local runs retain
the source default. [`runtime/NOTICE`](runtime/NOTICE) carries the source's MIT license.

Run the overlay tests from the cookbook root:

```bash
docker build -t harvey-lab-trajectory examples/harvey_labs/runtime
docker run --rm -v "$PWD/examples/harvey_labs:/example:ro" harvey-lab-trajectory \
  sh -c 'pip install -q pytest && cd /tmp && python -m pytest -q -p no:cacheprovider /example/tests'
```
