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

- [`ingest.py`](ingest.py): assign splits and upload the benchmark, optionally from your own
  checkout.
- [`tasks.txt`](tasks.txt): the 2,010 task directories at the pinned commit.
- [`runtime/Dockerfile`](runtime/Dockerfile): download the pinned source and copy the overlay
  over it.
- [`runtime/lab_core/`](runtime/lab_core/): the overlay. It adds the
  [Trajectory adapter](runtime/lab_core/harness/adapters/trajectory.py) and failure diagnostics,
  and changes the harness, judge, and sandbox files they hook into.
  [`runtime/NOTICE`](runtime/NOTICE) carries Harvey LAB's MIT license.
- [`tests/`](tests/): unit tests for the adapter, judge routing and deliverable matching, the
  sandbox's cgroups switch, and checkout staging.

## Run it

[Choose or create an agent](https://docs.trajectory.ai/agent). Then register the organization
secret `OPENAI_API_KEY` for the `gpt-5.4-mini` judge. Its value never enters the image or the
upload.

```python
from trajectory import Client

Client().secrets.create(name="OPENAI_API_KEY", value="YOUR_OPENAI_API_KEY")
```

From the repository root, upload the benchmark:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME
```

The judge defaults to Harvey's `gpt-5.4-mini` on OpenAI. For another OpenAI-compatible provider,
store its key as `OPENAI_API_KEY` and pass the provider's model ID and endpoint. The endpoint
becomes each task's `OPENAI_BASE_URL`, and the judge then sends the ID to that endpoint unchanged:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME \
  --judge-model openai/gpt-6-luna --judge-base-url https://openrouter.ai/api/v1
```

To package a checkout of harveyai/harvey-labs at the pinned commit, including your edits to
its `lab_core/`, run the script from that checkout:

```bash
uv run PATH_TO_COOKBOOK/examples/harvey_labs/ingest.py --upstream . --agent-name YOUR_AGENT_NAME
```

The script stages the checkout's `lab_core/` with the overlay copied over it, and stops if the
checkout is at another commit. The image still downloads `tasks/` from the pinned commit: its
63,000 files exceed the 4,096-file build-context limit.

The script prints the benchmark ID and waits for the runtime image, which downloads the
3 GB source archive. Image readiness alone doesn't verify model inference, grading, or reward
logging, so run a [benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark)
before an evaluation or training run. Pass the model you plan to evaluate; the platform's default
diagnostic model checks the pipeline but its Harvey scores say little:

```python
from trajectory import Client

diagnostic = Client().diagnostics.start_benchmark(
    bench_id="YOUR_BENCH_ID", base_model_slug="Qwen/Qwen3.6-35B-A3B"
)
print(diagnostic.benchmark_diagnostic_id)
```

The unit tests run in the runtime image:

```bash
docker build -t harvey-lab-trajectory examples/harvey_labs/runtime
docker run --rm -v "$PWD/examples/harvey_labs:/example:ro" harvey-lab-trajectory \
  sh -c 'pip install -q pytest && python -m pytest -q -p no:cacheprovider /example/tests'
```

## Evaluate and train

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

baseline = client.evals.create(
    bench_id=bench_id,
    base_model_slug=model,
    options={**limits, "max_turns_per_trajectory": 200},
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
checkpoint = client.training.checkpoints.retrieve(training.training_run_id, 3)
final = client.evals.create(
    bench_id=bench_id,
    base_model_slug=checkpoint.base_model_slug,
    parent_checkpoint_id=checkpoint.checkpoint_id,
    options={**limits, "max_turns_per_trajectory": 200},
)
```

The reward is the fraction of rubric criteria passed. Harvey's all-pass score rarely differs
across a group of attempts at one task, so it gives training little signal; it is logged as the
zero-weight `all_pass` component, shown with each trajectory's reward. Check the step
summaries' `no_signal_groups` early (see the [training guide](https://docs.trajectory.ai/guides/training)).
An attempt that prints the whole document store in one command can overflow the context and
end as `TRUNCATION`; training skips those attempts by default.

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
judge from `--judge-model`. Secrets come from the platform, not the benchmark source. In managed
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
        "--reasoning-effort low --judge-model gpt-5.4-mini"
    ),
    env_vars={
        "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
        "HARVEY_PODMAN_DISABLE_CGROUPS": "1",
    },
    env_resources=EnvResources(network_mode="public", docker_engine=True),
)
```

The public benchmark contains 2,010 tasks: 1,890 training tasks and 120 held-out test tasks.

## Preserve the original harness

The added model adapter creates one TID and sends the harness's existing Responses API calls
through `trajectory-session`. After the run, it invokes Harvey LAB's normal judge, records the
fraction of criteria passed as `criteria_fraction` and the all-pass result as the zero-weight
`all_pass`, and completes the same trajectory. If the agent loop or
the judge fails, it completes the trajectory with `ERROR` and logs no reward. This keeps prompts,
tools, output files, and grading behavior owned by the upstream benchmark.

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
