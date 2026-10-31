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
  [Trajectory adapter](runtime/lab_core/harness/adapters/trajectory.py) and failure diagnostics,
  and changes the harness, judge, and sandbox files they hook into.
  [`runtime/NOTICE`](runtime/NOTICE) carries Harvey LAB's MIT license.
- [`tests/`](tests/): unit tests for the adapter and the sandbox's cgroups switch.

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
store its key as `OPENAI_API_KEY` and pass a model name and endpoint; the endpoint becomes each
task's `OPENAI_BASE_URL`. The judge picks its client from the model name, so use a `gpt` name:

```bash
uv run examples/harvey_labs/ingest.py --agent-name YOUR_AGENT_NAME \
  --judge-model gpt-6-luna --judge-base-url https://openrouter.ai/api/v1
```

The script prints the benchmark ID and waits for the runtime image, which downloads the
3 GB source archive. Image readiness alone doesn't verify model inference, grading, or reward
logging, so run a [benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark)
before an evaluation or training run.

The unit tests run in the runtime image:

```bash
docker build -t harvey-lab-trajectory examples/harvey_labs/runtime
docker run --rm -v "$PWD/examples/harvey_labs/tests:/tests:ro" harvey-lab-trajectory \
  sh -c 'pip install -q pytest && python -m pytest -q -p no:cacheprovider /tests'
```

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
through `trajectory-session`. After the run, it invokes Harvey LAB's normal judge, records its
all-pass result as `reward_accuracy`, and completes the same trajectory. If the agent loop or
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
