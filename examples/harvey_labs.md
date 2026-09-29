# Harvey LAB

The public [Harvey LAB SDK integration PR](https://github.com/Trajectorylabs/harvey-labs/pull/7)
shows how to adapt an existing multi-turn benchmark without replacing its agent loop, tools,
grader, or Podman sandbox. The integration is a small overlay on a pinned revision of the public
[Harvey LAB repository](https://github.com/harveyai/harvey-labs).

Use this pattern when a benchmark already owns its task lifecycle and evaluation logic. The SDK
adapter only needs to connect that lifecycle to a Trajectory session:

1. Create one trajectory when a task starts.
2. Forward every model request with the same trajectory ID.
3. Run the benchmark's original judge after the agent loop.
4. Log the resulting reward and complete the trajectory.

## Ingest the existing task tree

The uploader discovers the benchmark's existing `task.json` files and assigns a deterministic
held-out split. Hashing the task path keeps the same 120 test tasks across repeated ingestions:

```python
ordered = sorted(
    tasks,
    key=lambda task: hashlib.sha256(f"harvey-test-v1:{task}".encode()).digest(),
)
test_tasks = set(ordered[:120])
```

Each task runs the original harness through the Trajectory-backed model adapter. Secret values
are supplied by the platform and are not stored in the benchmark source:

```python
TaskSpec(
    name=task,
    split="test" if task in test_tasks else "train",
    run_command=(
        "python -m lab_core.harness.run --model trajectory/session "
        f"--task {task} --run-id trajectory --max-turns 200"
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
all-pass result as `reward_accuracy`, and completes the same trajectory. This keeps prompts,
tools, output files, and grading behavior owned by the upstream benchmark.

The runtime image follows the same overlay approach: download a pinned public source revision,
then copy only the SDK adapter and the narrow harness changes into the image. Pinning the source
prevents upstream changes from silently changing an ingested benchmark.

## Run Podman inside the benchmark runtime

Harvey LAB already isolates document tools in Podman. The outer benchmark runtime therefore
requests a container engine with `docker_engine=True`, and the original harness launches its
inner Podman container as usual.

Some nested environments use `crun` without delegated cgroup controllers. The example detects
that runtime and adds `--cgroups=disabled` only for the managed nested-Podman path. Local Harvey
LAB runs retain their default cgroup behavior.

See the [complete public PR](https://github.com/Trajectorylabs/harvey-labs/pull/7) for the
uploader, runtime Dockerfile, adapter, and Podman integration.
