# Run a diagnostic on one task

Use `start_task_diagnostic()` to check a task before uploading a full benchmark. It uploads
local runtime files, registers the task, waits for the image build, and starts diagnostics.
You need an agent ID, but no existing benchmark, image, or runtime ID.

Install the latest SDK and authenticate:

```bash
pip install --upgrade trajectory-sdk
export TRAJECTORY_API_KEY="..."
```

This example uses the cookbook's [T Factory runtime](t_factory/runtime/). Run the Python code
from the repository root, replacing `YOUR_AGENT_ID` with your agent's ID. If you need an agent,
create one with `Client().agents.create(name="task-diagnostic-example").agent_id`.

```python
from pathlib import Path

from trajectory import Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_task_diagnostic

client = Client()
task = TaskSpec(
    name="t-word-density-diagnostic",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    run_command="python -u /opt/t_factory/t_factory_harness.py",
    env_vars={"USER_PROMPT": "Do you enjoy rainy weather? Why?"},
)
diagnostic = start_task_diagnostic(
    client,
    task,
    agent_id="YOUR_AGENT_ID",
    root=Path("examples/t_factory"),
)
print(diagnostic.benchmark_diagnostic_id, diagnostic.bench_id)
```

The helper returns after starting the diagnostic; wait for its result separately:

```python
import time

while True:
    status = client.diagnostics.get_status(diagnostic.benchmark_diagnostic_id)
    if status.status in {"completed", "failed", "cancelled"}:
        break
    time.sleep(5)

result = client.diagnostics.get_diagnostics(diagnostic.benchmark_diagnostic_id)
print(result.status, result.task_counts, result.failure)
for task_result in result.tasks:
    print(task_result.task_id, task_result.status, task_result.failure)
```

Inspect both the run's failure and each task's result before moving on to a full evaluation
or training run. Keep the printed diagnostic ID to retrieve the result again later.

For your own task, set `root` to your local project directory and make the Dockerfile path
relative to it. The Dockerfile's directory is the build context: include the harness and its
required files there, and make `run_command` match their paths inside the image. See
[runtime packaging](../README.md#package-a-benchmark-runtime) for dependencies and build limits.

You can pass `base_model_slug` to select the diagnostic model and `timeout_seconds` to adjust
the image-build wait. If you already have an image or registered runtime, use `ImageRef(...)`
or `RuntimeRef(...)` in the task's `runtime` field instead of `DockerfileBuild(...)`.
