# Run a diagnostic on one task

Use `start_task_diagnostic()` to check a task before uploading a full benchmark. No existing
benchmark is needed. A diagnostic reports each task's status. A failed task also reports its
termination reason, trajectory ID and, if the harness raised, a redacted exception summary.

Install the SDK and authenticate:

```bash
pip install --upgrade trajectory-sdk
export TRAJECTORY_API_KEY="..."
```

This example uses the cookbook's [T Factory runtime](t_factory/runtime/).
[Choose or create an agent](https://docs.trajectory.ai/agent), then run the Python code
from the repository root, replacing `YOUR_AGENT_NAME` with its name.

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
    agent_name="YOUR_AGENT_NAME",
    root=Path("examples/t_factory"),
    base_model_slug="Qwen/Qwen3.5-4B",
    options={"reasoning_effort": "off", "max_output_tokens_per_step": 2048},
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
    print(
        task_result.task_id,
        task_result.status,
        task_result.termination_reason,
        task_result.reward,
        task_result.failure,
    )
```

`options` takes the settings an evaluation accepts, apart from sample counts, such as
`reasoning_effort`, `max_output_tokens_per_step`, and `execution_timeout_seconds`. Pass the
model and settings you will evaluate and train with. Each task reports its trajectory ID,
ending, and reward, which is null when the trajectory ended before the harness recorded one.
Inspect both the run's failure and each task's result before moving on to a full evaluation
or training run. Keep the printed diagnostic ID to retrieve the result again later.

## Check an uploaded benchmark

For a benchmark whose images are ready, use benchmark diagnostics for a quick check of a few
tasks across different runtimes. It selects up to 10 tasks, prioritizing distinct runtimes,
without executing a full evaluation.

```python
diagnostic = client.diagnostics.start_benchmark(
    bench_id="YOUR_BENCH_ID",
    base_model_slug="Qwen/Qwen3.5-4B",
    options={"reasoning_effort": "off", "max_output_tokens_per_step": 2048},
)
print(diagnostic.benchmark_diagnostic_id)
```

Use the same status polling and result inspection shown above.
