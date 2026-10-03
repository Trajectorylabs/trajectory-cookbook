# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

To connect your own harness, start with [runtime packaging](#package-a-benchmark-runtime).

## Setup

Install the latest SDK and authenticate:

```bash
pip install --upgrade trajectory-sdk
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment.

## Quickstart

This quickstart trains and evaluates Qwen 3.5 4B on a small GSM8K benchmark with 64 training
problems and 16 held-out test problems. Start by capturing one math task, then package the same
task loop and grader as a benchmark for repeatable evaluation and training.

### 1. Inject the Trajectory SDK into your benchmark

#### Replace the OpenAI client with the Trajectory client

```python
# Before
from openai import OpenAI

client = OpenAI()
```

```python
# After
from trajectory import Client

client = Client()
```

#### Create a TID when a task starts

Create one trajectory for each task and keep its ID for the full task lifecycle.

```python
tid = client.trajectories.create().tid
```

#### Pass the TID to LLM calls

```python
response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": prompt}],
    extra_headers={"X-Trajectory-Id": tid},
)
model_answer = response.choices[0].message.content
```

#### Log reward to the trajectory

Pass the same TID when the benchmark calculates reward, then mark the task complete.

```python
reward = float(check_answer(model_answer, expected_answer))
client.trajectories.log_reward(
    tid,
    reward_id="correctness",
    name="reward_accuracy",
    value=reward,
)
client.trajectories.complete(tid)
```

#### Run your benchmark and see the result

```python
from trajectory import Client
client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "What is 6 × 7?"}], extra_headers={"X-Trajectory-Id": tid})
reward = float(response.choices[0].message.content.strip() == "42")
client.trajectories.log_reward(tid, reward_id="correctness", name="reward_accuracy", value=reward)
client.trajectories.complete(tid)
print(client.trajectories.retrieve(tid, include_steps=True))
```

### 2. Upload GSM8K to the Trajectory Platform

The GSM8K example contains 64 training tasks and 16 held-out test tasks. The uploader writes each
task to a local JSON file and packages those files into the runtime image. Each task selects its
file through `run_command`; no task content is passed through environment variables. The harness
sends only the question to the model and requires it to call `submit_answer`:

```python
TaskSpec(
    name="gsm8k/train_0001",
    split="train",
    run_command=(
        "python -u /opt/gsm8k/gsm8k_harness.py "
        "--task-file /opt/gsm8k/tasks/train_0001.json"
    ),
    tags=["gsm8k"],
)
```

`TaskSpec.name` is your dataset's task identifier. When you retrieve a benchmark with
`client.benchmarks.specs.retrieve(..., include_tasks=True)`, each task exposes that name as
`label` and a platform-assigned `task_id` for API calls. Omit `TaskSpec.id` when registering
new tasks.

Set `TaskSpec.split` explicitly to preserve a dataset's train/test membership. If omitted,
ingestion deterministically assigns approximately 15% of those tasks to TEST and the rest to
TRAIN. For an evaluation-only dataset, set `split="test"` on every task.

Package the tasks and runtime, upload the benchmark, and wait for its runtime image to become
ready:

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

client = Client()
agent = client.agents.create(name="gsm8k-cookbook")
print(agent.agent_id)
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=tasks,
)
result = push(
    client,
    benchmark,
    agent_id=agent.agent_id,
    root=Path("my-benchmark"),
)
bench_id = result.bench_id
wait_for_benchmark_images(client, bench_id)
```

Run the complete uploader with the printed agent ID, then save the benchmark ID:

```bash
uv run examples/gsm8k/ingest.py --agent-id agt_<your-agent-id>
```

```text
agent_id=agt_<your-agent-id>
bench_id=bm_<32-hex>
```

If your benchmark needs an organization secret, register it after ingestion. For example,
register a local OpenAI API key without putting its value in source:

```python
import os

from trajectory import Client

client = Client()
client.secrets.create(
    name="OPENAI_API_KEY",
    value=os.environ["OPENAI_API_KEY"],
)
```

`SecretRef` is a named pointer, not the secret value itself. In a `TaskSpec`,
`SecretRef(secret_ref="OPENAI_API_KEY")` injects this organization secret at runtime.

Managed task runtimes receive `TRAJECTORY_API_KEY` and `TRAJECTORY_BASE_URL` automatically;
use `Client()` inside the harness. Do not put `TRAJECTORY_*` names in `env_vars`, including
through `SecretRef`. Use `SecretRef` for additional credentials the benchmark needs.

### 3. Evaluate, train, and compare on the Trajectory Platform

Training and evaluation use `create`, `base_model_slug`, `parent_checkpoint_id`, and the
same `options` schema. Discover the supported settings and bounds for each mode:

```python
for resource in (client.training, client.evals):
    catalog = resource.list_options(bench_id=bench_id, base_model_slug="Qwen/Qwen3.5-4B")
    for model in catalog.models:
        print(model.base_model_slug, model.options)
```

Omit `options`, or pass `{}`, for defaults. Leave computed defaults unset so the server
can resolve them. Run the benchmark before training so you have a frozen baseline:

```python
baseline = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    display_name="GSM8K baseline",
    options={"disable_thinking": True, "evaluation_max_samples": 16},
)
baseline_eval_run_id = baseline.eval_run_id
```

Start training:

```python
training = client.training.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    options={
        "disable_thinking": True,
        "num_steps": 20,
        "train_batch_size": 4,
        "max_output_tokens_per_step": 32_768,
        "max_turns_per_trajectory": 1,
        "max_response_chars_per_tool_call": 128,
    },
)
training_run_id = training.training_run_id
```

Each optimizer step uses four task groups. With eight samples per group, that produces 32
rollouts per step; check `samples_per_instance` in the model's option defaults.
Poll `client.training.runs.retrieve(training_run_id)` until the run terminates.

Resolve and evaluate the final checkpoint on the same held-out tasks:

```python
checkpoint = client.training.checkpoints.retrieve(training_run_id, step_index=20)
final = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    parent_checkpoint_id=checkpoint.checkpoint_id,
    display_name="GSM8K trained checkpoint",
    options={"disable_thinking": True, "evaluation_max_samples": 16},
)
```

### 4. Deploy and query the trained checkpoint

Deploy the final checkpoint through Model Endpoint. A production deployment becomes the active
deployment for its model slug:

```python
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    model_slug="gsm8k-trained",
    role="production",
)
print(deployment.deployment_id)
```

Query the deployment through the same OpenAI-compatible client:

```python
response = client.chat.completions.create(
    model="gsm8k-trained",
    messages=[{"role": "user", "content": "What is 17 × 6?"}],
)
print(response.choices[0].message.content)
```

## Package a benchmark runtime

Set `run_command` to the entrypoint that runs a task and reports its reward. It can invoke
an existing harness that handles both solving and grading. Connect the solving agent's model
calls to the SDK, and preserve the harness's grading logic and configured model settings.
A managed trajectory routes completion requests to its configured actor endpoint, including
requests that name a different model. Keep auxiliary judges and tools on their own provider
clients and credentials.
The entrypoint must call `client.trajectories.log_reward(...)` with the computed reward, then
`client.trajectories.complete(...)` with the same trajectory ID. Exiting the process does not
complete the trajectory. Putting a grader description in `TaskSpec.spec` does not execute it.
See the [Harvey LAB](examples/harvey_labs.md) and
[Big Finance Benchmark](examples/big_finance_benchmark.md) integrations for examples.

Use the harness's dependency declarations and lockfile together. Add the SDK while retaining
compatible locked versions and Git revisions, and review any required dependency changes.
Install the system tools those dependencies need, such as Git for packages pinned to Git revisions.
Freezing a freshly resolved environment pins those new versions; it does not preserve the harness's tested
dependencies. Before building all task images, test a short model request through the same
client and dependency set the runtime will use. Successful imports alone do not verify this path.

Forward the native grader's score, including zero. Preserve its handling of failed candidate
solutions; additional checks on test counts or exit codes can reject valid native scores.
Report infrastructure or integration failures as execution errors rather than assigning a score.

Preserve the benchmark's primary metric when logging reward: the platform sums reward
components using their weights, which default to 1. Record auxiliary scores with
`client.trajectories.log_event(..., payload=...)` or
`client.trajectories.log_reward(..., weight=0)` unless the benchmark intentionally includes
them in its combined reward. Save grading evidence and original error details as trajectory
events or artifacts before completing the trajectory and cleaning up its environment.

The SDK uploads the files under each Dockerfile's directory, filtered by `.dockerignore`
at the benchmark root. It does not select files by reading `COPY` statements. A Dockerfile
at the repository root therefore includes the whole checkout unless files are excluded.

Prefer one shared `BenchmarkSpec.runtime` when the same harness can select a task through
its `run_command`. This also works when the harness builds and launches task containers from
their original Dockerfiles and build contexts.

```python
from trajectory import BenchmarkSpec, TaskSpec
from trajectory.lib import DockerfileBuild

benchmark = BenchmarkSpec(
    name="coding-tasks",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=[
        TaskSpec(
            name=task_id,
            run_command=f"python /app/harness.py --task-id {task_id}",
        )
        for task_id in ("task-a", "task-b")
    ],
)
```

Set `TaskSpec.runtime` to override the shared runtime when tasks need different dependencies,
isolation, or smaller build contexts, for example
`runtime=DockerfileBuild(f"runtimes/{task_id}/Dockerfile")`. Each Dockerfile's directory must
contain the files its build needs. Supply common harness code and dependencies through a
base image or include them in each context.

Preserve the native harness's directory layout, behavior, and isolation between task runs.
Keep private answers and hidden tests in the grader's protected environment, inaccessible to
the solving agent and its tools. The SDK uploads shared file paths once per submission, but
it does not automatically separate one task's files from another's.

If the harness launches a container, preserve its image configuration as well as its files.
Importing a root-filesystem archive does not retain the original entrypoint, user, or other
image settings. Use the original image build or transfer the image with
[`docker save` / `docker load`](https://docs.docker.com/reference/cli/docker/image/load/).

`BenchmarkSpec.runtime`, or a task's runtime override, is the image in which `run_command`
executes. If the harness manages separate task containers, install the harness and its
dependencies in this runtime, and include the original task Dockerfiles and build contexts
as inputs to its existing build flow. Building those task Dockerfiles as stages of the
harness image does not make their images available to the harness's Docker daemon.

Choose an existing harness backend that works with the runtime's available resources.
For example, a harness with a local Docker backend can use the daemon described below.
Supply any credentials an external sandbox backend requires; Trajectory does not pass
its infrastructure credentials into your runtime. Preserve the task environment and grading
behavior when choosing a backend.

For local Docker builds, image loads, or container runs, set
`env_resources=EnvResources(docker_engine=True)` on the task. Configure network access
separately: use `network_mode="public"` when the harness requires unrestricted external access.
The harness runtime must include the Docker daemon at `/usr/bin/dockerd`, the Docker CLI,
and any plugins the harness uses, such as Compose. Enabling `docker_engine` starts the
daemon; it does not install these dependencies or replace the native task environment.
Use the versions required by your harness, including its CLI and plugin requirements.
Check required executables during the image build so missing tools fail before task execution.
For a harness that uses Compose:

```dockerfile
RUN /usr/bin/dockerd --version && docker --version && docker compose version
```

Then run the harness's prerequisite checks inside a task runtime before a full evaluation.
These build-time checks do not establish that the daemon or harness can start.
Set `EnvResources.cpus` and `memory_mb` for the work done inside each task runtime,
including native image builds. A ready image does not establish that those resources are sufficient.

The current service limits are 4,096 uploaded files and 3 GiB per runtime build context.
For many small files, create a compressed archive and extract it during the image build.
Use `COPY` followed by `RUN tar`, as below: Modal-backed builds do not support local
archive extraction with `ADD`.
Keep the unpacked source outside the context so it is not uploaded alongside the archive.
The archive still counts toward the byte limit; separate runtimes can use smaller contexts
containing only their required files.

```dockerfile
COPY tasks.tar.gz /tmp/tasks.tar.gz
RUN mkdir -p /opt/benchmark \
    && tar -xzf /tmp/tasks.tar.gz -C /opt/benchmark \
    && rm /tmp/tasks.tar.gz
```

For external services, use `network_mode="allowlist"` with `allowed_hosts`, or
`network_mode="public"` when unrestricted access is required. Supply credentials through
`SecretRef`. Import `EnvResources` from `trajectory.types.benchmarks.task_spec`.
Test task execution and grading after the image builds. Pin external task or service versions
when supported; otherwise record any version information available and note that repeated
runs may differ.

## Examples

When adapting an existing benchmark, read the cookbook recipe together with its complete public
implementation PR:

- [Big Finance Benchmark SDK integration](https://github.com/Trajectorylabs/big-finance-benchmark-public/pull/1)
- [Harvey LAB SDK integration](https://github.com/Trajectorylabs/harvey-labs/pull/11)

- [GSM8K](examples/gsm8k/README.md): adapt a public exact-match math benchmark for evaluation
  and training.
- [Big Finance Benchmark](examples/big_finance_benchmark.md): connect an existing ReAct research
  benchmark and rubric grader to evaluation-only Trajectory sessions.
- [Harvey LAB](examples/harvey_labs.md): inject the SDK into an existing multi-turn benchmark
  while retaining its original harness, judge, and nested Podman sandbox.
- [T Factory](examples/t_factory/README.md): train, deploy, and query a model rewarded for using
  words beginning with `T`.

## Repository layout

```text
examples/
├── big_finance_benchmark.md  # Existing ReAct harness and rubric grader
├── gsm8k/          # Exact-match math through submit_answer
├── harvey_labs.md  # Existing harness integration with nested Podman
└── t_factory/      # Maximize the fraction of words beginning with T
```

## Beta testing and support

- **Schedule a beta testing call:** [Book a 30-minute call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing) — invite your friends!
- **Feedback and support:** [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
