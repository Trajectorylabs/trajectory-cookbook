# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

To connect your own harness, start with [native integration and runtime packaging](#package-a-benchmark-runtime).
See [Inspect integration](examples/inspect.md) for its actor-client setup.

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
    x_trajectory_id=tid,
)
model_answer = response.choices[0].message.content
```

#### Log reward to the trajectory

Pass the same TID when the benchmark calculates reward, then mark the task complete.
For an existing harness, preserve its [native score and completion outcome](#report-native-results).
Keep full reports in [trajectory artifacts](examples/diagnostic_artifacts.md), and finish
uploading them before completing the trajectory.

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
response = client.chat.completions.create(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "What is 6 × 7?"}], x_trajectory_id=tid)
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

Set `run_command` to the native entrypoint that solves and grades one task. Preserve its
solver, tools, prompts, stopping conditions, model settings and grader. The SDK integration
connects actor calls to Trajectory and reports the native result; putting a grader description
in `TaskSpec.spec` does not execute it. See the [Harvey LAB](examples/harvey_labs.md),
[Big Finance Benchmark](examples/big_finance_benchmark.md) and [Inspect](examples/inspect.md)
examples.

### Connect the actor

In a managed task runtime, `Client()` routes model calls to the actor selected for evaluation
or training, even if a request names a different model. Keep auxiliary judges and tools on
their configured providers, models and credentials. Do not send their requests through the
actor client or attach its `X-Trajectory-Id` header.

SDK responses are Pydantic models. If the harness validates responses with another library's
model class, pass `response.model_dump()` to that validator.

If the harness also retries model calls, make its retry hook honor `x-should-retry: false`
in error response headers; otherwise retain its existing retry policy. Preserve status,
headers and body when translating SDK exceptions. Retrying every 5xx response can keep a
non-retryable failure running until the task times out.

### Report native results

Call `client.trajectories.log_reward(...)` with the computed native reward, then
`client.trajectories.complete(...)` with the same trajectory ID. Exiting the process does
not complete the trajectory. Preserve the native outcome in `termination_reason`: use
`ENV_DONE` for normal completion, or `MAX_STEPS`, `TIMEOUT` or `ERROR` when applicable.
A reward may exist even when the task did not finish normally.

Forward the native grader's score, including zero, and preserve its handling of failed
candidate solutions. Additional checks on test counts or exit codes can reject valid native
scores. Report infrastructure, integration or grading exceptions as errors rather than
substituting a zero reward.

The platform sums reward components using their weights, which default to 1. Preserve the
benchmark's defined conversion and weighting. Record auxiliary scores with
`client.trajectories.log_event(..., payload=...)` or
`client.trajectories.log_reward(..., weight=0)` unless the benchmark includes them in its
combined reward. Save native grading evidence and original errors before completing the
trajectory and cleaning up its environment. Use events for small summaries and
[artifacts](examples/diagnostic_artifacts.md) for full reports.

### Preserve runtime dependencies

Use the harness's dependency declarations and lockfile together. Add the SDK while retaining
compatible locked versions and Git revisions. Use the lockfile's package manager or export
command to preserve platform markers and dependency groups; installing every entry in a
cross-platform lockfile can select packages unsupported by the runtime operating system.
Install required system tools, including Git for dependencies pinned to Git revisions.
Freezing a newly resolved environment does not preserve the harness's tested dependencies.

With `uv sync --locked`, keep the package index consistent with the lockfile. The managed
builder can supply a mirror that causes a valid lock to be rejected. For a PyPI lock, use
`--default-index https://pypi.org/simple`; use the corresponding index for a private registry.
Keep `--locked` so dependency changes fail the build.

Managed builds do not supply Docker BuildKit's automatic platform arguments, such as
`TARGETARCH`. For binaries that run in the build environment, detect the architecture with
`uname -m` and map it to the vendor's download names. Set cross-compilation targets explicitly.

Check imports, executables and required versions during the image build, using the runtime
entrypoint's plugin-loading sequence. An import failure in an unused helper does not show
that the configured runtime is blocked. After registration and image readiness, test the
actor client in a small managed run using the runtime's dependencies. A standalone request
does not test routing through the selected managed endpoint, which may offer different models.

Inspect ingestion failure items while other runtimes are building. `runtime_build_failed`
with `retryable: true` does not mean the image will rebuild automatically in that operation.
Correct the cause before submitting again. A polling timeout leaves server processing
running; reconnect to the existing operation instead of resubmitting.

### Choose the runtime and its files

`BenchmarkSpec.runtime`, or a task's `runtime` override, is the image in which `run_command`
executes. Use one shared runtime when the same harness can select a task through its command:

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

Use `TaskSpec.runtime` when tasks need different dependencies, isolation or smaller build
contexts, for example `runtime=DockerfileBuild(f"runtimes/{task_id}/Dockerfile")`. Include
common harness code in each context or supply it through a base image.

The SDK uploads files under each Dockerfile's directory, filtered by `.dockerignore` at the
benchmark root; it does not select files by reading `COPY` statements. A root Dockerfile
therefore includes the checkout unless files are excluded. Keep files required by package
metadata, such as a README referenced by `pyproject.toml`, alongside the code and runtime inputs.
Shared file paths upload once per submission, but the SDK does not automatically isolate one
task's files from another's. Preserve native directory layout and task isolation.

If the harness starts separate task containers or remote sandboxes, deliver each task's
inputs and attachments there before the actor starts. Preserve their contents and expected
paths, and verify access through the actor's tools. Files in the harness image are not
automatically available in another environment. Keep private answers and hidden tests in
the grader's protected environment, inaccessible to the actor and its tools.

For harness-managed containers, package the native harness and its dependencies in the
runtime, with the original task Dockerfiles and build contexts as inputs to its build flow.
Building task Dockerfiles as stages of the harness image does not make those images available
to the harness's Docker daemon. Preserve image configuration as well as files: use the original
build or [`docker save` / `docker load`](https://docs.docker.com/reference/cli/docker/image/load/).
Importing a root-filesystem archive loses settings such as the entrypoint and user.

Choose an existing native backend compatible with the runtime's resources. A local Docker
backend can use the daemon below; an external sandbox backend needs its own credentials.
Trajectory does not pass infrastructure credentials into the runtime. Preserve native task
and grading behavior whichever backend you use.

### Run native Docker environments

For local Docker builds, image loads or container runs, set
`env_resources=EnvResources(docker_engine=True)` on the task. Import `EnvResources` from
`trajectory.types.benchmarks.task_spec`. Set network access separately.

The harness runtime must include `dockerd` on `PATH`, the Docker CLI and any required plugins,
such as Compose, at the versions the harness needs. Enabling `docker_engine` starts the daemon;
it does not install dependencies or replace the native task environment. Install the engine
with its operating-system dependencies. On Debian-based images, include `kmod` and `iproute2`
for module and network utilities, and preserve helper scripts' installation paths.

Check executables during the image build. For a harness that uses Compose:

```dockerfile
RUN dockerd --version && docker --version && docker compose version
```

Then check daemon startup and native prerequisites inside a task runtime before a full
evaluation. Version checks alone do not establish that the daemon or harness can start.
Set `EnvResources.cpus` and `memory_mb` for the work inside each runtime, including native
image builds. A ready image does not show that those resources are sufficient.

### Keep build contexts within service limits

Each runtime build context can contain at most 4,096 uploaded files and 3 GiB. Scope the
context to the files it needs. For many small files, archive them and extract them during
the image build. Keep unpacked sources outside the context so both copies are not uploaded.
The archive still counts toward the byte limit; task-specific runtimes can use smaller contexts.

Use `COPY` followed by `RUN tar`; managed builds do not support local archive extraction
with `ADD`:

```dockerfile
COPY tasks.tar.gz /tmp/tasks.tar.gz
RUN mkdir -p /opt/benchmark \
    && tar -xzf /tmp/tasks.tar.gz -C /opt/benchmark \
    && rm /tmp/tasks.tar.gz
```

For external services, use `network_mode="allowlist"` with `allowed_hosts`, or
`network_mode="public"` when unrestricted access is required. Supply credentials through
`SecretRef`. Pin external task or service versions where supported; otherwise record available
version information. Test native execution and grading after the image builds.

## Examples

- [Inspect](examples/inspect.md): connect a native Inspect actor while preserving its solver and scorer.
- [Diagnostic artifacts](examples/diagnostic_artifacts.md): retain full reports alongside trajectory events.

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
