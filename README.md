# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

To connect your own harness, follow the [ingestion and runtime-readiness walkthrough](examples/ingestion.md)
and [native integration and runtime packaging](#package-a-benchmark-runtime).
See [Inspect integration](examples/inspect.md) for its actor-client setup.
Use [task diagnostics](examples/task_diagnostics.md) to validate one task directly from local
runtime files—significantly faster than ingesting the whole benchmark and running a full
evaluation. For an uploaded benchmark, use
[benchmark diagnostics](examples/task_diagnostics.md#check-an-uploaded-benchmark)
to quickly validate a few tasks across different runtimes.

## Setup

This branch targets the upcoming matching API/SDK release. The examples require the
`log_reward(tid, name=..., value=...)` signature, `start_push` and managed ingestion with
`build_images=True`; the ingestion and diagnostics guides identify additional release
dependencies. Wait for the compatible SDK to be published and its API deployed before
running this branch against a released service. Released-client verification remains pending.

After that release, install the compatible SDK and authenticate:

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
client.trajectories.log_reward(tid, name="reward_accuracy", value=reward)
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
from trajectory.lib import DockerfileBuild, push

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
    build_images=True,
)
bench_id = result.bench_id
```

`build_images=True` builds the runtime as part of the ingestion operation. `push` waits
for registration and those builds to finish.

Run the complete uploader with the printed agent ID, then save the benchmark ID:

```bash
uv run examples/gsm8k/ingest.py --agent-id agt_<your-agent-id>
```

```text
agent_id=agt_<your-agent-id>
bench_id=bm_<32-hex>
```

The [ingestion walkthrough](examples/ingestion.md#authenticate-each-operation) explains SDK and upload
authorization. Public prebuilt images need no registry credentials. Register task credentials before execution.
For example, register a local OpenAI API key without putting its value in source:

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

If an evaluation fails, [inspect every selected task and attempt](examples/diagnostic_artifacts.md#2-run-a-small-managed-evaluation-and-inspect-it-while-it-runs).
Reward listings omit attempts that failed without a grade.

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

`run_command` starts the code that solves and grades a task. It can call your harness's
entrypoint directly or use a wrapper to add SDK integration. Connect the actor's model calls
and report the computed reward while retaining the harness's solving and grading logic.
A grader description in `TaskSpec.spec` does not execute the grader. See the
[Harvey LAB](examples/harvey_labs.md), [Big Finance Benchmark](examples/big_finance_benchmark.md)
and [Inspect](examples/inspect.md) examples.

### Connect the actor

In a managed task runtime, `Client()` routes model calls to the actor selected for evaluation
or training, even if a request names a different model. For a fixed judge or other auxiliary
model, use a separate client configured for its endpoint and credentials. Sending its requests
through the managed actor client would use the actor model instead. See the
[fixed auxiliary model example](examples/auxiliary_clients.md) for endpoint, authentication,
independent trajectory and native OpenAI/LiteLLM configuration.

SDK responses are Pydantic models. If the harness validates responses with another library's
model class, pass `response.model_dump()` to that validator.

If the harness retries model calls, honor `x-should-retry: false` in error response headers.
Preserve status, headers and body when translating SDK exceptions so the harness can distinguish
retryable failures from failures that should stop the task.

### Report native results

Call `client.trajectories.log_reward(...)` with the grader's score, including zero, then
`client.trajectories.complete(...)` with the same trajectory ID. Use the benchmark's score
conversion and weighting. Report execution or grading errors as failures rather than
substituting a zero reward.

Preserve penalties and fallback scores defined by the native scorer, with their reasons.
An auxiliary or report failure must not erase an independently valid native score. If required
inputs to the native score are missing, leave it ungraded; do not invent a score or average
over only the available grading outputs. Use the benchmark's source to determine which
outputs are required and what its fallback scores mean; record missing outputs separately
from the score and execution outcome.

Set `termination_reason` from the outcome of the task's execution and grading. Use `ENV_DONE`
when that operation completes normally, including when a solver reaches its own stopping
condition and the benchmark grades the result. Use the corresponding failure or limit reason
when the trajectory itself stops early. `MAX_STEPS`, `LIMIT_REACHED` and `TRUNCATION` cap
positive evaluation rewards at zero, even if a raw reward was logged. Platform-enforced
limits still apply. Report exception paths with the failure reason before re-raising the
original exception; process exit alone does not complete the trajectory.

Reward components are summed with their weights, which default to 1. Record diagnostic scores
that are not part of the benchmark's reward with `client.trajectories.log_event(..., payload=...)`.
Events store JSON data without changing the reward. Even with `weight=0`, `log_reward` creates
a reward record and can prevent the default automatic grader from running. A missing optional
grader output need not invalidate a score; preserve the benchmark's handling of required and
optional outputs.
Use events for summaries and [artifacts](examples/diagnostic_artifacts.md) for full reports.
Finish recording them before completing the trajectory and cleaning up the environment.
If a platform limit closes model calls, the running harness can still publish diagnostics
while the trajectory is `cancelling`; new reward writes remain closed. See the
[reporting window](examples/diagnostic_artifacts.md#record-events-and-artifacts) and finish
uploads before finalization begins.

### Choose the runtime and its files

`BenchmarkSpec.runtime` sets the default image in which tasks' `run_command` executes.
Tasks that use the same harness and dependencies can share a runtime and select their inputs
through the command:

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

Use `TaskSpec.runtime` to override the image for tasks with different dependencies or smaller
build contexts, for example `runtime=DockerfileBuild(f"runtimes/{task_id}/Dockerfile")`.
Include common harness code in each context or supply it through a base image.

Use `DockerfileBuild` for ordinary local files or a digest-pinned `ImageRef` for an image you
build and publish. Both use the same [submit, inspect and repair loop](examples/ingestion.md).

The upcoming managed path automatically packages each Dockerfile's directory, filtered by
`.dockerignore` at the benchmark root. It retains the selected Dockerfile and does not infer files
from `COPY` statements. Keep ordinary files and `COPY` commands; no customer archive step is needed.
A root Dockerfile selects the checkout unless you exclude files. Selecting a task does not exclude
other tasks' files. See the walkthrough's release note before using the new delivery features.

If the harness starts separate task containers or remote sandboxes, make its inputs available
there at the expected paths. Files in the harness image are not automatically available in
another environment. Preserve the benchmark's separation between actor-visible inputs and
private answers or hidden tests.

For harness-managed containers, package the task Dockerfiles and build inputs where the
harness expects them. To reuse prebuilt images, [`docker save` / `docker load`](https://docs.docker.com/reference/cli/docker/image/load/)
preserves image configuration. Importing a root-filesystem archive loses settings such as the
entrypoint and user. Building a task image in an outer Dockerfile stage does not load it into
the harness's Docker daemon.

### Install and check dependencies

Install the harness's dependencies and the SDK. Where the harness provides a lockfile, use its
package manager to retain compatible versions, platform markers and dependency groups.
Check the imports, executables and plugins used by the runtime entrypoint during the image
build. Check execution and grading in a small managed run after image readiness; build success
alone does not verify model routing, external services or task execution.

For `uv sync --locked`, the package index must match the lockfile. If the build environment
supplies a different index, pass the lockfile's index explicitly; for PyPI, use
`--default-index https://pypi.org/simple`.

Managed builds do not supply Docker BuildKit's automatic platform arguments, such as
`TARGETARCH`. For binaries that run in the build environment, detect the architecture with
`uname -m` and map it to the vendor's download names. Set cross-compilation targets explicitly.

Inspect failed ingestion items while other runtimes build. `runtime_build_failed` with
`retryable: true` does not automatically rebuild the image in that operation. Correct the
cause and use the [appropriate retry](examples/ingestion.md#4-fix-the-cause-and-retry-the-affected-work).
A polling timeout leaves server processing running; reconnect to the existing operation to check
its outcome.

### Run native Docker environments

If your harness uses local Docker, set `env_resources=EnvResources(docker_engine=True)` on
the task. Import `EnvResources` from `trajectory.types.benchmarks.task_spec`.
The runtime must include `dockerd` on `PATH`, the Docker CLI, their operating-system dependencies,
and any plugins the harness needs. The flag starts the daemon; it does not install those tools.
For a harness that uses Compose, check the installed executables during the image build:

```dockerfile
RUN dockerd --version && docker --version && docker compose version
```

These checks confirm that the tools are installed, not that their versions satisfy your
pinned harness. In a managed task runtime, run the harness's own container prerequisite check
before requesting model work. Verify its required client/server versions and command output;
a working daemon or a successful `--version` command alone does not establish compatibility.

Size `EnvResources.cpus` and `memory_mb` for the work done there, including any task image
builds. If the harness uses an external sandbox service instead, supply that service's
credentials through `SecretRef`.

Network access is configured separately: use `network_mode="allowlist"` with `allowed_hosts`,
or `network_mode="public"` when unrestricted access is required.

For external task or grading services, pin a version where supported. Otherwise record the
available version information and note that later runs may use different service behavior.

### Keep build contexts focused

Select files through the context directory and `.dockerignore`. The upcoming SDK packages
ordinary files automatically. Source-context limits protect upload and worker resources; they
are separate from registry-image limits and the dataset's total task count. Follow the
[ingestion walkthrough](examples/ingestion.md#managed-build-keep-your-dockerfile-and-files)
for the supported dimensions, observed/allowed errors and recovery steps.

## Examples

- [Task diagnostics](examples/task_diagnostics.md): upload local runtime files and diagnose one
  task without an existing benchmark.
- [Ingestion and runtime readiness](examples/ingestion.md): choose a delivery path, account for every task and repair runtime failures.
- [Inspect](examples/inspect.md): connect a native Inspect actor while preserving its solver and scorer.
- [Execution and native reports](examples/diagnostic_artifacts.md): inspect attempts, preserve grading outcomes and retain reports.

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
- [Number guessing](examples/number_guessing/README.md): guess a fake user's number with
  same-model context compaction every three guesses and an efficiency reward.

## Repository layout

```text
examples/
├── big_finance_benchmark.md  # Existing ReAct harness and rubric grader
├── gsm8k/          # Exact-match math through submit_answer
├── harvey_labs.md  # Existing harness integration with nested Podman
├── t_factory/      # Maximize the fraction of words beginning with T
└── number_guessing/ # Same-model compaction and a two-number training distribution
```

## Beta testing and support

- **Schedule a beta testing call:** [Book a 30-minute call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing) — invite your friends!
- **Feedback and support:** [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
