# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

For coding agents, use the [Trajectory skill](skills/trajectory/SKILL.md) alongside your
benchmark's source. It covers integration, validation, evaluation, training, and diagnosis.
Copy `skills/trajectory/` into `.agents/skills/` in your benchmark repository for Codex, or
`.claude/skills/` for Claude Code. Ask your agent to read and use the `trajectory` skill for your task.

Runnable examples and runtime images pin `trajectory-sdk==0.9.20`.
The [public Tau 3 Banking recipe](examples/tau3_banking/banking.md) preserves the upstream evaluator.

**Start here:** [Run your first task with T Factory](examples/t_factory/README.md).
See one model response, its reward, and a completed execution. 

## Examples Guide


| Example | What you learn |
| --- | --- |
| [T Factory: first task](examples/t_factory/README.md) | Simplest example of model response, reward, completion |
| [GSM8K: full workflow](examples/gsm8k/README.md) | Next: validate one task, evaluate and train a small benchmark, scale to more tasks, and optionally deploy. Uses a real dataset with private answers and tool-based grading. |
| [T Factory: evaluation and training](examples/t_factory/training.md) | An alternative training walkthrough using the first lesson's toy reward. |
| [Number guessing](examples/number_guessing/README.md) | A multi-turn environment, efficiency rewards, and same-model context compaction. |
| [Inspect](examples/inspect.md) | Actor routing, native scoring, errors, and a separate judge. |
| [Harvey LAB](examples/harvey_labs.md) | An existing multi-turn harness, its native judge, and nested Podman. |
| [Big Finance Benchmark](examples/big_finance_benchmark.md) | An existing ReAct research agent, web tools, and rubric grading. |

The SDK lifecycle stays the same as the benchmark becomes more complex. Keep the benchmark's
native solving and grading rules when adapting it.

Supporting Guides:

- [Task Upload Validation](examples/task_diagnostics.md): validate one task or inspect a few uploaded tasks.
- [Diagnostic artifacts](examples/diagnostic_artifacts.md): keep full reports and logs with a trajectory.
- [Ending a trajectory](examples/termination_reasons.md): choose the termination reason and see how each one is scored.
- [Runtime packaging](#package-a-benchmark-runtime): files, dependencies, secrets, and nested containers.
- [Production traces](examples/production_traces.md): capture application requests under an agent (private beta).

The sections below are SDK reference material for adapting your own benchmark. Follow the
[first-task guide](examples/t_factory/README.md) for a short runnable introduction.

## Setup

In the [platform](https://platform.trajectory.ai), select your organization and create a key
under **Settings → API keys**. Give your coding agent that organization's key before live runs.

Install the SDK version used by these examples and authenticate:

```bash
pip install trajectory-sdk==0.9.20
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment. The SDK authenticates resource requests with `X-API-Key`; send that header
for direct resource API calls. The OpenAI-compatible inference routes also accept a Bearer key.

### Choose or create an agent

An agent groups related benchmarks, training history, and the serving model for a project or
use case. Keep benchmarks together when they measure capabilities you want to improve in the
same agent; a distinct project or use case is a reason to create another one.

- Use the agent explicitly selected for the project when one is provided.
- Reuse an agent when its purpose fits the project and benchmarks.
- Create an agent when no existing one fits, even if the organization has unrelated agents.

Inspect agents by name and description:

```python
from trajectory import Client

client = Client()
for candidate in client.agents.list():
    print(candidate.name, candidate.description)
```

For a new use case, create an agent with a descriptive name. For example, for the T Factory
training walkthrough:

```python
agent = client.agents.create(
    name="t-factory-cookbook",
    description="Learning task rewards and training with T Factory",
)
print(agent.agent_id)
```

Use the selected or newly created agent's name wherever `YOUR_AGENT_NAME` appears below.

## SDK guidance

The following snippets show individual SDK calls for existing integrations. For runnable,
step-by-step onboarding, start with [T Factory](examples/t_factory/README.md), then follow the
[GSM8K workflow](examples/gsm8k/README.md).

This reference trains and evaluates Qwen 3.5 4B on a small GSM8K benchmark with 64 training
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

Creation calls this ID `tid`; listed and retrieved trajectories call it `trajectory_id`.

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

Explicitly map the grader’s documented score field to `log_reward(value=...)`; do not assume
its output contains a key named `reward`.

Preserve the benchmark's scoring formula. See the [reward integration guide](examples/reward_integration.md)
for score mapping, diagnostic signals, and validation.

#### Run your benchmark and see the result

```python
from trajectory import Client
client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(model="openai/gpt-5.4-mini", messages=[{"role": "user", "content": "What is 6 × 7?"}], x_trajectory_id=tid)
reward = float(response.choices[0].message.content.strip() == "42")
client.trajectories.log_reward(tid, name="reward_accuracy", value=reward)
client.trajectories.complete(tid)
print(client.trajectories.retrieve(tid, include_steps=True, include_reward_components=True))
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
If the source gives no split, mark any partition you choose as provisional and keep it
separate from the benchmark's published comparison score.

Using the agent chosen in [Setup](#choose-or-create-an-agent), package the tasks and runtime,
upload the benchmark, and wait for its runtime image to become ready:

```python
from pathlib import Path

from trajectory import BenchmarkSpec, Client
from trajectory.lib import DockerfileBuild, start_push
from trajectory.lib.benchmark_operations import BenchmarkOperationError

client = Client()
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=tasks,
)
operation = start_push(
    client,
    benchmark,
    agent_name="YOUR_AGENT_NAME",
    root=Path("my-benchmark"),
    build_images=True,
)
print(operation.id)
try:
    result = operation.result(timeout=2700)
except BenchmarkOperationError as error:
    for failure in error.failures:
        print(failure.to_json())
    raise
bench_id = result.status.bench_id
```

Keep the operation ID to reconnect after a local timeout:

```python
from trajectory.lib import get_operation

operation = get_operation(client, "YOUR_OPERATION_ID")
print(operation.refresh())
for failure in operation.pages("failure"):
    print(failure)
```

Failures identify the stage, runtime or task, and message; build failures may include a
redacted provider-error excerpt, though full logs are not guaranteed. A local `result()`
timeout leaves the operation running. To stop it, request
`client.benchmarks.ingestion.cancel(operation.id)` and verify its state.

To revise the harness or tasks, upload the full desired task set under the same benchmark
name and agent. This creates a version; earlier versions remain available. Pin `bench_id`
for comparisons, since selecting by agent and benchmark names uses the latest version.

`push()`, `start_push()` and `start_task_diagnostic()` accept `upload_timeout` (default
3,600 seconds; `None` disables it) for each blob-store HTTP operation, not the full upload,
API calls, image build or signed-URL lifetime.

Run the complete uploader with that agent name, then save the benchmark ID:

```bash
uv run examples/gsm8k/ingest.py --agent-name "YOUR_AGENT_NAME"
```

```text
agent_name=YOUR_AGENT_NAME
bench_id=bm_<32-hex>
```

If your benchmark needs an organization secret, check which secrets are already stored
for the organization. List them to determine which missing secrets need to be uploaded
before running tasks:

```python
import os

from trajectory import Client

client = Client()
existing_secrets = {secret.name for secret in client.secrets.list()}
if "OPENAI_API_KEY" not in existing_secrets:
    client.secrets.create(
        name="OPENAI_API_KEY",
        value=os.environ["OPENAI_API_KEY"],
    )
```

`client.secrets.list()` returns names and metadata, not secret values themselves.
If a required secret is absent, upload its value before running the benchmark.
In a `TaskSpec`, `env_vars={"JUDGE_KEY": SecretRef(secret_ref="OPENAI_API_KEY")}`
injects that organization secret as the runtime variable `JUDGE_KEY`. The `env_vars`
key sets the variable name; `secret_ref` names the stored secret.

Managed task runtimes receive `TRAJECTORY_API_KEY` and `TRAJECTORY_BASE_URL` automatically;
use `Client()` inside the harness. Do not put `TRAJECTORY_*` names in `env_vars`, including
through `SecretRef`. Use `SecretRef` for additional credentials the benchmark needs.

### 3. Evaluate, train, and compare on the Trajectory Platform

Select a benchmark by `bench_id` to pin a version, or by both `agent_name` and
`benchmark_name` to use the latest version. Do not combine those selectors. `agent_id` is an
optional ownership input, not a benchmark selector.

Training and evaluation use `create`, `base_model_slug`, `parent_checkpoint_id`, and the
same `options` schema. Discover the supported settings and bounds for each mode:

```python
for resource in (client.training, client.evals):
    catalog = resource.list_options(bench_id=bench_id, base_model_slug="Qwen/Qwen3.5-4B")
    for model in catalog.models:
        print(model.base_model_slug, model.options)
```

Before packaging the harness, check `client.evals.list_models()` or
`client.training.list_models()` and the actor inference route. Inference availability alone
does not establish eval support.
Omit `options`, or pass `{}`, for defaults. Leave computed defaults unset so the server
can resolve them. Run the benchmark before training so you have a frozen baseline.
Use a one-task managed eval with execution and concurrency limits before a larger eval;
[diagnostic helpers](examples/task_diagnostics.md) do not accept run options.
Check its raw grade, effective reward, and ending. Retain the source's generation settings and
output limit when checking its native grade; a shorter cap can prevent grading and makes
the score incomparable.

```python
smoke = client.evals.create(
    bench_id=bench_id,
    base_model_slug="Qwen/Qwen3.5-4B",
    display_name="GSM8K smoke test",
    options={"disable_thinking": True, "evaluation_max_samples": 1},
)
smoke_eval_run_id = smoke.eval_run_id
```

After this passes, run the held-out baseline on the planned subset and keep its run ID.

`execution_timeout_seconds` starts when the harness runs, after queueing and sandbox startup;
a pending eval can outlast it. A failed eval can contain graded tasks without a published mean.
Inspect each trajectory's reward and rollout status before reporting a score.

Capped evals of the same benchmark and split can reuse tasks. Track task IDs across runs
when counting distinct coverage or enforcing an attempt budget.

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
rollouts per step; samples in a group share one task. Check `samples_per_instance` in the
model's option defaults.
`train_batch_size` and `samples_per_instance` size an optimizer batch, not the total run.
`max_training_samples` caps planned TRAIN samples, including rejected ones; it must cover
requested batches in whole task groups. Reduce the batch under a fixed cap if endings are
rejected. Retries add attempts, and a run without an eligible batch waits for its collection
timeout; cancel if the cap is exhausted and no batch can form. Hosted training has no
active-rollout concurrency option. Training also runs held-out eval rounds; options limit but
do not disable them. Per round, `evaluation_max_samples` caps distinct tasks, while
`evaluation_samples_per_task` sets attempts per task. Count every round and use
`evaluation_max_active_rollouts` for concurrency.
Use `client.training.rewards.list_held_out_rewards(training_run_id)` for published round
rewards and their coverage; a round without a published reward is absent. For attempts,
including pending or cancelled ones, call:

```python
path = f"/api/v1/train/{training_run_id}/steps/{step}/trajectories"
held_out = client.get(path, cast_to=dict, options={"params": {"mode": "eval"}})
```

These rounds do not appear in standalone eval-run lists. With `Client.get`, use
`options["params"]`; released SDK versions can drop an inline `?mode=eval` in the path.
Sample limits are not a dollar cap; check pricing and spending.

Poll `client.training.runs.retrieve(training_run_id)` until the run terminates.
In step summaries, `no_signal_groups` means identical rewards within a group. A
single-sample group always has that label, even when its algorithm trains on it; check
the rollout's trained-on status before calling an update skipped.
For batch membership, call the same path with
`options={"params": {"selection": "trained"}}` at the consuming step: the first update
consumes policy-step-0 rollouts at step 1.

Checkpoint evaluations inherit supported token and turn limits from the source training run
unless the eval overrides them. Compare the effective limits with the baseline, not only the
options sent to each eval. Resolve and evaluate the final checkpoint on the same held-out tasks:

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

Follow the [deployment example](examples/gsm8k/README.md#optional-deploy-and-query-the-trained-checkpoint)
to wait for readiness, query with a trajectory ID, and remove the deployment when finished.
A production deployment becomes the active target for its slug; use a test deployment when
checking a replacement before promotion.

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
through the managed actor client would use the actor model instead.
During training and evaluation, the run's `max_output_tokens_per_step` overrides a model
request's `max_tokens`. Set the run option to bound each actor reply.
In `trajectory-sdk==0.9.20`, send tool definitions through
`extra_body={"tools": tools, "tool_choice": "auto"}`; check the installed signature after upgrading.

The runtime's injected `TRAJECTORY_API_KEY` is a trajectory session key, not the organization
key used for ingestion. Do not copy it into judge or auxiliary-tool credentials. Supply those
through separately named [organization secrets](examples/inspect.md), and keep their calls
off the actor trajectory.

For a fixed judge served by Trajectory, use an organization key. An OpenAI-compatible client
uses the API origin plus `/v1`; `trajectory.Client` uses the origin alone. List available
models with `client.inference.list_models()` and use the returned ID exactly. Training-model
choices differ from standalone inference models; one model's 404 does not mean the account
lacks inference access.

Standalone inference also requires `X-Trajectory-Id`. Create and complete a separate
trajectory with the organization client, passing its `tid` in that header. A harness that
accepts only a URL and key needs an adapter for the header. Test each auxiliary call path
through its actual wrapper and confirm its ID differs from the actor's.

[Auxiliary inference example](examples/auxiliary_inference.py) covers the separate client,
trajectory and completion reasons. Adapt each harness model-call path; environment variables
alone do not add the header. The example leaves retries to its caller.

SDK responses are Pydantic models. If the harness validates responses with another library's
model class, pass `response.model_dump()` to that validator.
Bound tool output before appending it to actor messages; keep full results in artifacts when
needed. Large tool replies consume context and can leave too little room for a final answer.

If the harness retries model calls, honor `x-should-retry: false` in error response headers.
Preserve status, headers and body when translating SDK exceptions so the harness can distinguish
retryable failures from failures that should stop the task.

### Report native results

Check for unresolved execution errors before publishing a reward. **Successful grading does
not justify `ENV_DONE`.** If `client.chat.completions.create(...)` fails on every retry and
prevents the core task from completing, call
`client.trajectories.complete(tid, termination_reason="ERROR")`, even if the grader succeeds.
That grade is not a valid measure of model performance: do not publish it with `log_reward`
or use it for training. Keep any resulting grade only as diagnostic event or artifact data.

For valid executions, call `client.trajectories.log_reward(...)` with the grader's score,
including a genuine zero, then `client.trajectories.complete(...)` with the same trajectory
ID. Use the benchmark's score conversion and weighting. A recovered retry does not invalidate
an otherwise valid execution. Report grading errors as failures rather than substituting zero.

Set `termination_reason` from how the attempt actually ended: `ENV_DONE` only when it finished
normally, otherwise `ERROR`, `TIMEOUT`, `TRUNCATION` or `MISSING_SUBMIT`. The platform does not
infer the reason from the model's replies. [Ending a trajectory](examples/termination_reasons.md)
lists every reason, how to choose one, and how each affects training and evaluation. Report
exception paths with the failure reason before re-raising the original exception.

The reward is the weighted average of its components; weights default to 1. Consider weight
`0` for additional scores you want to display without affecting the reward. See the
[reward integration guide](examples/reward_integration.md#log-several-reward-components-and-grader-output)
for logging several components and the grader's output.
Use events for summaries and [artifacts](examples/diagnostic_artifacts.md) for full reports.
Finish recording them before completing the trajectory and cleaning up the environment.

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

The SDK uploads files under each Dockerfile's directory, filtered by `.dockerignore` at the
benchmark root; it does not select files by reading `COPY` statements. A root Dockerfile
therefore includes the checkout unless files are excluded. Include the files needed to build
and run that runtime, including files referenced by package metadata. Shared file paths upload
once per submission; selecting a task does not automatically exclude other tasks' files.

If the harness starts separate task containers or remote sandboxes, make its inputs available
there at the expected paths. Files in the harness image are not automatically available in
another environment. Preserve the benchmark's separation between actor-visible inputs and
private answers or hidden tests.

Check that separation through the actor's actual tools. A shell running as the image's user
can read any files that user can access, even if their paths are absent from the prompt.
Keep private grader files and oracle solutions outside that access while the actor solves.
Do not run actor-writable code with judge or organization credentials. Probe the actor's tools
with a test-only private file and fake secret, including relative paths and `file://` URLs.

For harness-managed containers, package the task Dockerfiles and build inputs where the
harness expects them. To reuse prebuilt images, [`docker save` / `docker load`](https://docs.docker.com/reference/cli/docker/image/load/)
preserves image configuration. Importing a root-filesystem archive loses settings such as the
entrypoint and user. Building a task image in an outer Dockerfile stage does not load it into
the harness's Docker daemon.

### Install and check dependencies

Install the harness's dependencies and the SDK. Where the harness provides a lockfile, use its
package manager to retain compatible versions, platform markers and dependency groups.
Resolve dependencies for the image's Python version before uploading. An orchestration tool
and its grader may require incompatible package versions; retain their separate environments
when the native harness separates them.
Preserve the original working directory and create the input, output and scratch directories
used by the entrypoint and its subprocesses. Copying the source files alone does not recreate
the runtime's filesystem layout.
Check the imports, executables and plugins used by the runtime entrypoint during the image
build. Include data files that dependencies fetch on first use when the task runs without
network access, and verify the runtime searches the installation path. Check execution and
grading in a small managed run after image readiness;
build success alone does not verify model routing, external services or task execution.

For `uv sync --locked`, the package index must match the lockfile. If the build environment
supplies a different index, pass the lockfile's index explicitly; for PyPI, use
`--default-index https://pypi.org/simple`.

Managed builds do not supply Docker BuildKit's automatic platform arguments, such as
`TARGETARCH`. For binaries that run in the build environment, detect the architecture with
`uname -m` and map it to the vendor's download names. Set cross-compilation targets explicitly.

Inspect failed ingestion items while other runtimes build. `runtime_build_failed` with
`retryable: true` does not automatically rebuild the image in that operation. Correct the
cause before submitting again. A polling timeout leaves server processing running; reconnect
to the existing operation to check its outcome.

Preserve the benchmark's separate solver and verifier deadlines. If its verifier grades
work after the solver times out, allow that grading phase to finish, but retain the timeout
termination reason. The rollout execution limit must cover setup, solving and grading.
Upload required outputs and logs before completion; the sandbox may be cleaned up afterward.

### Run native Docker environments

If your harness uses local Docker, set `env_resources=EnvResources(docker_engine=True)` on
the task. Import `EnvResources` from `trajectory.types.benchmarks.task_spec`.
The runtime must include `dockerd` on `PATH`, the Docker CLI, their operating-system dependencies,
and any plugins the harness needs. The flag starts the daemon; it does not install those tools.
For a harness that uses Compose, check the installed executables during the image build:

```dockerfile
RUN dockerd --version && docker --version && docker compose version
```

Test daemon startup inside a task runtime. Size `EnvResources.cpus` and `memory_mb` for the
work done there, including any task image builds. If the harness uses an external sandbox
service instead, supply that service's credentials through `SecretRef`.

Task sandboxes default to no network. Set each task's `env_resources` to
`EnvResources(network_mode="allowlist", allowed_hosts=[...])` or
`EnvResources(network_mode="public")` as needed. Inspect the constructed specs before upload;
image readiness does not verify network mode.

For external task or grading services, pin a version where supported. Otherwise record the
available version information and note that later runs may use different service behavior.

### Keep build contexts within service limits

Each runtime build context can contain at most 4,096 uploaded files and 3 GiB. Scope the
context to the files it needs. For many small files, you can archive them and extract them
during the image build. Exclude the unpacked copy from uploads. The archive still counts
toward the byte limit; task-specific runtimes can use smaller contexts.

Use `COPY` followed by `RUN tar`; managed builds do not support local archive extraction
with `ADD`:

```dockerfile
COPY tasks.tar.gz /tmp/tasks.tar.gz
RUN mkdir -p /opt/benchmark \
    && tar -xzf /tmp/tasks.tar.gz -C /opt/benchmark \
    && rm /tmp/tasks.tar.gz
```

## Complete native integrations

Read the integration recipe together with its full public implementation when adapting an existing harness:

- [Harvey LAB SDK integration](https://github.com/Trajectorylabs/harvey-labs/pull/11)
- [Big Finance Benchmark SDK integration](https://github.com/Trajectorylabs/big-finance-benchmark-public/pull/1)

## Beta testing and support

- **Schedule a beta testing call:** [Book a 30-minute call](https://calendly.com/d/dvxq-4dj-6pt/trajectory-beta-testing) — invite your friends!
- **Feedback and support:** [Join the Trajectory Discord](https://discord.gg/s5t2tpbNEE)

## License

Apache 2.0. See [LICENSE](LICENSE).
