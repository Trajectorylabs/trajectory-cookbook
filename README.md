# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

Before starting your coding agent, copy the [Trajectory skill](skills/trajectory/SKILL.md) to
`<benchmark-repo>/.agents/skills/trajectory/SKILL.md` for Codex or
`<benchmark-repo>/.claude/skills/trajectory/SKILL.md` for Claude Code. Ask the agent to use it.
Keep the skill and examples from the same cookbook commit when pinning a workflow.

Runnable examples and runtime images pin `trajectory-sdk==0.9.23`.
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

Supporting Guides:

- [Task Upload Validation](examples/task_diagnostics.md): validate one task or inspect a few uploaded tasks.
- [Diagnostic artifacts](examples/diagnostic_artifacts.md): keep full reports and logs with a trajectory.
- [Ending a trajectory](examples/termination_reasons.md): choose the termination reason and see how each one is scored.
- [Runtime packaging](#package-a-benchmark-runtime): files, dependencies, secrets, and nested containers.
- [Production traces](examples/production_traces.md): capture application requests under an agent (private beta).

The sections below cover SDK choices for adapting your own benchmark.

## Setup

In the [platform](https://platform.trajectory.ai), select your organization and create a key
under **Settings → API keys**. Give your coding agent that organization's key before live runs.

Install the SDK version used by these examples and authenticate:

```bash
pip install trajectory-sdk==0.9.23
export TRAJECTORY_API_KEY="..."
```

The SDK connects to `https://api.trajectory.ai` by default. Set `TRAJECTORY_BASE_URL` to use
another deployment. The SDK authenticates resource requests with `X-API-Key`; send that header
for direct resource API calls. The OpenAI-compatible inference routes also accept a Bearer key.

### Choose or create an agent

An agent groups related benchmarks, training history, and serving models. Use the agent
selected for your project, reuse one with the same purpose, or create one for a new use case.

Confirm the selected organization, then inspect agents by name and description:

```python
from trajectory import Client

client = Client()
print(client.organizations.retrieve_current()["organization_id"])
for candidate in client.agents.list():
    print(candidate.name, candidate.description)
```

Generated SDK list calls such as `agents.list()` and `trajectories.list()` return iterable
pages; use `.items` for the current page, not `.data`.

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

## SDK workflow reference

The [GSM8K walkthrough](examples/gsm8k/README.md) runs one task through validation,
evaluation, training, checkpoint comparison, and optional deployment. The notes below cover
choices when adapting that flow to another harness.

### Record a task

Create one trajectory per task. Pass its ID to every actor model call; log the native grade
and complete the same trajectory after grading. The create response calls the ID `tid`;
retrieved trajectories call it `trajectory_id`.

```python
from trajectory import Client

client = Client()
tid = client.trajectories.create().tid
response = client.chat.completions.create(
    model="openai/gpt-5.4-mini",
    messages=[{"role": "user", "content": prompt}],
    x_trajectory_id=tid,
)
client.trajectories.log_reward(tid, name="reward_accuracy", value=grade)
client.trajectories.complete(tid)
```

Map `grade` from the benchmark's documented score field; it need not be named `reward`.
Keep full reports in [artifacts](examples/diagnostic_artifacts.md) before completion.
For errors, retries, components, and non-normal endings, follow
[reward integration](examples/reward_integration.md) and
[ending a trajectory](examples/termination_reasons.md).

### Register a benchmark

Give each `TaskSpec` the source task's stable `name`, an executable `run_command`, and
its `split`. The retrieved task exposes `name` as `label` and has a generated `task_id`;
omit `TaskSpec.id` for new tasks. Without an explicit split, ingestion assigns roughly 15%
to TEST and the rest to TRAIN. Set every split to `test` for evaluation-only data. If the
source has no split, mark your partition provisional and do not compare it with the
benchmark's published score.

Use [task diagnostics](examples/task_diagnostics.md) before the full upload. Then register
the full desired task set under the same agent and benchmark name:

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
    client, benchmark, agent_name="YOUR_AGENT_NAME",
    root=Path("my-benchmark"), build_images=True,
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

A local timeout leaves the operation running. Reconnect with
`get_operation(client, operation_id)`, then inspect `refresh()` and `pages("failure")`
before retrying. Failures identify the stage and affected runtime or task; full build logs
may be unavailable. Cancel through `client.benchmarks.ingestion.cancel(operation_id)`
and verify its state. Reuploading under the same agent and benchmark name creates a new
version; pin `bench_id` for comparisons. `start_push`, `push`, and
`start_task_diagnostic` accept `upload_timeout` for each blob upload, not the whole
operation.

The runtime receives `TRAJECTORY_API_KEY` and `TRAJECTORY_BASE_URL` automatically.
Task variables cannot start with `TRAJECTORY_` or use `MODEL_ENDPOINT_ID`,
`MODEL_ENDPOINT_ACCESS_TOKEN`, or `MODEL_ENDPOINT_URL`. For other credentials, inspect
`client.secrets.list()`, upload missing values with `client.secrets.create()`, and
inject them with `TaskSpec.env_vars={"JUDGE_KEY": SecretRef(secret_ref="STORED_KEY")}`.
The environment variable name and stored secret name can differ.

### Evaluate, train, and compare

Use `bench_id` to select a specific version, or both `agent_name` and
`benchmark_name` for the latest version. Do not mix those selectors. `agent_id`
sets ownership; it does not select a benchmark. Check `evals.list_models()` and
`training.list_models()` before packaging: inference availability does not establish
run support. Use `list_options(bench_id=..., base_model_slug=...)` for each mode's
supported options and bounds; omit computed defaults.

Run a one-task managed eval with limits before a larger held-out baseline. Diagnostics
do not accept eval options. Preserve the source's output limit and generation settings
when checking its native grade. A failed eval can contain graded tasks without a
published mean; inspect attempts before interpreting the aggregate.

`train_batch_size` counts task groups per optimizer step; `samples_per_instance`
sets model samples per group. Training also runs held-out eval rounds;
`evaluation_max_samples` and `evaluation_samples_per_task` apply per round, not to
the entire run. Use `max_active_rollouts` to limit concurrent training rollouts.

Read `client.training.rewards.list_held_out_rewards(run_id)` for published rounds.
For an unpublished round, retrieve
`/api/v1/train/{run_id}/steps/{step}/trajectories` with
`options={"params": {"mode": "eval"}}`; SDK 0.9.23 can drop an inline query string when
pagination adds a cursor.
`used_groups` counts only fully graded groups with reward contrast; a no-signal or
incomplete group can still enter the optimizer batch. Check `in_training_batch` at
`/api/v1/train/{run_id}/steps/{step}/groups` (policy step 0 is consumed at step 1).
`no_signal_groups` means equal rewards within a fully graded group.

Checkpoint evals inherit supported token and turn limits from training unless
overridden. Compare effective limits and identical held-out task IDs with the baseline.
Capped evals on the same version and split select the same tasks; a new version may
change the subset. To compare base models, evaluate each on the same `bench_id` and task IDs
with common supported settings; compare graded counts and per-task scores before the mean.
For spend by run, use platform Billing (`/<org-key>/billing`),
which shows the last 30 days. See the
[GSM8K walkthrough](examples/gsm8k/README.md#5-train-on-one-task-and-evaluate-the-checkpoint)
for runnable calls and [deployment](examples/gsm8k/README.md#optional-deploy-and-query-the-trained-checkpoint)
for serving a checkpoint.

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
In `trajectory-sdk==0.9.23`, send tool definitions through
`extra_body={"tools": tools, "tool_choice": "auto"}`; check the installed signature after upgrading.

The runtime's injected `TRAJECTORY_API_KEY` is a trajectory session key, not the organization
key used for ingestion. Do not copy it into judge or auxiliary-tool credentials. Supply those
through separately named [organization secrets](examples/inspect.md), and keep their calls
off the actor trajectory.

For a fixed judge served by Trajectory, use an organization key and the managed runtime's
injected `TRAJECTORY_BASE_URL`; the uploader's `localhost` points at the sandbox instead.
An OpenAI-compatible client uses the API origin plus `/v1`; `trajectory.Client` uses the origin
alone. List models with `client.inference.list_models()` and use the returned ID exactly.
Training-model choices differ from standalone inference models; one model's 404 does not mean
the account lacks inference access.

Standalone inference also requires `X-Trajectory-Id`. Create and complete a separate
trajectory with the organization client, passing its `tid` in that header. A harness that
accepts only a URL and key needs an adapter for the header. Test each auxiliary call path
through its actual wrapper and confirm its ID differs from the actor's.

Use the [capture pattern](#record-a-task) with that separate client and the auxiliary
request's own completion reason. Environment variables alone do not add the trajectory header;
adapt the harness's model-call wrapper. Let the harness own retries when it already retries calls.

SDK responses are Pydantic models. If the harness validates responses with another library's
model class, pass `response.model_dump()` to that validator.
Validate tool-call arguments before executing tools. Follow the source harness's retry or error
rule for malformed calls; a broken call is not a graded answer.
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

The SDK uploads files under each Dockerfile's directory; `COPY` paths are relative to that
directory. `.dockerignore` at the benchmark root filters uploads. A root Dockerfile
therefore includes the checkout unless files are excluded. Include the files needed to build
and run that runtime, including files referenced by package metadata. Shared file paths upload
once per submission; selecting a task does not automatically exclude other tasks' files.

Smoke-test the source grader in the final image, including its runtime data. Without a local
container engine, check files and tools after Dockerfile `COPY`, then run a managed one-task
validation. Resolve missing-resource warnings even when a sample grade returns; a ready image
alone does not show that a task can be graded.

If the harness starts separate task containers or remote sandboxes, make its inputs available
there at the expected paths. Files in the harness image are not automatically available in
another environment. Preserve the benchmark's separation between actor-visible inputs and
private answers or hidden tests.

Check that separation through the actor's actual tools: hiding paths from the prompt does
not prevent a shell from reading accessible files. Keep private grader files, solutions,
and judge or organization credentials outside actor-writable code. Preserve the source's
solver/verifier boundary; probe it with a test-only private file and fake secret.

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
Preserve the original working directory and create the input, output, scratch, and verifier
result directories used by the entrypoint and its subprocesses. Copying source files alone
does not recreate the runtime's filesystem layout.
Check the imports, executables and plugins used by the runtime entrypoint during the image
build. Include data files that dependencies fetch on first use when the task runs without
network access, and verify the runtime searches the installation path. Check execution and
grading in a small managed run after image readiness;
build success alone does not verify model routing, external services or task execution.

For `uv sync --locked`, the package index must match the lockfile. If the build environment
supplies a different index, pass the lockfile's index explicitly; for PyPI, use
`--default-index https://pypi.org/simple`.

Do not rely on BuildKit's automatic platform arguments, such as `TARGETARCH`, in managed
builds. For binaries that run in the build environment, detect the architecture with
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
The managed readiness probe runs `docker info`, even if your harness uses the Engine API directly.
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

Extract archives explicitly during the build:

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
