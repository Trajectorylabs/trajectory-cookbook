# Trajectory Cookbook

Examples for evaluating and training models with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

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
The entrypoint must call `client.trajectories.log_reward(...)` with the computed reward, then
`client.trajectories.complete(...)` with the same trajectory ID. Exiting the process does not
complete the trajectory. Putting a grader description in `TaskSpec.spec` does not execute it.
See the [Harvey LAB](examples/harvey_labs.md) and
[Big Finance Benchmark](examples/big_finance_benchmark.md) integrations for examples.

The SDK uploads the files under each Dockerfile's directory, filtered by `.dockerignore`
at the benchmark root. It does not select files by reading `COPY` statements. A Dockerfile
at the repository root therefore includes the whole checkout unless files are excluded.

Use a shared `BenchmarkSpec.runtime` when tasks need the same runtime contents. When tasks
need different files or environments, set `TaskSpec.runtime` separately. For example:

```python
from trajectory import BenchmarkSpec, TaskSpec
from trajectory.lib import DockerfileBuild

benchmark = BenchmarkSpec(
    name="coding-tasks",
    tasks=[
        TaskSpec(
            name=task_id,
            runtime=DockerfileBuild(f"runtimes/{task_id}/Dockerfile"),
            run_command=f"python /app/harness.py --task-id {task_id}",
        )
        for task_id in ("task-a", "task-b")
    ],
)
```

Each `runtimes/<task-id>/` directory must contain the files its Dockerfile needs, including
that task's inputs and grading assets. Supply shared harness code and dependencies through
a common base image or include them in each context. Preserve the native harness's directory
layout and behavior; copying the entire task collection into every runtime is not required.
The SDK uploads shared file paths once per submission, but it does not automatically separate
one task's files from another's.

If the harness launches a container, preserve its image configuration as well as its files.
Importing a root-filesystem archive does not retain the original entrypoint, user, or other
image settings. Use the original image build or transfer the image with
[`docker save` / `docker load`](https://docs.docker.com/reference/cli/docker/image/load/).

The current service limits are 4,096 uploaded files and 3 GiB per runtime build context.
For many small files, create a compressed archive and extract it during the image build.
Keep the unpacked source outside the context so it is not uploaded alongside the archive.
The archive still counts toward the byte limit; separate runtimes can use smaller contexts
containing only their required files.

```dockerfile
COPY tasks.tar.gz /tmp/tasks.tar.gz
RUN mkdir -p /opt/benchmark \
    && tar -xzf /tmp/tasks.tar.gz -C /opt/benchmark \
    && rm /tmp/tasks.tar.gz
```

Include the harness and its dependencies in the runtime. If tasks call external services,
set `env_resources=EnvResources(network_mode="public")` on the task and supply credentials
through `SecretRef`. Import `EnvResources` from `trajectory.types.benchmarks.task_spec`.
Test task execution and grading after the image builds.

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
