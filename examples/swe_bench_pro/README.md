# SWE-Bench Pro

[SWE-Bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) is Scale AI's benchmark of
long-horizon software engineering tasks taken from real pull requests. This example packages
its public V2 release: 642 tasks from 11 repositories in Python, Go, JavaScript, and
TypeScript, pinned to upstream commit `66f92766bba642462d4bbe5479e83f91f9211862` (`v2.0.0`).
The model reads a pull request description, edits the repository with a bash tool, and is
graded by the task's hidden tests.

## How an attempt runs

All tasks share one runtime image with the Docker engine, this harness, and the upstream task
instructions, prompt, and verifiers. Reference solutions are left out. For one attempt, the
[harness](runtime/swe_bench_pro_harness.py):

1. pulls the task's public image, `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`;
2. starts the repository in a container with no network, so neither the internet nor the
   hidden tests are within the model's reach;
3. runs the model with mini-swe-agent's tool-call protocol and upstream's `mini_toolcall.yaml`
   from [`v2/tooling/configs`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2/tooling/configs):
   its prompts, one `bash` tool, and the `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   submission;
4. captures the patch with `git add -A && git diff --cached`, applies it to a fresh container
   from the same image, and runs the task's unchanged `tests/test.sh`, which applies the hidden
   test patch and checks its `fail_to_pass` and `pass_to_pass` tests;
5. records the verifier's `reward.txt` as the `resolved` reward (1 or 0) and completes the
   trajectory.

This follows upstream's locked protocol: an offline agent with a 50-minute budget, re-graded on
a pristine image. As in upstream's `locked_mini_swe` run, there is no step limit, each command
runs with `sh -c` for at most 30 seconds, and a reply without a valid `bash` call is dropped and
answered with the config's format-error message.

| Ending | When |
| --- | --- |
| `ENV_DONE` | The model ran the submission command, or sent three replies in a row without a valid `bash` call. |
| `TIMEOUT` | The 50-minute agent budget ran out. |
| `TRUNCATION` | A reply hit its output-token limit, so its tool calls are not run, or the attempt reached a run's turn or token limit. |
| `ERROR` | The image pull, a container, or the verifier failed. No reward is recorded. |

Every ending except `ERROR` grades the patch the model left behind, and evaluation scores each
graded attempt. Training uses `TIMEOUT` attempts only with `train_on_timeout`, which `train.py`
sets because many attempts use the full 50 minutes. The trajectory's
`swe_bench_pro_result` event keeps the native grade, the patch, and the verifier output.

Tasks are split by repository: 15% of each repository's tasks, ordered by the SHA-256 of their
instance ID, form the 97-task `test` split, and the other 545 form `train`. The split depends
only on the instance IDs, so every upload keeps the same held-out tasks.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An organization API key from **Settings → API keys** in the
  [platform](https://platform.trajectory.ai), with credits for evaluation and training.
- An agent to own the benchmark; see [choose or create an agent](https://docs.trajectory.ai/agent).
- This cookbook checkout.

No other secrets are needed: the images pull anonymously, and the model runs through Trajectory.
Export your API key, then create an agent named `swe-bench-pro` if you do not already have one.
If you use another agent, pass its name to both scripts below.

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="swe-bench-pro", description="SWE-Bench Pro code repair")'
```

## 1. Upload two tasks and check them

From the cookbook repository root:

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 1 --test-tasks 1
```

[`ingest.py`](ingest.py) checks the pinned release against its `v2/SHA256SUMS` and
packages the public task instructions and verifiers. It does not stage `solution/`.

The script uploads the selected tasks as benchmark `swe-bench-pro`, waits for the runtime image,
and prints `bench_id`. It then runs a
[benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark) on Qwen 3.6 35B A3B
with the settings `train.py` uses, and prints each task's status and reward. A reward of 0 is a
valid grade. The command exits with an error if a task fails or has no reward.

`--train-tasks` and `--test-tasks` take the first N tasks of each split, so repeated uploads
select the same tasks. Omit both to upload all 642.

## 2. Upload a demo subset, evaluate, and train

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 40 --test-tasks 8
uv run examples/swe_bench_pro/train.py --agent-name swe-bench-pro
```

The upload creates a new version of `swe-bench-pro` and diagnoses it. [`train.py`](train.py)
then:

1. evaluates Qwen 3.6 35B A3B on the version's test tasks;
2. trains for two steps of one task group with four attempts each, sampling four groups at a
   time;
3. evaluates the final checkpoint on the same test tasks.

Each evaluation prints its ID, resolved count, and endings. Use `--num-steps`,
`--train-batch-size`, and `--group-size` for a longer run. Training samples until each
step has a group with both resolved and unresolved attempts, up to six hours per step.

### Run settings

Every run passes the same settings for the 50-minute agent budget and subsequent verifier.

```python
RUN_OPTIONS = {
    "reasoning_effort": "on",
    "max_turns_per_trajectory": 200,
    "max_output_tokens_per_step": 16_384,
    "max_total_tokens_per_trajectory": 262_144,
    "execution_timeout_seconds": 9_000,
}
```

To start the runs yourself:

```python
from trajectory import Client

client = Client()
baseline = client.evals.create(
    agent_name="swe-bench-pro",
    benchmark_name="swe-bench-pro",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options=RUN_OPTIONS,
)
training = client.training.create(
    bench_id=baseline.bench_id,
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options={
        **RUN_OPTIONS,
        "num_steps": 2,
        "train_batch_size": 1,
        "group_size": 4,
        "max_active_rollouts": 16,
        "batch_collection_idle_timeout_seconds": 21_600,
        "train_on_timeout": True,
        "evaluation_enabled": False,
    },
)
```

After training succeeds, select the checkpoint with the highest `step_index` from
`client.training.checkpoints.list(training.training_run_id)`. Evaluate its `checkpoint_id`
with `parent_checkpoint_id` and the same `RUN_OPTIONS`.

If the local command stops, the runs continue; inspect them by ID as in the
[GSM8K walkthrough](../gsm8k/README.md#inspect-a-run).

## Expected scale and time

An attempt pulls a 0.6–4.7 GB compressed image, works for up to 50 minutes, and
then grades the patch. The pull may take 30 minutes and the verifier up to 50, so
`execution_timeout_seconds` allows two and a half hours. Each task sandbox uses
2 CPUs and 8 GB of memory. Attempts run in parallel.

Measured with Qwen 3.6 35B A3B and the settings above:

| Step | Run ID | Wall time | Result |
| --- | --- | --- | --- |
| 48-task upload diagnostic | `evr_06ae68e3ca4c749c8000d36662af8860` | 53 minutes | 10 of 10 graded, 9 resolved; 4 `TIMEOUT`, 6 `ENV_DONE` |
| Baseline on 8 test tasks | `evr_06ae69aacaf3777d8000943c986c0c00` | 54 minutes | 8 of 8 graded, 7 resolved; 2 `TIMEOUT`, 6 `ENV_DONE` |
| Training, 2 steps | `trn_06ae6a7737697f6f8000c6215ea26a00` | About 72 minutes | Both steps succeeded |
| Checkpoint on the same 8 tasks | `evr_06ae6b86fe9f7e3d8000f59eeb7a4ac1` | 42 minutes | 8 of 8 graded, 5 resolved; 8 `ENV_DONE` |

All four stages used one uploaded benchmark version. Baseline and checkpoint evaluation used
the same eight test tasks and options. All timed-out attempts above were graded.

Eight held-out tasks with one attempt each cannot measure a change from two training
steps; use more test tasks and `evaluation_samples_per_task` before reading a difference.

## Inspect an attempt

```python
from trajectory import Client

client = Client()
for event in client.trajectories.list_events("YOUR_TRAJECTORY_ID"):
    if event.event["name"] == "swe_bench_pro_result":
        result = event.event["payload"]
        print(result["ending"], result["resolved"])
        print(result["fail_to_pass"], result["pass_to_pass"])
        print(result["patch"])
        print(result["verifier_output"])
```

The event also records the image digest, the model call count, and whether the patch applied.

## Test the harness

From the cookbook root, run the tests without Docker or the platform:

```bash
uv run --with trajectory-sdk --with jinja2 --with pyyaml --with pytest pytest examples/swe_bench_pro
```

## License and attribution

SWE-Bench Pro's harness and tooling are MIT-licensed by Scale AI. Task content (problem
statements, patches, and tests) comes from the 11 upstream repositories and keeps their
licenses. Cite the benchmark as Deng et al., "SWE-Bench Pro: Can AI Agents Solve Long-Horizon
Software Engineering Tasks?" ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)).
Test-split scores from this example are not leaderboard results.
