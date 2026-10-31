# SWE-Bench Pro

[SWE-Bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) is Scale AI's benchmark of
long-horizon software engineering tasks taken from real pull requests. This example packages
its public V2 release: 642 tasks from 11 repositories in Python, Go, JavaScript, and
TypeScript, pinned to upstream commit `66f92766bba642462d4bbe5479e83f91f9211862` (`v2.0.0`).
The model reads a pull request description, edits the repository with a bash tool, and is
graded by the task's hidden tests.

## How an attempt runs

Each task's runtime is one shared image with the Docker engine, this harness, and the upstream
task directories without their reference solutions. For one attempt, the
[harness](runtime/swe_bench_pro_harness.py):

1. pulls the task's public image, `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>`;
2. starts the repository in a container with no network, so neither the internet nor the
   hidden tests are within the model's reach;
3. runs the model in a bash-tool loop with upstream's `mini_toolcall.yaml` prompt from
   [`v2/tooling/configs`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2/tooling/configs):
   the instruction, one `bash` tool, and the `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   submission;
4. captures the repository's `git diff`, applies it to a fresh container from the same image,
   and runs the task's unchanged `tests/test.sh`, which applies the hidden test patch and checks
   its `fail_to_pass` and `pass_to_pass` tests;
5. records the verifier's `reward.txt` as the `resolved` reward (1 or 0) and completes the
   trajectory.

This is upstream's locked protocol: an offline agent and a re-grade on a pristine image. The
agent has upstream's 50-minute budget, each command runs for at most 120 seconds, and an attempt
makes at most 100 model calls.

| Ending | When |
| --- | --- |
| `ENV_DONE` | The model ran the submission command. |
| `MAX_STEPS` | The attempt reached 100 model calls or a run limit. |
| `TIMEOUT` | The 50-minute agent budget ran out. |
| `TRUNCATION` | A reply hit its output-token limit; its tool calls are not run. |
| `ERROR` | The image pull, a container, or the verifier failed. No reward is recorded. |

Every ending except `ERROR` grades the patch the model left behind. Evaluation scores a
`TIMEOUT` by its grade and counts `MAX_STEPS` and `TRUNCATION` as unresolved; training uses
`TIMEOUT` attempts only with `train_on_timeout`, which `train.py` sets. The trajectory's
`swe_bench_pro_result` event keeps the native grade, the patch, and the verifier output.

Tasks are split by repository: 15% of each repository's tasks, ordered by the SHA-256 of their
instance ID, form the 97-task `test` split, and the other 545 form `train`. The split depends
only on the instance IDs, so every upload keeps the same held-out tasks.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- An organization API key from **Settings → API keys** in the
  [platform](https://platform.trajectory.ai), with credits for evaluation and training.
- An agent to own the benchmark; see [choose or create an agent](https://docs.trajectory.ai/agent).

No other secrets are needed: the images pull anonymously, and the model runs through Trajectory.

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="swe-bench-pro", description="SWE-Bench Pro code repair")'
```

Run the harness tests from the cookbook repository root. They need neither Docker nor the
platform:

```bash
uv run --with trajectory-sdk --with jinja2 --with pyyaml --with pytest pytest examples/swe_bench_pro
```

## 1. Upload two tasks and check them

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 1 --test-tasks 1
```

[`ingest.py`](ingest.py) reads the task list from the pinned commit, uploads the selected tasks
as benchmark `swe-bench-pro`, and waits for the runtime image. It prints `bench_id`, then runs a
[benchmark diagnostic](../task_diagnostics.md#check-an-uploaded-benchmark) on Qwen 3.6 35B A3B
and prints each task's status and reward. A reward of 0 is a valid grade. The command exits
with an error if a task fails or no reward is recorded.

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
2. trains for two steps of one task group with four attempts each, running four attempts at a
   time;
3. evaluates the final checkpoint on the same test tasks.

Each evaluation prints its ID, resolved count, and endings. Every run passes
`max_output_tokens_per_step=16384`, so the two evaluations compare like for like. Use
`--num-steps`, `--train-batch-size`, and `--group-size` for a longer run. A group whose
attempts all get the same reward has no training signal, so training keeps sampling until each
step has a group with both resolved and unresolved attempts.

If the local command stops, the runs continue; inspect them by ID as in the
[GSM8K walkthrough](../gsm8k/README.md#inspect-a-run).

## Expected scale and time

An attempt pulls a 0.6–4.7 GB compressed image, works for up to 50 minutes, and grades in a
few minutes; the verifier may take up to 50 minutes. Allow up to two hours per attempt, the
default `execution_timeout_seconds`. Each task sandbox uses 2 CPUs and 8 GB of memory.

<!-- Filled in from the production verification. -->

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

## License and attribution

SWE-Bench Pro's harness and tooling are MIT-licensed by Scale AI. Task content (problem
statements, patches, and tests) comes from the 11 upstream repositories and keeps their
licenses. Cite the benchmark as Deng et al., "SWE-Bench Pro: Can AI Agents Solve Long-Horizon
Software Engineering Tasks?" ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)).
Test-split scores from this example are not leaderboard results.
