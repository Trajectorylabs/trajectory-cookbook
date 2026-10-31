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
3. runs the model in a bash-tool loop with upstream's `mini_toolcall.yaml` prompt from
   [`v2/tooling/configs`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2/tooling/configs):
   the instruction, one `bash` tool, and the `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
   submission;
4. captures the repository's `git diff`, applies it to a fresh container from the same image,
   and runs the task's unchanged `tests/test.sh`, which applies the hidden test patch and checks
   its `fail_to_pass` and `pass_to_pass` tests;
5. records the verifier's `reward.txt` as the `resolved` reward (1 or 0) and completes the
   trajectory.

This follows upstream's locked protocol: an offline agent with a 50-minute budget, re-graded on
a pristine image. The harness adds two limits: each command runs for at most 120 seconds, and
an attempt makes at most 100 model calls. Command output is shown in the template's JSON layout
without Jinja's HTML escaping, so quotes and angle brackets read as they appear in the files.

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
- This cookbook checkout. A checkout of the SWE-Bench Pro repository is optional.

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

From the cookbook repository root:

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 1 --test-tasks 1
```

If you already have a checkout of the SWE-Bench Pro repository, pass it with `--upstream`
instead of downloading the release. It must be at the pinned commit:

```bash
git clone https://github.com/scaleapi/SWE-bench_Pro-os
git -C SWE-bench_Pro-os checkout 66f92766bba642462d4bbe5479e83f91f9211862
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 1 --test-tasks 1 \
  --upstream SWE-bench_Pro-os
```

From inside that checkout, point `uv run` at the cookbook's script instead:

```bash
uv run PATH/TO/trajectory-cookbook/examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro \
  --train-tasks 1 --test-tasks 1 --upstream .
```

[`ingest.py`](ingest.py) downloads the pinned commit, or checks that the `--upstream` checkout's
`HEAD` is that commit. It checks every staged file against the release's `v2/SHA256SUMS`, then
builds the runtime from a temporary directory holding the Dockerfile, the harness, and one
archive of each task's `instruction.md`, `task.toml`, and `tests/` plus upstream's
`mini_toolcall.yaml`. The archive keeps the build context to three files; the task directories
alone hold over 4,000. `solution/` is never staged.

The script uploads the selected tasks as benchmark `swe-bench-pro`, waits for the runtime image,
and prints `bench_id`. It then runs a
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

Each evaluation prints its ID, resolved count, and endings. Use `--num-steps`,
`--train-batch-size`, and `--group-size` for a longer run. A group whose attempts all get the
same reward has no training signal, so training keeps sampling until each step has a group
with both resolved and unresolved attempts.

### Run settings

Every run passes the same settings, so the baseline and checkpoint scores are comparable.
They fit the harness: 100 model calls, a 50-minute agent budget, and a verifier that can take
as long again.

```python
RUN_OPTIONS = {
    "reasoning_effort": "on",
    "max_turns_per_trajectory": 100,
    "max_output_tokens_per_step": 16_384,
    "max_total_tokens_per_trajectory": 262_144,
    "execution_timeout_seconds": 7_200,
}
```

On the production runs [below](#expected-scale-and-time), an attempt used 5,000–20,000 output
tokens in total, at most about 4,000 in one reply, and up to 65,000 prompt tokens. To start the
runs yourself:

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
        "max_active_rollouts": 4,
        "train_on_timeout": True,
        "evaluation_enabled": False,
    },
)
```

After training succeeds, evaluate
`client.training.checkpoints.retrieve(training.training_run_id, 2).checkpoint_id` with
`parent_checkpoint_id` and the same `RUN_OPTIONS`.

If the local command stops, the runs continue; inspect them by ID as in the
[GSM8K walkthrough](../gsm8k/README.md#inspect-a-run).

## Expected scale and time

An attempt pulls a 0.6–4.7 GB compressed image, works for up to 50 minutes, and grades in a
few minutes; the verifier may take up to 50 minutes, so `execution_timeout_seconds` allows two
hours. Each task sandbox uses 2 CPUs and 8 GB of memory. Attempts run in parallel, so a run
takes about as long as its slowest wave of attempts.

Measured with Qwen 3.6 35B A3B and the settings above:

| Step | Wall time | Result |
| --- | --- | --- |
| Two-task upload and diagnostic | 1 minute to build, 55 minutes to diagnose | 2 of 2 graded, 1 resolved |
| 48-task demo upload and diagnostic | 1 minute to build, 55 minutes to diagnose | 10 of 10 graded, 5 resolved |
| Baseline on 8 test tasks | 56 minutes | 5 of 8 resolved; 5 `TIMEOUT`, 3 `ENV_DONE` |
| Training, 2 steps | 2 hours 50 minutes | 5 task groups sampled; 2 trained on, 2 resolved in every attempt |
| Checkpoint on the same 8 tasks | 55 minutes | 3 of 8 resolved; 2 `TIMEOUT`, 6 `ENV_DONE` |

Attempts made 25–75 model calls. Eight held-out tasks with one attempt each cannot measure a
change from two training steps; use more test tasks and `evaluation_samples_per_task` before
reading a difference. The full 97-task test split runs in about two waves at the default 50
concurrent attempts.

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
