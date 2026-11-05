# SWE-Bench Pro

For coding-agent setup, [install the SDK skills](../../README.md#use-with-a-coding-agent).

## Dataset

[SWE-Bench Pro](https://github.com/scaleapi/SWE-bench_Pro-os) is Scale AI's benchmark of
long-horizon software engineering tasks from real pull requests. This example uses its public
V2 release: 642 tasks from 11 repositories in Python, Go, JavaScript, and TypeScript, pinned
to commit `66f92766bba642462d4bbe5479e83f91f9211862` (`v2.0.0`).

For this training example, a stable hash of each instance ID selects 15% of each repository's
tasks for the 97-task held-out split; the other 545 tasks are for training. This is a cookbook
split, not an upstream leaderboard split.

## Harness

The model reads a pull request description, edits the repository with a bash tool, and is
graded by the task's hidden tests. For each attempt, the
[harness](runtime/swe_bench_pro_harness.py):

1. pulls the task's public `ghcr.io/scaleapi/swe-bench_pro-v2:<instance_id>` image and starts
   an offline repository container;
2. uses mini-swe-agent's tool-call protocol and upstream's
   [`mini_toolcall.yaml`](https://github.com/scaleapi/SWE-bench_Pro-os/tree/66f92766bba642462d4bbe5479e83f91f9211862/v2/tooling/configs)
   prompts, bash tool, and submission command;
3. captures the patch, applies it in a fresh container from the same image, and runs the
   unchanged `tests/test.sh` verifier with its hidden test patch;
4. records the verifier's `reward.txt` as `resolved` (1 or 0) and completes the trajectory.

The task instructions, prompts, and verifiers come from upstream; reference solutions are
excluded. The offline 50-minute agent budget and fresh-image grading remain. Commands run
with `sh -c` for at most 30 seconds. A reply without a valid bash call gets the upstream
format-error message. The managed run uses the turn, token, and time limits below instead of
upstream's $3 model-call cap.

| Ending | When |
| --- | --- |
| `ENV_DONE` | The model submitted, or sent three consecutive replies without a valid bash call. |
| `TIMEOUT` | The 50-minute agent budget ran out. |
| `TRUNCATION` | A reply or run hit an output, turn, or token limit. The cut-off reply's tool calls are not run. |
| `ERROR` | The image pull, container, or verifier failed; no reward is recorded. |

Every ending except `ERROR` grades the patch left behind. Training uses `TIMEOUT` attempts
only with `train_on_timeout`, which [`train.py`](train.py) sets because many attempts use the
full budget.

## Ingestion

You need Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/),
this cookbook checkout, and an organization API key from
**Settings → API keys** in the [platform](https://platform.trajectory.ai). The images pull
anonymously; no other secrets are needed. [Choose or create an agent](https://docs.trajectory.ai/agent),
then export your key from the repository root. If you need a new agent, run the create command:

```bash
export TRAJECTORY_API_KEY="YOUR_TRAJECTORY_API_KEY"
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="swe-bench-pro", description="SWE-Bench Pro code repair")'
```

Upload one task from each split, using your agent name if it differs:

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 1 --test-tasks 1
```

The uploader checks the pinned release against `v2/SHA256SUMS`, packages the task files
without `solution/`, waits for the runtime image, and diagnoses the uploaded tasks with
Qwen 3.6 35B A3B and the evaluation settings below. It prints the benchmark ID, task
statuses, and rewards; a zero reward is a valid grade, while a missing reward fails the
command. See the [SDK diagnostic guide](https://docs.trajectory.ai/guides/spec#upload-and-diagnose-the-full-spec).

For a larger demo, upload 40 training and 8 held-out tasks:

```bash
uv run examples/swe_bench_pro/ingest.py --agent-name swe-bench-pro --train-tasks 40 --test-tasks 8
```

This creates a new version of the benchmark. `--train-tasks` and `--test-tasks` select
the first N tasks in each stable split; omit both to upload all 642.

## Evaluation

The example uses these run settings for the agent and subsequent verifier:

```python
RUN_OPTIONS = {
    "reasoning_effort": "on",
    "max_turns_per_trajectory": 200,
    "max_output_tokens_per_turn": 16_384,
    "max_output_tokens_per_trajectory": 262_144,
    "trajectory_timeout_seconds": 9_000,
}
```

In the same Python session, evaluate the uploaded version without training using its
printed benchmark ID:

```python
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

client = Client()
links = PlatformLinks.from_client(client)
baseline = client.evals.create(
    bench_id="YOUR_BENCH_ID",
    base_model_slug="Qwen/Qwen3.6-35B-A3B",
    options=RUN_OPTIONS,
)
print("Evaluation:", baseline.eval_run_id, links.evaluation(baseline.eval_run_id))
```

An attempt can pull a 0.6–4.7 GB compressed image, work for up to 50 minutes, and then
grade the patch. The pull may take 30 minutes and the verifier up to 50, so the run limit
allows two and a half hours. Each task sandbox uses 2 CPUs and 8 GB of memory; attempts
run in parallel.

## Training

After the 40/8 upload, run the small training example:

```bash
uv run examples/swe_bench_pro/train.py --agent-name swe-bench-pro
```

The script evaluates Qwen 3.6 35B A3B on the eight test tasks, trains two steps with one
group of four attempts per step and up to four groups running at once, then evaluates the
final-step checkpoint on the same test tasks and settings. It prints run IDs, resolved
counts, and endings. Use `--num-steps`, `--train-batch-size`, and `--group-size` to scale.
Training samples until it finds a group with both resolved and unresolved attempts, up to
six hours per step. If the local command stops, the runs continue on the platform.

Measured with Qwen 3.6 35B A3B and the settings above:

| Step | Run ID | Wall time | Result |
| --- | --- | --- | --- |
| 48-task upload diagnostic | `evr_06ae68e3ca4c749c8000d36662af8860` | 53 minutes | 10 of 10 graded, 9 resolved; 4 `TIMEOUT`, 6 `ENV_DONE` |
| Baseline on 8 test tasks | `evr_06ae69aacaf3777d8000943c986c0c00` | 54 minutes | 8 of 8 graded, 7 resolved; 2 `TIMEOUT`, 6 `ENV_DONE` |
| Training, 2 steps | `trn_06ae6a7737697f6f8000c6215ea26a00` | About 72 minutes | Both steps succeeded |
| Checkpoint on the same 8 tasks | `evr_06ae6b86fe9f7e3d8000f59eeb7a4ac1` | 42 minutes | 8 of 8 graded, 5 resolved; 8 `ENV_DONE` |

All four stages used one benchmark version; baseline and checkpoint evaluation used the
same eight tasks and options. Timed-out attempts were graded. Eight held-out tasks with
one attempt each cannot measure lift from two training steps; use more tasks and
`val_trajectories_per_task` for a comparison.

## Appendix

The `swe_bench_pro_result` trajectory event retains the native grade, patch, verifier
output, image digest, and model-call count. Inspect one attempt with:

```python
from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

for event in Client().trajectories.list_events("YOUR_TRAJECTORY_ID"):
    if event.event["name"] == "swe_bench_pro_result":
        result = event.event["payload"]
        print(result["ending"], result["resolved"])
        print(result["fail_to_pass"], result["pass_to_pass"])
        print(result["patch"], result["verifier_output"])
```

The staged task files and tooling include
[Scale AI's MIT license](https://github.com/scaleapi/SWE-bench_Pro-os/blob/66f92766bba642462d4bbe5479e83f91f9211862/LICENSE).
Each task's base image contains a separate source repository, whose own license applies.
Cite Deng et al., “SWE-Bench Pro: Can AI Agents Solve Long-Horizon Software Engineering
Tasks?” ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)). Scores from this
cookbook split are not leaderboard results.
