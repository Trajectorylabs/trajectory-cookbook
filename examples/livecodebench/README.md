# LiveCodeBench through the public SDK

Reuse the existing adapter from [LiveCodeBench PR #7](https://github.com/Trajectorylabs/livecodebench/pull/7),
at `d9ae7dee756c4e81d2fb37ff8baeb5670847c31f`. The adapter prepares all 400 release_v1
problems and 6,103 public/private tests. Its experimental temporal split is 294 train / 106 test;
the held-out score is not the full leaderboard score.

`public-sdk.patch` switches the adapter's model call to `client.chat.completions.create`
and updates its locked runtime SDK from 0.6.11 to 0.6.20. It also makes the absolute
launcher enter the source directory before importing the original prompts, which open
few-shot files using relative paths; a regression test covers arbitrary working directories.
It leaves the sealed source,
prompts, ten-attempt retry loop, extraction, grader, task selection, and private-file permissions
unchanged. No new harness or SDK implementation is added here.

## Prepare

Run from the cookbook root. Keep the source checkout and generated data outside the cookbook;
preparation downloads about 1.25 GB and produces about 935 MB of upload files.

```bash
git clone https://github.com/Trajectorylabs/livecodebench.git /tmp/livecodebench
git -C /tmp/livecodebench checkout d9ae7dee756c4e81d2fb37ff8baeb5670847c31f
git -C /tmp/livecodebench apply "$PWD/examples/livecodebench/public-sdk.patch"
```

In the source checkout:

```bash
cd /tmp/livecodebench
uv run --project sdk --locked python -m sdk.prepare \
  --name livecodebench-release-v1-public-sdk --output /tmp/livecodebench-prepared
uv run --project sdk --locked python -m pytest -q -c sdk/pyproject.toml sdk/tests
```

The worker tests require root in Linux and skip on macOS. Build `Dockerfile.sdk` from
the generated `package/`, not the source checkout. The upstream workflow documents how
to run the tests inside that image with networking disabled and 4 GiB / 0.5 CPU limits.
The patch adds one launcher regression to the original 19 tests.

## Ingest with mainline

From the cookbook root, use the public SDK's mainline revision checked for this example:

```bash
git clone https://github.com/Trajectorylabs/trajectory-platform.git /tmp/trajectory-public-sdk
git -C /tmp/trajectory-public-sdk checkout e6161dc143b36052604f293fdcafcd5788952900
uv run --project /tmp/trajectory-public-sdk --no-dev --env-file .env \
  python examples/livecodebench/ingest.py \
  --prepared /tmp/livecodebench-prepared \
  --idempotency-key livecodebench-release-v1-public-sdk
```

`.env` supplies `TRAJECTORY_API_KEY`; `TRAJECTORY_BASE_URL` optionally selects another deployment.
Pass `--agent-id` to select an existing agent. The uploader calls the public
`trajectory.lib.submit(..., build_images=True)` hook and waits for registration and image readiness.
It saves the operation ID and final status in `ingestion.json` beside the manifest.
Re-running with the same prepared directory reconnects using `get_operation`; if upload was
interrupted before a receipt was written, reuse the same idempotency key and unchanged package.
Use a fresh output directory and key for a different ingestion, and keep the same endpoint and
credentials when resuming.

The runtime uses the published SDK wheel; ingestion uses the pinned mainline checkout.
Mainline includes upload and operation fixes beyond the published 0.6.20 package.

The verified ingestion produced benchmark `bm_06ab348ca3797519800026f6592485ad`
through operation `iop_e974e95a5f3f1f2c24c627e3b007e4cd`. The operation succeeded
with 400 registered tasks, one ready Runloop runtime, and zero failures. A fresh SDK read
confirmed all task labels, splits, and launch commands match the prepared manifest, and
all 400 task images are ready. See [verification.json](verification.json) for the pins,
hashes, image checks, and server results. This ingestion evidence predates the hosted evaluation and training submissions described below.

## Prepared-image verification

The patched image passed all 20 tests (no skips) on Linux amd64 with networking disabled,
4 GiB of memory, and 0.5 CPU. Checks include original grading controls, credential and
private-file isolation, retry recovery/exhaustion, reward-before-completion behavior, and
launcher imports from an arbitrary directory. The launcher regression fails on the original
PR #7 launcher and passes with this patch.

All 400 restored task files match the source archive hashes; the private directory is `0700`
and its files are `0600`. The final package contains 55 files totaling 935,377,551 bytes.
These checks use synthetic grading controls, not reference solutions for all 400 problems.

## Scoring limitation

[TRA-1272](https://linear.app/trajectoryai/issue/TRA-1272) remains an admission gate:
the original grader can accept custom equality as a correct answer. Ingestion and synthetic
controls do not establish trustworthy rewards or a canonical-solution sweep. This release
does not supply original reference solutions.

## Evaluate on Nemotron Lightning

The shared [evaluation scaffold](../evaluate.py) uses
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`, verified in the public eval model
catalog. From the cookbook root, preview the request without starting a run:

```bash
uv run --env-file .env examples/evaluate.py \
  --bench-id bm_06ab348ca3797519800026f6592485ad \
  --run-key livecodebench-nemotron-lightning-baseline
```

Add another `--bench-id` to evaluate a second benchmark on the same model.
Add `--launch` to start the runs;
the script prints each run ID. Keep the same run key and options when retrying a submission;
use a new key for a new evaluation. The script exits after submission; it does not wait for scores.

By default this previews a two-task TEST smoke evaluation with at most two active rollouts
per run. Use `--max-samples 106 --max-active-rollouts 8` for all 106 LiveCodeBench TEST
tasks with one attempt per task. The scaffold enables thinking, allows up to 32,768 output
tokens per step, and sets a 7,200-second execution timeout; LiveCodeBench's original harness
still requests at most 2,000 completion tokens, so the higher platform cap does not increase
that harness limit. The public eval request has
no train/test selector: evaluating the 294 TRAIN tasks would require a separate evaluation
benchmark that marks those tasks as TEST. Do not report the 106-task result as the full
400-task leaderboard score.

Monitor and retrieve rewards through the existing SDK resources:

```python
from trajectory import Client

client = Client()
progress = client.evals.runs.retrieve_progress(eval_run_id)
print(progress.status, progress.terminal_rollouts, progress.total_rollouts)
if progress.status == "completed":
    for run in client.evals.runs.list(bench_id):
        if run.eval_run_id == eval_run_id:
            print(run.reward_mean)
elif progress.status in {"failed", "cancelled"}:
    print(progress.failure)
```

The LiveCodeBench scoring limitation above also applies to these evaluations.

## Train Nemotron Lightning

```bash
uv run --env-file .env examples/livecodebench/train.py
```

This reproduces the submitted request for benchmark `bm_06ab348ca3797519800026f6592485ad`:
40 optimizer steps, 32 task groups per step, eight samples per group (256 rollouts per
step), thinking enabled, and a 32,768-token output cap. The benchmark has 294 TRAIN
and 106 TEST tasks. BigCodeBench has no original TRAIN split and is not trained here.

The script intentionally reuses the original idempotency key. Re-running it reconnects
to the same submission; change the benchmark and key in the script for a new run.
It saves the response under `.work/nemotron-lightning-training-v1.json`.

Submitted training: `trn_06ab35aae80f716d80007c99c90eb273`, XID `1057610`.
See [run-status.json](run-status.json) for a timestamped status snapshot of training and
both full TEST evaluations. Submission is not evidence of successful training or reward
improvement. At launch, the LiveCodeBench evaluation reported 65 terminal rollouts but
no recorded rewards; that observation requires investigation rather than treating missing
rewards as zero or claiming a successful baseline.
