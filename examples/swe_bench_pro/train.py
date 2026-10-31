# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.33"]
# ///
"""Evaluate the test tasks, train briefly, and evaluate the final checkpoint."""

import argparse
import time
from collections import Counter

from trajectory import Client

MODEL = "Qwen/Qwen3.6-35B-A3B"
# Shared by every run so the baseline and checkpoint scores are comparable.
RUN_OPTIONS = {
    "reasoning_effort": "on",
    # The 50-minute agent budget ends an attempt well before these caps.
    "max_turns_per_trajectory": 200,
    "max_output_tokens_per_step": 16_384,
    "max_total_tokens_per_trajectory": 262_144,
    # Image pull (up to 30 minutes), the 50-minute agent budget, and the verifier's 50 minutes.
    "execution_timeout_seconds": 9_000,
}


def wait_for_evaluation(client: Client, eval_run_id: str) -> None:
    print(f"eval_run_id={eval_run_id}", flush=True)
    while True:
        progress = client.evals.runs.progress(eval_run_id)
        print(
            f"status={progress.status} "
            f"rollouts={progress.terminal_rollouts}/{progress.total_rollouts}",
            flush=True,
        )
        if progress.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(60)

    endings, resolved, graded = Counter(), 0, 0
    for item in client.trajectories.list(eval_run_id=eval_run_id):
        detail = client.trajectories.retrieve(
            item.trajectory_id, include_reward_components=True
        )
        endings[detail.termination_reason] += 1
        grades = [
            c.value for c in detail.reward_components or [] if c.name == "resolved"
        ]
        graded += bool(grades)
        resolved += grades == [1.0]
    run = client.evals.runs.retrieve(eval_run_id)
    print(
        f"status={progress.status} resolved={resolved}/{progress.total_rollouts} "
        f"graded={graded} reward_mean={run.reward_mean} endings={dict(endings)}",
        flush=True,
    )
    if progress.status != "completed":
        raise RuntimeError(f"Evaluation {eval_run_id} ended {progress.status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--bench-name", default="swe-bench-pro")
    parser.add_argument("--num-steps", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    args = parser.parse_args()
    client = Client(max_retries=20)

    baseline = client.evals.create(
        agent_name=args.agent_name,
        benchmark_name=args.bench_name,
        base_model_slug=MODEL,
        display_name="SWE-Bench Pro baseline",
        options=RUN_OPTIONS,
    )
    bench_id = baseline.bench_id
    print(f"bench_id={bench_id}", flush=True)
    wait_for_evaluation(client, baseline.eval_run_id)

    training = client.training.create(
        bench_id=bench_id,
        base_model_slug=MODEL,
        display_name="SWE-Bench Pro training",
        options={
            **RUN_OPTIONS,
            "num_steps": args.num_steps,
            "train_batch_size": args.train_batch_size,
            "group_size": args.group_size,
            # Sample four groups at a time; many groups resolve every attempt or none.
            "max_active_rollouts": 4 * args.train_batch_size * args.group_size,
            "batch_collection_idle_timeout_seconds": 21_600,
            # An attempt stopped at the 50-minute agent budget still has a graded patch.
            "train_on_timeout": True,
            "evaluation_enabled": False,
        },
    )
    run_id = training.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    while True:
        run = client.training.runs.retrieve(run_id)
        progress = client.training.runs.progress(run_id)
        print(
            f"status={run.status} steps={progress.completed_steps}/{progress.total_steps}",
            flush=True,
        )
        if run.status in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(60)
    if run.status != "succeeded":
        raise RuntimeError(f"Training {run_id} ended {run.status}: {run.failure}")

    checkpoint = client.training.checkpoints.retrieve(run_id, args.num_steps)
    final = client.evals.create(
        bench_id=bench_id,
        base_model_slug=MODEL,
        parent_checkpoint_id=checkpoint.checkpoint_id,
        display_name=f"SWE-Bench Pro step {args.num_steps}",
        options=RUN_OPTIONS,
    )
    wait_for_evaluation(client, final.eval_run_id)


if __name__ == "__main__":
    main()
