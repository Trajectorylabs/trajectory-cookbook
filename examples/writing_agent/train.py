# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.10.1"]
# ///
"""Evaluate the base writer, train a fixed number of steps, and evaluate its checkpoint."""

import argparse
import json
import time
from pathlib import Path

from trajectory import Client
from trajectory.lib.platform_links import PlatformLinks

OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 512,
    "max_turns_per_trajectory": 1,
    "trajectory_timeout_seconds": 300,
}


def validate_calibration(report: dict, benchmark_description: str) -> None:
    contract = json.loads(benchmark_description)
    if report["author_confirmed"] is not True:
        raise ValueError(
            "Author-confirmed judge calibration is required before training"
        )
    for key in ("scorer_sha256", "voice_sha256", "judge_model", "judge_provider"):
        if report[key] != contract[key]:
            raise ValueError(f"Calibration does not match benchmark {key}")
    pairs = report["pairs"]
    labels = {"as_is", "minor_edit", "major_edit", "reject"}
    if len(pairs) < 20 or len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("Calibration needs at least 20 distinct author-rated pairs")
    if any(
        pair["author_label"] not in labels or pair["judge_label"] not in labels
        for pair in pairs
    ):
        raise ValueError("Unknown calibration label")
    acceptable = sum(pair["author_label"] in {"as_is", "minor_edit"} for pair in pairs)
    if acceptable < 5 or len(pairs) - acceptable < 5:
        raise ValueError(
            "Calibration needs both acceptable and rejected/major-edit examples"
        )
    agreement = sum(
        pair["author_label"] == pair["judge_label"] for pair in pairs
    ) / len(pairs)
    if agreement < 0.8:
        raise ValueError("Judge agreement is below the recipe's 80% pilot threshold")


def wait_evaluation(client: Client, run_id: str, timeout: float) -> float:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        progress = client.evals.runs.progress(run_id)
        print(
            f"eval_run_id={run_id} status={progress.status} "
            f"terminal={progress.terminal_rollouts}/{progress.total_rollouts}",
            flush=True,
        )
        if progress.status == "completed":
            if (
                progress.live.failed_rollouts
                or progress.live.cancelled_rollouts
                or progress.live.scored_rollouts != progress.total_rollouts
            ):
                raise RuntimeError(
                    f"Evaluation {run_id} has failed, cancelled or ungraded rollouts; inspect it before training"
                )
            result = client.evals.runs.retrieve(run_id)
            if result.reward_mean is None:
                raise RuntimeError(f"Evaluation {run_id} completed without a reward")
            print(
                f"eval_run_id={run_id} scored={progress.live.scored_rollouts} "
                f"failed={progress.live.failed_rollouts} "
                f"cancelled={progress.live.cancelled_rollouts}",
                flush=True,
            )
            return result.reward_mean
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Evaluation {run_id} ended {progress.status}: {progress.failure}"
            )
        time.sleep(5)
    raise TimeoutError(
        f"Evaluation {run_id} is still remote; inspect it before starting another"
    )


def train_and_evaluate(
    client: Client,
    bench_id: str,
    model: str,
    steps: int,
    timeout: float,
    calibration: dict | None = None,
) -> None:
    if steps < 1 or timeout <= 0:
        raise ValueError("steps and timeout must be positive")
    if calibration is None:
        raise ValueError(
            "Author-confirmed judge calibration is required before training"
        )
    benchmark = client.benchmarks.specs.retrieve(bench_id, include_tasks=True)
    if not benchmark.tasks or {task.split for task in benchmark.tasks} != {
        "train",
        "test",
    }:
        raise ValueError("Need a benchmark with explicit train/test splits")
    metadata = client.benchmarks.retrieve(bench_id)
    validate_calibration(calibration, metadata.description)
    test_count = sum(task.split == "test" for task in benchmark.tasks)
    links = PlatformLinks.from_client(client)
    evaluation_options = {
        **OPTIONS,
        "val_num_tasks": test_count,
        "val_max_concurrent_trajectories": 8,
    }
    baseline = client.evals.create(
        bench_id=bench_id,
        base_model_slug=model,
        display_name="Writing baseline",
        options=evaluation_options,
    )
    print(f"baseline={links.evaluation(baseline.eval_run_id)}", flush=True)
    baseline_reward = wait_evaluation(client, baseline.eval_run_id, timeout)
    training = client.training.create(
        bench_id=bench_id,
        base_model_slug=model,
        display_name="Writing training",
        options={
            **OPTIONS,
            "num_steps": steps,
            "batch_size": 4,
            "group_size": 4,
            "max_concurrent_trajectories": 16,
            "batch_timeout_seconds": 600,
            "val_enabled": False,
        },
    )
    run_id = training.training_run_id
    print(f"training={links.training(run_id)}", flush=True)
    deadline = time.monotonic() + timeout
    succeeded = False
    while time.monotonic() < deadline:
        run = client.training.runs.progress(run_id)
        print(
            f"training_run_id={run_id} status={run.status} "
            f"steps={run.completed_steps}/{run.total_steps}",
            flush=True,
        )
        if run.status == "succeeded":
            succeeded = True
            break
        if run.status in {"failed", "cancelled"}:
            raise RuntimeError(f"Training {run_id} ended {run.status}: {run.failure}")
        time.sleep(5)
    if not succeeded:
        raise TimeoutError(
            f"Training {run_id} is still remote; inspect it before starting another"
        )
    checkpoint = client.training.checkpoints.retrieve(run_id, steps)
    final = client.evals.create(
        bench_id=bench_id,
        base_model_slug=model,
        parent_checkpoint_id=checkpoint.checkpoint_id,
        display_name="Writing trained checkpoint",
        options=evaluation_options,
    )
    print(f"trained_evaluation={links.evaluation(final.eval_run_id)}", flush=True)
    final_reward = wait_evaluation(client, final.eval_run_id, timeout)
    print(
        f"baseline_reward={baseline_reward:.4f} trained_reward={final_reward:.4f}",
        flush=True,
    )
    print(f"judge_reward_delta={final_reward - baseline_reward:+.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=7200)
    args = parser.parse_args()
    train_and_evaluate(
        Client(),
        args.bench_id,
        args.model,
        args.num_steps,
        args.timeout_seconds,
        json.loads(args.calibration.read_text()),
    )


if __name__ == "__main__":
    main()
