# /// script
# dependencies = ["trajectory-sdk==0.6.14"]
# ///
"""Train a model and compare initial and final checkpoints on GSM8K."""

import argparse
import time
from dataclasses import dataclass

from trajectory import Client
from trajectory.types import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


@dataclass(frozen=True)
class RewardComparison:
    baseline: float
    final: float


def train_and_evaluate(
    client: Client,
    bench_id: str,
    model: str,
    num_steps: int,
    poll_seconds: float,
) -> RewardComparison:
    benchmark = client.benchmarks.specs.retrieve(bench_id)
    if not benchmark.tasks or {task.split for task in benchmark.tasks} != {
        "train",
        "test",
    }:
        raise ValueError("The benchmark must contain explicit train and test splits")

    created = client.training.create(
        bench_id=bench_id,
        base_model_id=model,
        training_options={
            "num_steps": num_steps,
            "max_output_tokens_per_step": 1024,
            "max_total_tokens_per_trajectory": 2048,
            "max_turns_per_trajectory": 1,
            "max_tool_calls_per_step": 0,
            "max_response_chars_per_tool_call": 128,
        },
    )
    run_id = created.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    run = _wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    comparison = RewardComparison(
        baseline=_evaluate_checkpoint(client, bench_id, run_id, model, 0, poll_seconds),
        final=_evaluate_checkpoint(
            client, bench_id, run_id, model, num_steps, poll_seconds
        ),
    )
    print(f"baseline_reward={comparison.baseline:.6f}", flush=True)
    print(f"final_reward={comparison.final:.6f}", flush=True)
    print(f"reward_delta={comparison.final - comparison.baseline:+.6f}", flush=True)
    return comparison


def _evaluate_checkpoint(
    client: Client,
    bench_id: str,
    run_id: str,
    model: str,
    step: int,
    poll_seconds: float,
) -> float:
    checkpoint = client.training.checkpoints.retrieve(run_id, step)
    evaluation = client.evals.start(
        bench_id,
        model_slug=model,
        checkpoint_id=checkpoint.checkpoint_id,
        display_name=f"GSM8K {run_id} step {step}",
    )
    eval_id = evaluation.eval_run_id
    print(f"step={step} checkpoint_id={checkpoint.checkpoint_id} eval_run_id={eval_id}")

    while True:
        progress = client.evals.runs.retrieve_progress(eval_id)
        print(
            f"eval_step={step} status={progress.status} "
            f"rollouts={progress.terminal_rollouts}/{progress.total_rollouts}",
            flush=True,
        )
        if progress.status == "completed":
            break
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(f"checkpoint evaluation failed: {progress.failure}")
        time.sleep(poll_seconds)

    result = next(
        run for run in client.evals.runs.list(bench_id) if run.eval_run_id == eval_id
    )
    if result.reward_mean is None:
        raise RuntimeError(f"completed evaluation {eval_id} has no reward")
    return result.reward_mean


def _wait_for_training(
    client: Client,
    run_id: str,
    poll_seconds: float,
) -> TrainingRunResponse:
    while True:
        run = client.training.runs.retrieve(run_id)
        progress = client.training.runs.progress(run_id)
        print(
            f"status={run.status} steps={progress.completed_steps}/{progress.total_steps or '?'}",
            flush=True,
        )
        if str(run.status) in _TERMINAL_STATUSES:
            return run
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-id", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--num-steps", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()

    train_and_evaluate(
        Client(), args.bench_id, args.model, args.num_steps, args.poll_seconds
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
