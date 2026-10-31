# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Evaluate, train, and compare the final checkpoint on GSM8K."""

import argparse
import time
from dataclasses import dataclass

from trajectory import Client
from trajectory.types.training.training_run_response import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
# Enough concurrent attempts for one training batch.
_MAX_ACTIVE_ROLLOUTS = 32
_MAX_OUTPUT_TOKENS = 2_048


@dataclass(frozen=True)
class RewardComparison:
    baseline: float
    final: float


def train_and_evaluate(
    client: Client,
    agent_name: str,
    bench_name: str,
    model: str,
    num_steps: int,
    poll_seconds: float,
) -> RewardComparison:
    eval_options = {
        "reasoning_effort": "off",
        "evaluation_max_active_rollouts": _MAX_ACTIVE_ROLLOUTS,
        "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
    }
    baseline_run = client.evals.create(
        agent_name=agent_name,
        benchmark_name=bench_name,
        base_model_slug=model,
        display_name="GSM8K baseline",
        options=eval_options,
    )
    bench_id = baseline_run.bench_id
    print(f"comparison_bench_id={bench_id}", flush=True)
    # Training does not use the baseline score, so start it without waiting.
    created = client.training.create(
        bench_id=bench_id,
        base_model_slug=model,
        options={
            "reasoning_effort": "off",
            "num_steps": num_steps,
            "train_batch_size": 4,
            "max_active_rollouts": _MAX_ACTIVE_ROLLOUTS,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
        },
    )
    run_id = created.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    try:
        baseline = _wait_for_evaluation(
            client, baseline_run.eval_run_id, "GSM8K baseline", poll_seconds
        )
    except BaseException:
        client.training.runs.cancel(run_id)
        raise
    run = _wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    checkpoint = client.training.checkpoints.retrieve(run_id, num_steps)
    final_name = f"GSM8K {run_id} step {num_steps}"
    final_run = client.evals.create(
        bench_id=bench_id,
        base_model_slug=model,
        parent_checkpoint_id=checkpoint.checkpoint_id,
        display_name=final_name,
        options=eval_options,
    )
    comparison = RewardComparison(
        baseline=baseline,
        final=_wait_for_evaluation(
            client, final_run.eval_run_id, final_name, poll_seconds
        ),
    )
    print(f"baseline_reward={comparison.baseline:.6f}", flush=True)
    print(f"final_reward={comparison.final:.6f}", flush=True)
    print(f"reward_delta={comparison.final - comparison.baseline:+.6f}", flush=True)
    return comparison


def eval_smoketest(
    client: Client,
    bench_id: str,
    model: str = _DEFAULT_MODEL,
    poll_seconds: float = 5,
    parent_checkpoint_id: str | None = None,
) -> float:
    # Evaluate one held-out task to catch basic execution and grading errors quickly.
    evaluation = client.evals.create(
        bench_id=bench_id,
        base_model_slug=model,
        parent_checkpoint_id=parent_checkpoint_id,
        display_name="GSM8K smoketest",
        options={
            "reasoning_effort": "off",
            "evaluation_max_samples": 1,
            "evaluation_max_active_rollouts": 1,
            "execution_timeout_seconds": 600,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
        },
    )
    return _wait_for_evaluation(
        client, evaluation.eval_run_id, "GSM8K smoketest", poll_seconds
    )


def _wait_for_evaluation(
    client: Client, eval_id: str, display_name: str, poll_seconds: float
) -> float:
    print(f"evaluation={display_name} eval_run_id={eval_id}", flush=True)

    last = None
    while True:
        progress = client.evals.runs.progress(eval_id)
        line = (
            f"evaluation={display_name} status={progress.status} "
            f"rollouts={progress.terminal_rollouts}/{progress.total_rollouts}"
        )
        if line != last:
            print(line, flush=True)
            last = line
        if progress.status == "completed":
            break
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(f"checkpoint evaluation failed: {progress.failure}")
        time.sleep(poll_seconds)

    trajectories = list(client.trajectories.list(eval_run_id=eval_id))
    graded = 0
    failed = 0
    for trajectory in trajectories:
        failed += (
            trajectory.rollout is not None and trajectory.rollout.status == "FAILED"
        )
        detail = client.trajectories.retrieve(
            trajectory.trajectory_id, include_reward_components=True
        )
        graded += any(
            component.name == "reward_accuracy"
            for component in detail.reward_components or []
        )
    print(
        f"evaluation={display_name} attempted={len(trajectories)} "
        f"graded={graded} failed={failed}",
        flush=True,
    )
    if (
        not trajectories
        or graded != len(trajectories)
        or graded != progress.total_rollouts
        or failed
    ):
        raise RuntimeError(
            f"evaluation {eval_id}: {graded}/{progress.total_rollouts} graded, "
            f"{failed} failed; inspect its trajectories before continuing"
        )

    result = client.evals.runs.retrieve(eval_id)
    if result.reward_mean is None:
        raise RuntimeError(f"completed evaluation {eval_id} has no reward")
    return result.reward_mean


def _wait_for_training(
    client: Client,
    run_id: str,
    poll_seconds: float,
) -> TrainingRunResponse:
    last = None
    while True:
        run = client.training.runs.retrieve(run_id)
        progress = client.training.runs.progress(run_id)
        line = f"status={run.status} steps={progress.completed_steps}/{progress.total_steps or '?'}"
        if line != last:
            print(line, flush=True)
            last = line
        if str(run.status) in _TERMINAL_STATUSES:
            return run
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()

    train_and_evaluate(
        Client(max_retries=20),
        args.agent_name,
        args.bench_name,
        args.model,
        args.num_steps,
        args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
