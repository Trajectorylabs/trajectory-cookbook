# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Evaluate, train, and compare the final checkpoint on GSM8K."""

import argparse
import time
from dataclasses import dataclass

from trajectory import Client
from trajectory.types.training.training_run_response import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_MAX_ACTIVE_ROLLOUTS = 4
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
    baseline = _evaluate_model(
        client,
        agent_name,
        bench_name,
        model,
        "GSM8K baseline",
        poll_seconds,
    )

    created = client.training.create(
        agent_name=agent_name,
        benchmark_name=bench_name,
        base_model_slug=model,
        options={
            "disable_thinking": True,
            "num_steps": num_steps,
            "train_batch_size": 4,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
            "max_turns_per_trajectory": 1,
            "max_response_chars_per_tool_call": 128,
        },
    )
    run_id = created.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    run = _wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    comparison = RewardComparison(
        baseline=baseline,
        final=_evaluate_model(
            client,
            agent_name,
            bench_name,
            model,
            f"GSM8K {run_id} step {num_steps}",
            poll_seconds,
            parent_checkpoint_id=client.training.checkpoints.retrieve(
                run_id, num_steps
            ).checkpoint_id,
        ),
    )
    print(f"baseline_reward={comparison.baseline:.6f}", flush=True)
    print(f"final_reward={comparison.final:.6f}", flush=True)
    print(f"reward_delta={comparison.final - comparison.baseline:+.6f}", flush=True)
    return comparison


def eval_smoketest(
    client: Client,
    agent_name: str,
    bench_name: str,
    model: str = _DEFAULT_MODEL,
    poll_seconds: float = 30,
    parent_checkpoint_id: str | None = None,
) -> float:
    # Evaluate one held-out task to catch basic execution and grading errors quickly.
    evaluation = client.evals.create(
        agent_name=agent_name,
        benchmark_name=bench_name,
        base_model_slug=model,
        parent_checkpoint_id=parent_checkpoint_id,
        display_name="GSM8K smoketest",
        options={
            "disable_thinking": True,
            "evaluation_max_samples": 1,
            "evaluation_max_active_rollouts": 1,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
        },
    )
    return _wait_for_evaluation(
        client, evaluation.eval_run_id, "GSM8K smoketest", poll_seconds
    )


def _evaluate_model(
    client: Client,
    agent_name: str,
    bench_name: str,
    model: str,
    display_name: str,
    poll_seconds: float,
    parent_checkpoint_id: str | None = None,
) -> float:
    evaluation = client.evals.create(
        agent_name=agent_name,
        benchmark_name=bench_name,
        base_model_slug=model,
        parent_checkpoint_id=parent_checkpoint_id,
        display_name=display_name,
        options={
            "disable_thinking": True,
            "evaluation_max_active_rollouts": _MAX_ACTIVE_ROLLOUTS,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
        },
    )
    return _wait_for_evaluation(
        client, evaluation.eval_run_id, display_name, poll_seconds
    )


def _wait_for_evaluation(
    client: Client, eval_id: str, display_name: str, poll_seconds: float
) -> float:
    print(f"evaluation={display_name} eval_run_id={eval_id}", flush=True)

    while True:
        progress = client.evals.runs.progress(eval_id)
        print(
            f"evaluation={display_name} status={progress.status} "
            f"rollouts={progress.terminal_rollouts}/{progress.total_rollouts}",
            flush=True,
        )
        if progress.status == "completed":
            break
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(f"checkpoint evaluation failed: {progress.failure}")
        time.sleep(poll_seconds)

    result = client.evals.runs.retrieve(eval_id)
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
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--bench-name", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--poll-seconds", type=float, default=30)
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
