# /// script
# dependencies = ["trajectory-sdk>=0.9.33"]
# ///
"""Evaluate, train, and compare the final checkpoint on Tau 3 Retail."""

import argparse
import time
from dataclasses import dataclass

from run_options import ROLLOUT_OPTIONS
from trajectory import Client, RateLimitError
from trajectory.types.training.training_run_response import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
# One sample per test task swings the mean by several points between identical runs.
_EVAL_OPTIONS = {"evaluation_samples_per_task": 4}


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
    baseline = evaluate(
        client,
        agent_name,
        bench_name,
        model,
        "Tau 3 Retail baseline",
        poll_seconds,
    )
    run_id = start_training(client, agent_name, bench_name, model, num_steps)
    run = wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    comparison = RewardComparison(
        baseline=baseline,
        final=evaluate(
            client,
            agent_name,
            bench_name,
            model,
            f"Tau 3 Retail {run_id} step {num_steps}",
            poll_seconds,
            parent_checkpoint_id=max(
                client.training.checkpoints.list(run_id),
                key=lambda checkpoint: checkpoint.step_index,
            ).checkpoint_id,
        ),
    )
    print(f"baseline_reward={comparison.baseline:.6f}", flush=True)
    print(f"final_reward={comparison.final:.6f}", flush=True)
    print(f"reward_delta={comparison.final - comparison.baseline:+.6f}", flush=True)
    return comparison


def start_training(
    client: Client, agent_name: str, bench_name: str, model: str, num_steps: int
) -> str:
    created = client.training.create(
        agent_name=agent_name,
        benchmark_name=bench_name,
        base_model_slug=model,
        display_name=f"Tau 3 Retail {num_steps} steps",
        options={
            **ROLLOUT_OPTIONS,
            "evaluation_enabled": False,
            "num_steps": num_steps,
            "train_batch_size": 8,
            "learning_rate": 5e-5,
            "lr_warmup_steps": 3,
        },
    )
    print(f"training_run_id={created.training_run_id}", flush=True)
    return created.training_run_id


def evaluate(
    client: Client,
    agent_name: str,
    bench_name: str,
    model: str,
    display_name: str,
    poll_seconds: float,
    parent_checkpoint_id: str | None = None,
) -> float:
    while True:
        try:
            evaluation = client.evals.create(
                agent_name=agent_name,
                benchmark_name=bench_name,
                base_model_slug=model,
                parent_checkpoint_id=parent_checkpoint_id,
                display_name=display_name,
                options={**ROLLOUT_OPTIONS, **_EVAL_OPTIONS},
            )
            break
        except RateLimitError as error:
            # An active-evaluation limit can outlast client retries.
            print(f"evaluation={display_name} waiting: {error.message}", flush=True)
            time.sleep(poll_seconds)
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
            raise RuntimeError(f"evaluation {eval_id} ended: {progress.failure}")
        time.sleep(poll_seconds)

    result = client.evals.runs.retrieve(eval_id)
    if result.reward_mean is None:
        raise RuntimeError(f"completed evaluation {eval_id} has no reward")
    return result.reward_mean


def wait_for_training(
    client: Client, run_id: str, poll_seconds: float
) -> TrainingRunResponse:
    while True:
        run = client.training.runs.retrieve(run_id)
        progress = client.training.runs.progress(run_id)
        steps = f"{progress.completed_steps}/{progress.total_steps or '?'}"
        print(f"status={run.status} steps={steps}", flush=True)
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
