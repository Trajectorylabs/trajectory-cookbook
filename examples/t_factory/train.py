# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Evaluate, train, and compare Qwen 3.5 4B on the prompted T-starting-word task."""

import argparse
import time
from dataclasses import dataclass

from trajectory import Client
from trajectory.types.training.training_run_response import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_TEST_TASKS = 64
# One batch: four task groups of eight attempts, the default group size.
_MAX_ACTIVE_ROLLOUTS = 32
_MAX_OUTPUT_TOKENS = 2_048


@dataclass(frozen=True)
class Evaluation:
    eval_run_id: str
    reward: float


def train_and_evaluate(
    client: Client,
    bench_id: str,
    model: str,
    num_steps: int,
    poll_seconds: float,
) -> tuple[Evaluation, Evaluation, str]:
    # Spec detail omits tasks unless asked.
    benchmark = client.benchmarks.specs.retrieve(bench_id, include_tasks=True)
    if not benchmark.tasks or {task.split for task in benchmark.tasks} != {
        "train",
        "test",
    }:
        raise ValueError("The benchmark must contain explicit train and test splits")

    baseline_id = _start_evaluation(client, bench_id, model, "T Factory baseline")
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
            "max_turns_per_trajectory": 1,
        },
    )
    run_id = created.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    try:
        baseline = _wait_for_evaluation(
            client, baseline_id, "T Factory baseline", poll_seconds
        )
    except BaseException:
        client.training.runs.cancel(run_id)
        raise
    run = _wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    checkpoint = client.training.checkpoints.retrieve(run_id, num_steps)
    final = _evaluate_model(
        client,
        bench_id,
        model,
        f"T Factory {run_id} step {num_steps}",
        poll_seconds,
        parent_checkpoint_id=checkpoint.checkpoint_id,
    )
    print(f"baseline_reward={baseline.reward:.6f}", flush=True)
    print(f"final_reward={final.reward:.6f}", flush=True)
    print(f"reward_delta={final.reward - baseline.reward:+.6f}", flush=True)
    _print_task_results(
        client,
        baseline.eval_run_id,
        final.eval_run_id,
    )
    return baseline, final, run_id


def _evaluate_model(
    client: Client,
    bench_id: str,
    model: str,
    display_name: str,
    poll_seconds: float,
    parent_checkpoint_id: str | None = None,
) -> Evaluation:
    eval_id = _start_evaluation(
        client, bench_id, model, display_name, parent_checkpoint_id
    )
    return _wait_for_evaluation(client, eval_id, display_name, poll_seconds)


def _start_evaluation(
    client: Client,
    bench_id: str,
    model: str,
    display_name: str,
    parent_checkpoint_id: str | None = None,
) -> str:
    evaluation = client.evals.create(
        bench_id=bench_id,
        base_model_slug=model,
        parent_checkpoint_id=parent_checkpoint_id,
        display_name=display_name,
        options={
            "reasoning_effort": "off",
            "evaluation_max_samples": _TEST_TASKS,
            # Run every held-out task at once.
            "evaluation_max_active_rollouts": _TEST_TASKS,
            "max_output_tokens_per_step": _MAX_OUTPUT_TOKENS,
        },
    )
    eval_id = evaluation.eval_run_id
    print(f"evaluation={display_name} eval_run_id={eval_id}", flush=True)
    return eval_id


def _wait_for_evaluation(
    client: Client, eval_id: str, display_name: str, poll_seconds: float
) -> Evaluation:
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
            raise RuntimeError(f"evaluation failed: {progress.failure}")
        time.sleep(poll_seconds)

    result = client.evals.runs.retrieve(eval_id)
    if result.reward_mean is None:
        raise RuntimeError(f"completed evaluation {eval_id} has no reward")
    return Evaluation(eval_run_id=eval_id, reward=result.reward_mean)


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


def _print_task_results(
    client: Client,
    baseline_eval_id: str,
    final_eval_id: str,
) -> None:
    baseline = {
        reward.task_id: reward
        # Iterating the page fetches every following page, so all rewards are compared.
        for reward in client.evals.rewards.list_trajectory_rewards(baseline_eval_id)
    }
    final = {
        reward.task_id: reward
        for reward in client.evals.rewards.list_trajectory_rewards(final_eval_id)
    }
    # Compare only tasks that recorded a reward in both evaluations, and say how many.
    shared = [
        task_id for task_id in baseline.keys() & final.keys() if task_id is not None
    ]
    print(f"compared_tasks={len(shared)}/{_TEST_TASKS}", flush=True)
    improved_task_id = max(
        shared, key=lambda task_id: final[task_id].reward - baseline[task_id].reward
    )
    before = client.trajectories.retrieve(
        baseline[improved_task_id].trajectory_id,
        include_steps=True,
    )
    after = client.trajectories.retrieve(
        final[improved_task_id].trajectory_id,
        include_steps=True,
    )
    print(f"improved_task_id={improved_task_id}", flush=True)
    print(f"before_tid={before.trajectory_id}", flush=True)
    print(f"before={_last_assistant_message(before)}", flush=True)
    print(f"after_tid={after.trajectory_id}", flush=True)
    print(f"after={_last_assistant_message(after)}", flush=True)


def _last_assistant_message(trajectory: object) -> str:
    return next(
        message.content
        for step in reversed(trajectory.steps)
        for message in reversed(step.messages)
        if message.role == "assistant"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-id", required=True)
    parser.add_argument("--model", default=_DEFAULT_MODEL)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--poll-seconds", type=float, default=5)
    args = parser.parse_args()
    train_and_evaluate(
        Client(max_retries=20),
        args.bench_id,
        args.model,
        args.num_steps,
        args.poll_seconds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
