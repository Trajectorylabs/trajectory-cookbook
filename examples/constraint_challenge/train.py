# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Evaluate, train, and compare Qwen on the Constraint Challenge."""

import argparse
import time
from dataclasses import dataclass

from trajectory import Client
from trajectory.types import TrainingRunResponse

_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"
_TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}
_TEST_TASKS = 64


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
    benchmark = client.benchmarks.specs.retrieve(bench_id)
    if not benchmark.tasks or {task.split for task in benchmark.tasks} != {
        "train",
        "test",
    }:
        raise ValueError("The benchmark must contain explicit train and test splits")

    baseline = _evaluate_model(
        client,
        bench_id,
        model,
        "Constraint Challenge baseline",
        poll_seconds,
    )
    created = client.training.create(
        bench_id=bench_id,
        base_model_id=model,
        training_options={
            "disable_thinking": True,
            "num_steps": num_steps,
            "train_batch_size": 4,
            "max_output_tokens_per_step": 32_768,
            "max_turns_per_trajectory": 1,
            "max_response_chars_per_tool_call": 256,
        },
    )
    run_id = created.training_run_id
    print(f"training_run_id={run_id}", flush=True)
    run = _wait_for_training(client, run_id, poll_seconds)
    if str(run.status) != "succeeded":
        raise RuntimeError(f"training ended with status={run.status}: {run.failure}")

    checkpoint = client.training.checkpoints.retrieve(run_id, num_steps)
    final = _evaluate_model(
        client,
        bench_id,
        model,
        f"Constraint Challenge {run_id} step {num_steps}",
        poll_seconds,
        checkpoint_id=checkpoint.checkpoint_id,
    )
    print(f"baseline_reward={baseline.reward:.6f}", flush=True)
    print(f"final_reward={final.reward:.6f}", flush=True)
    print(f"reward_delta={final.reward - baseline.reward:+.6f}", flush=True)
    _print_task_results(
        client,
        bench_id,
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
    checkpoint_id: str | None = None,
) -> Evaluation:
    evaluation = client.evals.start(
        bench_id,
        model_slug=model,
        checkpoint_id=checkpoint_id,
        display_name=display_name,
        extra_body={
            "eval_options": {
                "disable_thinking": True,
                "max_samples": _TEST_TASKS,
            }
        },
    )
    eval_id = evaluation.eval_run_id
    print(f"evaluation={display_name} eval_run_id={eval_id}", flush=True)
    while True:
        progress = client.evals.runs.retrieve_progress(eval_id)
        print(
            f"evaluation={display_name} status={progress.status} "
            f"rollouts={progress.terminal_rollouts}/{progress.total_rollouts}",
            flush=True,
        )
        if progress.status == "completed":
            break
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(f"evaluation failed: {progress.failure}")
        time.sleep(poll_seconds)

    result = next(
        run for run in client.evals.runs.list(bench_id) if run.eval_run_id == eval_id
    )
    if result.reward_mean is None:
        raise RuntimeError(f"completed evaluation {eval_id} has no reward")
    return Evaluation(eval_run_id=eval_id, reward=result.reward_mean)


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


def _print_task_results(
    client: Client,
    bench_id: str,
    baseline_eval_id: str,
    final_eval_id: str,
) -> None:
    benchmark = client.benchmarks.specs.retrieve(bench_id)
    task_kinds = {
        task.task_id: task.task_category.task_category_id for task in benchmark.tasks
    }
    baseline = {
        reward.task_id: reward
        for reward in client.evals.runs.list_trajectory_rewards(
            baseline_eval_id
        ).trajectory_rewards
    }
    final = {
        reward.task_id: reward
        for reward in client.evals.runs.list_trajectory_rewards(
            final_eval_id
        ).trajectory_rewards
    }
    kinds = sorted(set(task_kinds.values()))
    deltas = {}
    for kind in kinds:
        task_ids = [task_id for task_id in baseline if task_kinds[task_id] == kind]
        baseline_reward = sum(baseline[task_id].reward for task_id in task_ids) / len(
            task_ids
        )
        final_reward = sum(final[task_id].reward for task_id in task_ids) / len(
            task_ids
        )
        deltas[kind] = final_reward - baseline_reward
        print(
            f"task={kind} baseline={baseline_reward:.6f} final={final_reward:.6f} "
            f"delta={deltas[kind]:+.6f}",
            flush=True,
        )
    most_improved_kind = max(deltas, key=deltas.__getitem__)
    improved_task_id = next(
        task_id
        for task_id, reward in baseline.items()
        if task_kinds[task_id] == most_improved_kind
        and reward.reward == 0
        and final[task_id].reward == 1
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
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    train_and_evaluate(
        Client(), args.bench_id, args.model, args.num_steps, args.poll_seconds
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
