# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.9.37"]
# ///
"""Evaluate, train briefly, and evaluate a checkpoint on the same chess tasks."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from trajectory import Client

MODEL = "Qwen/Qwen3.5-4B"
OPTIONS = {"reasoning_effort": "off", "max_output_tokens_per_turn": 4096,
           "max_output_tokens_per_trajectory": 45056, "max_turns_per_trajectory": 24,
           "trajectory_timeout_seconds": 900}


def save(path, state):
    path.write_text(json.dumps(state, indent=2) + "\n")


def wait_evaluation(client, run_id):
    last = None
    while True:
        progress = client.evals.runs.progress(run_id)
        line = f"eval={run_id} status={progress.status} tasks={progress.terminal_rollouts}/{progress.total_rollouts}"
        if line != last:
            print(line, flush=True)
            last = line
        if progress.status == "completed":
            break
        if progress.status in {"failed", "cancelled"}:
            raise RuntimeError(f"Evaluation failed: {progress.failure}")
        time.sleep(10)
    attempts = list(client.trajectories.list(eval_run_id=run_id))
    with ThreadPoolExecutor(max_workers=16) as pool:
        details = list(pool.map(lambda a: client.trajectories.retrieve(
            a.trajectory_id, include_reward_components=True), attempts))
    graded = sum(any(c.name == "reward_accuracy" for c in d.reward_components or []) for d in details)
    failed = sum(a.rollout is not None and a.rollout.status == "FAILED" for a in attempts)
    unfinished = sum(a.rollout is None or a.rollout.status not in {"COMPLETED", "FAILED", "CANCELLED"} for a in attempts)
    cancelled = sum(a.rollout is not None and a.rollout.status == "CANCELLED" for a in attempts)
    result = client.evals.runs.retrieve(run_id)
    report = {"eval_run_id": run_id, "attempted": len(attempts), "graded": graded,
              "failed": failed, "unfinished": unfinished, "cancelled": cancelled,
              "expected": progress.total_rollouts, "reward_mean": result.reward_mean,
              "endings": dict(Counter(d.termination_reason for d in details)),
              "task_ids": sorted(a.task_id for a in attempts)}
    print(json.dumps(report), flush=True)
    if graded != progress.total_rollouts or len(attempts) != graded or failed or cancelled or unfinished or result.reward_mean is None:
        raise RuntimeError("Evaluation has missing grades or failed attempts; inspect its trajectories")
    return report


def wait_training(client, run_id, max_attempts=128, max_seconds=3600):
    started, last = time.monotonic(), None
    while True:
        run = client.training.runs.retrieve(run_id)
        progress = client.training.runs.progress(run_id)
        line = f"training={run_id} status={run.status} steps={progress.completed_steps}/{progress.total_steps}"
        if line != last:
            print(line, flush=True)
            last = line
        if run.status in {"succeeded", "failed", "cancelled"}:
            if run.status != "succeeded" or progress.completed_steps < 1:
                raise RuntimeError(f"Training ended without a successful optimizer step: {run.status}, {run.failure}")
            return progress.completed_steps
        attempts = list(client.trajectories.list(training_run_id=run_id))
        if len(attempts) >= max_attempts or time.monotonic() - started > max_seconds:
            client.training.runs.cancel(run_id)
            raise RuntimeError(f"Cancelled validation training after {len(attempts)} attempts; inspect its progress")
        time.sleep(20)


def run(client, bench_id, state_path, model=MODEL, num_steps=1, eval_tasks=500):
    state = json.loads(state_path.read_text()) if state_path.exists() else {"bench_id": bench_id, "model": model}
    if state["bench_id"] != bench_id or state["model"] != model:
        raise ValueError("State belongs to a different benchmark or model")
    eval_options = {**OPTIONS, "val_num_tasks": eval_tasks,
                    "val_trajectories_per_task": 1, "val_max_concurrent_trajectories": 32}
    settings = {"num_steps": num_steps, "eval_options": eval_options}
    if "settings" in state and state["settings"] != settings:
        raise ValueError("Saved run settings differ; use a new state file")
    state["settings"] = settings
    if "baseline_id" not in state:
        baseline = client.evals.create(bench_id=bench_id, base_model_slug=model,
                                       display_name="Chess mates baseline", options=eval_options)
        state["baseline_id"] = baseline.eval_run_id
        save(state_path, state)
    if "baseline" not in state:
        state["baseline"] = wait_evaluation(client, state["baseline_id"])
        save(state_path, state)
    if "training_id" not in state:
        training = client.training.create(bench_id=bench_id, base_model_slug=model,
            display_name="Chess mates workflow validation", options={**OPTIONS,
                "num_steps": num_steps, "batch_size": 1, "group_size": 8,
                "max_concurrent_trajectories": 8, "val_enabled": False,
                "batch_timeout_seconds": 2400})
        state["training_id"] = training.training_run_id
        save(state_path, state)
    if "completed_steps" not in state:
        state["completed_steps"] = wait_training(client, state["training_id"])
        save(state_path, state)
    if "checkpoint_id" not in state:
        checkpoint = client.training.checkpoints.retrieve(state["training_id"], num_steps)
        state["checkpoint_id"] = checkpoint.checkpoint_id
        save(state_path, state)
    if "final_id" not in state:
        final = client.evals.create(bench_id=bench_id, base_model_slug=model,
            parent_checkpoint_id=state["checkpoint_id"], display_name="Chess mates checkpoint",
            options=eval_options)
        state["final_id"] = final.eval_run_id
        save(state_path, state)
    if "final" not in state:
        state["final"] = wait_evaluation(client, state["final_id"])
        save(state_path, state)
    if state["baseline"]["task_ids"] != state["final"]["task_ids"]:
        raise RuntimeError("Baseline and checkpoint evaluated different task IDs")
    state["reward_delta"] = state["final"]["reward_mean"] - state["baseline"]["reward_mean"]
    save(state_path, state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-id", required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--eval-tasks", type=int, default=500)
    args = parser.parse_args()
    print(json.dumps(run(Client(), args.bench_id, args.state, args.model,
                         args.num_steps, args.eval_tasks), indent=2))


if __name__ == "__main__":
    main()
