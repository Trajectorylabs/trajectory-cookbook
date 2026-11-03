# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.9.37", "matplotlib>=3.10"]
# ///
"""Run one or four chess training comparisons with 100 tasks and pass@1–4."""

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path
from statistics import mean

from trajectory import Client

ROOT = Path(__file__).resolve().parent.parent
TRAIN_MODELS = ("Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B")
OPTIONS = {
    "max_output_tokens_per_turn": 4000,
    "max_output_tokens_per_trajectory": 404000,
    "max_turns_per_trajectory": 204,
    "trajectory_timeout_seconds": 3600,
}
EVAL_OPTIONS = {
    "val_num_tasks": 100,
    "val_trajectories_per_task": 4,
    "val_max_concurrent_trajectories": 16,
}


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def model_options(client, bench_id, model, training=False, **overrides):
    """Check the current catalog before creating a job, with no silent option drops."""
    catalog = (client.training if training else client.evals).list_options(
        bench_id=bench_id
    )
    model_config = next((m for m in catalog.models if m.base_model_slug == model), None)
    if model_config is None:
        raise ValueError(f"Model is unavailable: {model}")
    requested = {**OPTIONS, "temperature": 1.0}
    effort = model_config.options.get("reasoning_effort")
    if effort is not None:
        choices = effort.choices or []
        value = "off" if "off" in choices else effort.default
        if value is None and choices:
            value = choices[0]
        if value is not None:
            requested["reasoning_effort"] = value
    requested.update(overrides)
    for key, value in requested.items():
        option = model_config.options.get(key)
        if option is None:
            raise ValueError(f"{model} does not support {key}")
        if option.choices and value not in option.choices:
            raise ValueError(f"Unsupported {key}={value} for {model}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if (option.min is not None and value < option.min) or (
                option.max is not None and value > option.max
            ):
                raise ValueError(f"{key}={value} is outside catalog bounds for {model}")
    return requested


def evaluation_tasks(client, bench_id):
    tasks = {}
    for task in client.benchmarks.tasks.list(bench_id, split="test"):
        record = json.loads(task.env_vars["CHESS_POSITION"])
        tasks[task.task_id] = {
            k: record[k] for k in ("position_id", "category", "rating_band")
        }
        tasks[task.task_id]["variant"] = task.env_vars["CHESS_VARIANT"]
    if len(tasks) != 100 or len({p["position_id"] for p in tasks.values()}) != 100:
        raise ValueError(
            "The benchmark must contain exactly 100 distinct held-out positions"
        )
    if len({p["variant"] for p in tasks.values()}) != 1:
        raise ValueError("A benchmark must contain one harness variant")
    if Counter(p["category"] for p in tasks.values()) != {
        category: 25 for category in ("openings", "tactics", "middlegame", "endgame")
    }:
        raise ValueError("Require twenty-five evaluation positions per category")
    return tasks


def pass_at_k(n, correct, k):
    if not 0 <= correct <= n or not 1 <= k <= n:
        raise ValueError("Invalid pass@k counts")
    return 1.0 if n - correct < k else 1.0 - comb(n - correct, k) / comb(n, k)


def summarize(outcomes, tasks):
    attempts_by_task = defaultdict(list)
    for outcome in outcomes:
        if outcome["task_id"] not in tasks:
            raise ValueError("Evaluation returned an unexpected task")
        attempts_by_task[outcome["task_id"]].append(outcome["solved"])

    missing_tasks = set(attempts_by_task) != set(tasks)
    wrong_attempt_count = any(
        len(attempts) != 4 for attempts in attempts_by_task.values()
    )
    if missing_tasks or wrong_attempt_count:
        raise ValueError(
            "Require exactly four attempts for each of the same 100 positions"
        )

    def scores(task_ids):
        result = {}
        for k in range(1, 5):
            per_position = [
                pass_at_k(4, sum(attempts_by_task[task_id]), k) for task_id in task_ids
            ]
            result[str(k)] = mean(per_position)
        return result

    def usage_stats(task_ids):
        selected = [outcome for outcome in outcomes if outcome["task_id"] in task_ids]
        return {
            "mean_python_calls": mean(outcome["python_calls"] for outcome in selected),
            "mean_model_turns": mean(outcome["model_turns"] for outcome in selected),
        }

    breakdowns = {}
    for field in ("category", "rating_band"):
        breakdowns[field] = {}
        categories = sorted({task[field] for task in tasks.values()})
        for category in categories:
            task_ids = [
                task_id for task_id, task in tasks.items() if task[field] == category
            ]
            breakdowns[field][str(category)] = {
                "positions": len(task_ids),
                "attempts": len(task_ids) * 4,
                "pass_at_k": scores(task_ids),
                **usage_stats(task_ids),
            }

    return {"pass_at_k": scores(tasks), **usage_stats(tasks), "breakdowns": breakdowns}


def wait_evaluation(client, run_id, tasks):
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
        details = list(
            pool.map(
                lambda a: client.trajectories.retrieve(
                    a.trajectory_id, include_reward_components=True
                ),
                attempts,
            )
        )
    outcomes = []
    for attempt, detail in zip(attempts, details):
        components = {c.name: c.value for c in detail.reward_components or []}
        grade = components.get("reward_accuracy")
        outcomes.append(
            {
                "task_id": attempt.task_id,
                "trajectory_id": attempt.trajectory_id,
                "native_reward": grade,
                "ending": detail.termination_reason,
                "python_calls": components.get("python_calls"),
                "model_turns": components.get("model_turns"),
                "solved": grade == 1.0 and detail.termination_reason == "ENV_DONE",
            }
        )
    failed = sum(
        a.rollout is not None and a.rollout.status == "FAILED" for a in attempts
    )
    unfinished = sum(
        a.rollout is None
        or a.rollout.status not in {"COMPLETED", "FAILED", "CANCELLED"}
        for a in attempts
    )
    cancelled = sum(
        a.rollout is not None and a.rollout.status == "CANCELLED" for a in attempts
    )
    graded = sum(o["native_reward"] is not None for o in outcomes)
    result = client.evals.runs.retrieve(run_id)
    report = {
        "eval_run_id": run_id,
        "attempted": len(attempts),
        "graded": graded,
        "failed": failed,
        "unfinished": unfinished,
        "cancelled": cancelled,
        "expected": progress.total_rollouts,
        "reward_mean": result.reward_mean,
        "endings": dict(Counter(o["ending"] for o in outcomes)),
        "outcomes": outcomes,
        "position_ids": sorted(p["position_id"] for p in tasks.values()),
    }
    if (
        progress.total_rollouts != 400
        or graded != 400
        or len(attempts) != 400
        or failed
        or cancelled
        or unfinished
        or result.reward_mean is None
    ):
        raise RuntimeError(
            f"Evaluation needs 400 graded attempts and no infrastructure failures: {report}"
        )
    if any(o["python_calls"] is None or o["model_turns"] is None for o in outcomes):
        raise RuntimeError("The harness did not record its tool-usage metrics")
    report.update(summarize(outcomes, tasks))
    print(f"eval={run_id} graded={graded} pass_at_k={report['pass_at_k']}", flush=True)
    return report


def evaluate(client, bench_id, model, state_path, checkpoint_id=None):
    tasks = evaluation_tasks(client, bench_id)
    options = model_options(client, bench_id, model, **EVAL_OPTIONS)
    identity = {
        "bench_id": bench_id,
        "model": model,
        "checkpoint_id": checkpoint_id,
        "variant": next(iter(tasks.values()))["variant"],
        "options": options,
        "position_ids": sorted(p["position_id"] for p in tasks.values()),
    }
    state = (
        json.loads(state_path.read_text()) if state_path.exists() else identity.copy()
    )
    if any(state.get(k) != v for k, v in identity.items()):
        raise ValueError(f"Saved evaluation settings differ: {state_path}")
    if "eval_run_id" not in state:
        checkpoint_options = (
            {"parent_checkpoint_id": checkpoint_id} if checkpoint_id else {}
        )
        result = client.evals.create(
            bench_id=bench_id,
            base_model_slug=model,
            task_split="test",
            display_name=f"Chess {model} {identity['variant']} {'checkpoint' if checkpoint_id else 'baseline'}",
            options=options,
            **checkpoint_options,
        )
        state["eval_run_id"] = result.eval_run_id
        save(state_path, state)
    if "report" not in state:
        state["report"] = wait_evaluation(client, state["eval_run_id"], tasks)
        save(state_path, state)
    return state


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
                raise RuntimeError(
                    f"Training ended without a successful optimizer step: {run.status}, {run.failure}"
                )
            return progress.completed_steps
        attempts = list(client.trajectories.list(training_run_id=run_id))
        if len(attempts) >= max_attempts or time.monotonic() - started > max_seconds:
            client.training.runs.cancel(run_id)
            raise RuntimeError(
                f"Cancelled validation training after {len(attempts)} attempts"
            )
        time.sleep(20)


def run(
    client,
    bench_id,
    state_path,
    model=TRAIN_MODELS[0],
    num_steps=100,
    baseline_path=None,
):
    state = (
        json.loads(state_path.read_text())
        if state_path.exists()
        else {"bench_id": bench_id, "model": model, "num_steps": num_steps}
    )
    if (state["bench_id"], state["model"], state["num_steps"]) != (
        bench_id,
        model,
        num_steps,
    ):
        raise ValueError("State belongs to a different training comparison")
    baseline_path = baseline_path or state_path.with_name(
        state_path.stem + "-baseline.json"
    )
    baseline = evaluate(client, bench_id, model, baseline_path)
    generation = {
        k: v for k, v in baseline["options"].items() if not k.startswith("val_")
    }
    options = model_options(
        client,
        bench_id,
        model,
        training=True,
        **generation,
        num_steps=num_steps,
        batch_size=16,
        group_size=8,
        max_concurrent_trajectories=128,
        val_enabled=False,
        batch_timeout_seconds=2400,
    )
    if "options" in state and state["options"] != options:
        raise ValueError("Saved training options differ")
    state["options"] = options
    if "training_id" not in state:
        training = client.training.create(
            bench_id=bench_id,
            base_model_slug=model,
            display_name=f"Chess {model} {baseline['variant']}",
            options=options,
        )
        state["training_id"] = training.training_run_id
        save(state_path, state)
    if "completed_steps" not in state:
        state["completed_steps"] = wait_training(
            client,
            state["training_id"],
            max_attempts=num_steps * 16 * 8 * 4,
            max_seconds=86400,
        )
        if state["completed_steps"] < num_steps:
            raise RuntimeError(
                "Training did not complete the requested optimizer steps"
            )
        save(state_path, state)
    if "checkpoint_id" not in state:
        checkpoint = client.training.checkpoints.retrieve(
            state["training_id"], num_steps
        )
        state["checkpoint_id"] = checkpoint.checkpoint_id
        save(state_path, state)
    final = evaluate(
        client,
        bench_id,
        model,
        state_path.with_name(state_path.stem + "-checkpoint.json"),
        state["checkpoint_id"],
    )
    if (
        baseline["position_ids"] != final["position_ids"]
        or baseline["options"] != final["options"]
    ):
        raise RuntimeError("Before/after record IDs or generation settings differ")
    state["before"] = baseline
    state["after"] = final
    state["pass_at_k_delta"] = {
        k: final["report"]["pass_at_k"][k] - baseline["report"]["pass_at_k"][k]
        for k in ("1", "2", "3", "4")
    }
    save(state_path, state)
    return state


def result_name(model, variant):
    return model.replace("/", "-") + "-" + variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bench-id")
    mode.add_argument("--reasoning-bench-id")
    parser.add_argument("--python-bench-id")
    parser.add_argument("--state", type=Path)
    parser.add_argument(
        "--results-dir", type=Path, default=ROOT / "results" / "best-move"
    )
    parser.add_argument("--model", default=TRAIN_MODELS[0])
    parser.add_argument("--num-steps", type=int, default=100)
    args = parser.parse_args()
    client = Client()
    print(client.organizations.retrieve_current(), flush=True)
    if args.bench_id:
        run(
            client,
            args.bench_id,
            args.state or args.results_dir / "training.json",
            args.model,
            args.num_steps,
        )
    else:
        if not args.python_bench_id:
            parser.error("--python-bench-id is required for the four-run matrix")
        benchmarks = {
            "reasoning": args.reasoning_bench_id,
            "python": args.python_bench_id,
        }
        tasks = [
            evaluation_tasks(client, benchmark_id)
            for benchmark_id in benchmarks.values()
        ]
        if {p["position_id"] for p in tasks[0].values()} != {
            p["position_id"] for p in tasks[1].values()
        }:
            raise ValueError(
                "The two benchmarks contain different evaluation positions"
            )
        for model in TRAIN_MODELS:
            for variant, benchmark_id in benchmarks.items():
                stem = result_name(model, variant)
                run(
                    client,
                    benchmark_id,
                    args.results_dir / f"{stem}-training.json",
                    model,
                    args.num_steps,
                    baseline_path=args.results_dir / f"{stem}-baseline.json",
                )


if __name__ == "__main__":
    main()
