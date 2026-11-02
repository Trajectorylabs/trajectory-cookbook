# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.37"]
# ///
"""Ingest GSM8K train/test tasks through the Trajectory SDK."""

import argparse
import time
from pathlib import Path

import trajectory
from trajectory.lib import (
    DockerfileBuild,
    start_push,
    start_task_diagnostic,
)
from trajectory.lib.benchmark_operations import BenchmarkOperationError

_RUNTIME_ROOT = Path(__file__).parent / "runtime"
_RUNTIME_DOCKERFILE = "Dockerfile"
_RUN_COMMAND = "python -u /opt/gsm8k/gsm8k_harness.py"
_BUILD_TIMEOUT_SECONDS = 45 * 60
# The settings train.py evaluates and trains with.
_RUN_OPTIONS = {"reasoning_effort": "off", "max_output_tokens_per_turn": 2_048}
_TRAIN_TASKS = 64
_TEST_TASKS = 16
_BENCHMARK_NAME = "gsm8k-trajectory-sdk"


def build_benchmark(name: str) -> trajectory.BenchmarkSpec:
    return trajectory.BenchmarkSpec(
        name=name,
        description="GSM8K exact-match training benchmark.",
        runtime=DockerfileBuild(_RUNTIME_DOCKERFILE),
        tasks=[
            trajectory.TaskSpec(
                name=f"gsm8k/{split}_{index:04d}",
                split=split,
                run_command=f"{_RUN_COMMAND} --task_id {task_id}",
                tags=["gsm8k"],
            )
            for split, start, count in (
                ("train", 0, _TRAIN_TASKS),
                ("test", _TRAIN_TASKS, _TEST_TASKS),
            )
            for index, task_id in enumerate(range(start, start + count))
        ],
    )


def ingest(name: str, agent_name: str) -> str:
    return _push_benchmark(name, agent_name)


def _push_benchmark(name: str, agent_name: str) -> str:
    client = trajectory.Client()
    operation = start_push(
        client,
        build_benchmark(name),
        agent_name=agent_name,
        root=_RUNTIME_ROOT,
        build_images=True,
    )
    print(f"operation_id={operation.id}", flush=True)
    print(f"agent_name={agent_name}", flush=True)
    print(f"bench_name={name}", flush=True)
    try:
        result = operation.result(timeout=_BUILD_TIMEOUT_SECONDS)
    except BenchmarkOperationError as error:
        for failure in error.failures:
            print(failure.to_json(), flush=True)
        raise
    return result.status.bench_id


def task_diagnose(agent_name: str) -> None:
    # Catch packaging, execution, and grading errors on one task before benchmark upload.
    client = trajectory.Client()
    diagnostic = start_task_diagnostic(
        client,
        trajectory.TaskSpec(
            name="gsm8k/train_0000",
            split="train",
            runtime=DockerfileBuild(_RUNTIME_DOCKERFILE),
            run_command=f"{_RUN_COMMAND} --task_id 0",
            tags=["gsm8k"],
        ),
        agent_name=agent_name,
        root=_RUNTIME_ROOT,
        base_model_slug="Qwen/Qwen3.5-4B",
        options=_RUN_OPTIONS,
        timeout_seconds=_BUILD_TIMEOUT_SECONDS,
    )
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def benchmark_diagnostic(bench_id: str) -> None:
    # Check a subset of uploaded tasks and runtimes before evaluation or training.
    client = trajectory.Client()
    diagnostic = client.diagnostics.start_benchmark(
        bench_id=bench_id,
        base_model_slug="Qwen/Qwen3.5-4B",
        options=_RUN_OPTIONS,
    )
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def _wait_for_diagnostic(client: trajectory.Client, diagnostic_id: str) -> None:
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while True:
        status = client.diagnostics.get_status(diagnostic_id)
        print(f"diagnostic_status={status.status}", flush=True)
        if status.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(5)

    result = client.diagnostics.get_diagnostics(diagnostic_id)
    print(result.to_json(), flush=True)
    if (
        result.status != "completed"
        or result.failure is not None
        or not result.tasks
        or any(
            task.status != "completed" or task.reward is None for task in result.tasks
        )
    ):
        raise RuntimeError(
            f"Diagnostic {diagnostic_id} did not pass; inspect the report above"
        )
    for task in result.tasks:
        print(
            f"task_id={task.task_id} ending={task.termination_reason} "
            f"reward={task.reward}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default=_BENCHMARK_NAME)
    args = parser.parse_args()

    bench_id = ingest(args.name, args.agent_name)
    benchmark_diagnostic(bench_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
