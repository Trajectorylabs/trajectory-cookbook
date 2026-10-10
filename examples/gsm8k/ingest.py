# /// script
# dependencies = ["trajectory-sdk", "httpx"]
# ///
"""Ingest GSM8K train/test tasks through the Trajectory SDK."""

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import httpx
from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import (
    DockerfileBuild,
    push,
    start_task_diagnostic,
    wait_for_benchmark_images,
)

_RUNTIME_SOURCE = Path(__file__).parent / "runtime"
_RUNTIME_DOCKERFILE = "Dockerfile"
_RUN_COMMAND = "python -u /opt/gsm8k/gsm8k_harness.py --task-file {task_file}"
_BUILD_TIMEOUT_SECONDS = 45 * 60
_TRAIN_TASKS = 64
_TEST_TASKS = 16
_DATASET_URLS = {
    "train": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
    "test": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
}


def build_benchmark(rows_by_split: dict[str, list[dict]], name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="gsm8k",
        description="GSM8K exact-match training benchmark.",
        runtime=DockerfileBuild(_RUNTIME_DOCKERFILE),
        tasks=[
            TaskSpec(
                name=f"gsm8k/{split}_{index:04d}",
                split=split,
                run_command=_RUN_COMMAND.format(
                    task_file=f"/opt/gsm8k/tasks/{split}_{index:04d}.json"
                ),
                tags=["gsm8k"],
            )
            for split, rows in rows_by_split.items()
            for index, row in enumerate(rows)
        ],
    )


def ingest(name: str, agent_name: str) -> str:
    rows = {
        "train": _load_rows("train", _TRAIN_TASKS),
        "test": _load_rows("test", _TEST_TASKS),
    }
    return _push_benchmark(rows, name, agent_name)


def ingest_smoketest(agent_name: str) -> str:
    # Use one task per split for the quickest check of ingestion and runtime errors.
    rows = {split: _load_rows(split, 1) for split in ("train", "test")}
    return _push_benchmark(rows, "gsm8k-smoketest", agent_name)


def _push_benchmark(rows: dict[str, list[dict]], name: str, agent_name: str) -> str:
    client = Client()
    with tempfile.TemporaryDirectory(prefix="gsm8k-benchmark-") as directory:
        package_root = Path(directory)
        _stage_runtime(rows, package_root)
        result = push(
            client,
            build_benchmark(rows, name),
            agent_name=agent_name,
            root=package_root,
        )
    print(f"agent_name={agent_name}", flush=True)
    print(f"bench_name={name}", flush=True)
    wait_for_benchmark_images(
        client,
        result.bench_id,
        timeout_seconds=_BUILD_TIMEOUT_SECONDS,
    )
    return result.bench_id


def task_diagnose(agent_name: str) -> None:
    # Catch packaging, execution, and grading errors on one task before benchmark upload.
    client = Client()
    rows = {"train": _load_rows("train", 1)}
    with tempfile.TemporaryDirectory(prefix="gsm8k-diagnostic-") as directory:
        package_root = Path(directory)
        _stage_runtime(rows, package_root)
        diagnostic = start_task_diagnostic(
            client,
            TaskSpec(
                name="gsm8k/train_0000",
                split="train",
                runtime=DockerfileBuild(_RUNTIME_DOCKERFILE),
                run_command=_RUN_COMMAND.format(
                    task_file="/opt/gsm8k/tasks/train_0000.json"
                ),
                tags=["gsm8k"],
            ),
            agent_name=agent_name,
            root=package_root,
            base_model_slug="Qwen/Qwen3.5-4B",
            timeout_seconds=_BUILD_TIMEOUT_SECONDS,
        )
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def benchmark_diagnostic(bench_id: str) -> None:
    # Check a subset of uploaded tasks and runtimes before evaluation or training.
    client = Client()
    diagnostic = client.diagnostics.start_benchmark(
        bench_id=bench_id,
        base_model_slug="Qwen/Qwen3.5-4B",
    )
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def _wait_for_diagnostic(client: Client, diagnostic_id: str) -> None:
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
            task.status != "completed" or task.failure is not None
            for task in result.tasks
        )
    ):
        raise RuntimeError(
            f"Diagnostic {diagnostic_id} did not pass; inspect the report above"
        )
    evaluation = client.evals.runs.retrieve(result.eval_run_id)
    if evaluation.reward_mean is None:
        raise RuntimeError(
            f"Diagnostic {diagnostic_id} completed without a recorded reward"
        )
    print(f"diagnostic_reward={evaluation.reward_mean}", flush=True)


def _load_rows(split: str, limit: int) -> list[dict]:
    response = httpx.get(_DATASET_URLS[split], timeout=60)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines()]
    return rows[:limit]


def _stage_runtime(rows_by_split: dict[str, list[dict]], package_root: Path) -> None:
    shutil.copy(_RUNTIME_SOURCE / "Dockerfile", package_root / "Dockerfile")
    shutil.copy(_RUNTIME_SOURCE / "gsm8k_harness.py", package_root / "gsm8k_harness.py")
    tasks_root = package_root / "tasks"
    tasks_root.mkdir()
    for split, rows in rows_by_split.items():
        for index, row in enumerate(rows):
            task = {"question": row["question"], "expected_answer": row["answer"]}
            (tasks_root / f"{split}_{index:04d}.json").write_text(json.dumps(task))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="gsm8k-trajectory-sdk")
    args = parser.parse_args()

    bench_id = ingest(args.name, args.agent_name)
    benchmark_diagnostic(bench_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
