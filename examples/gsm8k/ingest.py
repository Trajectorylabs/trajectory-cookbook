# /// script
# dependencies = ["trajectory-sdk", "httpx"]
# ///
"""Ingest GSM8K train/test tasks through the Trajectory SDK."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import httpx
from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push

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


def ingest(name: str, agent_name: str, skip_build: bool) -> str:
    client = Client()
    rows = {
        "train": _load_rows("train", _TRAIN_TASKS),
        "test": _load_rows("test", _TEST_TASKS),
    }
    with tempfile.TemporaryDirectory(prefix="gsm8k-benchmark-") as directory:
        package_root = Path(directory)
        _stage_runtime(rows, package_root)
        operation = start_push(
            client,
            build_benchmark(rows, name),
            agent_name=agent_name,
            root=package_root,
            build_images=not skip_build,
        )
    print(f"agent_name={agent_name}", flush=True)
    print(f"operation_id={operation.id}", flush=True)
    result = operation.result(timeout=_BUILD_TIMEOUT_SECONDS)
    print(f"bench_id={result.status.bench_id}", flush=True)
    return result.status.bench_id


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
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    ingest(args.name, args.agent_name, args.skip_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
