# /// script
# dependencies = ["trajectory-sdk", "httpx"]
# ///
"""Ingest GSM8K train/test tasks through the Trajectory SDK."""

import argparse
import json
from pathlib import Path

import httpx
from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUNTIME_DOCKERFILE = "runtime/Dockerfile"
_RUN_COMMAND = "python -u /opt/gsm8k/gsm8k_harness.py"
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
                run_command=_RUN_COMMAND,
                env_vars={
                    "GSM8K_QUESTION": row["question"],
                    "GSM8K_ANSWER": row["answer"],
                },
                tags=["gsm8k"],
            )
            for split, rows in rows_by_split.items()
            for index, row in enumerate(rows)
        ],
    )


def ingest(name: str, skip_build: bool) -> str:
    client = Client()
    agent_id = client.agents.get_default().agent_id
    rows = {
        "train": _load_rows("train", _TRAIN_TASKS),
        "test": _load_rows("test", _TEST_TASKS),
    }
    result = push(client, build_benchmark(rows, name), agent_id=agent_id, root=_ROOT)
    print(f"agent_id={agent_id}", flush=True)
    print(f"bench_id={result.bench_id}", flush=True)
    if not skip_build:
        wait_for_benchmark_images(
            client,
            result.bench_id,
            timeout_seconds=_BUILD_TIMEOUT_SECONDS,
        )
    return result.bench_id


def _load_rows(split: str, limit: int) -> list[dict]:
    response = httpx.get(_DATASET_URLS[split], timeout=60)
    response.raise_for_status()
    rows = [json.loads(line) for line in response.text.splitlines()]
    return rows[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="gsm8k-trajectory-sdk")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    ingest(args.name, args.skip_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
