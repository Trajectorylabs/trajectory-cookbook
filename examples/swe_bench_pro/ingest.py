# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Upload SWE-Bench Pro V2 tasks with a fixed train/test split, then run a diagnostic."""

import argparse
import hashlib
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push
from trajectory.lib.benchmark_operations import BenchmarkOperationError

SOURCE_REPOSITORY = "https://github.com/scaleapi/SWE-bench_Pro-os"
SOURCE_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"
TASK_LIST_URL = f"https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/{SOURCE_COMMIT}/v2/SHA256SUMS"
TASK_COUNT = 642
TEST_FRACTION = 0.15
MODEL = "Qwen/Qwen3.6-35B-A3B"
BENCHMARK_NAME = "swe-bench-pro"
RUNTIME_ROOT = Path(__file__).parent / "runtime"
BUILD_TIMEOUT_SECONDS = 45 * 60


def load_instance_ids() -> list[str]:
    with urllib.request.urlopen(TASK_LIST_URL, timeout=60) as response:
        paths = [line.split()[1] for line in response.read().decode().splitlines()]
    ids = sorted({path.split("/")[1] for path in paths if path.endswith("/task.toml")})
    if len(ids) != TASK_COUNT:
        raise RuntimeError(
            f"Expected {TASK_COUNT} SWE-Bench Pro V2 tasks, found {len(ids)}"
        )
    return ids


def repository(instance_id: str) -> str:
    owner, name = re.match(r"instance_(.+?)__(.+?)-[0-9a-f]{40}", instance_id).groups()
    return f"{owner}/{name}"


def _order(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode()).hexdigest()


def split_tasks(instance_ids: list[str]) -> dict[str, list[str]]:
    """Hold out 15% of each repository's tasks, chosen by a hash of the instance ID."""
    by_repository = defaultdict(list)
    for instance_id in instance_ids:
        by_repository[repository(instance_id)].append(instance_id)
    splits = {"train": [], "test": []}
    for ids in by_repository.values():
        ordered = sorted(ids, key=_order)
        held_out = round(len(ordered) * TEST_FRACTION)
        splits["test"] += ordered[:held_out]
        splits["train"] += ordered[held_out:]
    return {split: sorted(ids, key=_order) for split, ids in splits.items()}


def task_spec(instance_id: str, split: str) -> TaskSpec:
    return TaskSpec(
        name=instance_id,
        split=split,
        run_command=f"python -u /opt/swe_bench_pro/swe_bench_pro_harness.py {instance_id}",
        env_resources={
            "docker_engine": True,
            # Pulls the task image and runs verifiers; the agent's container has no network.
            "network_mode": "public",
            "cpus": 2,
            "memory_mb": 8192,
        },
        tags=[repository(instance_id)],
        spec={
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "docker_image": f"ghcr.io/scaleapi/swe-bench_pro-v2:{instance_id}",
        },
    )


def build_benchmark(
    name: str, train_tasks: int | None, test_tasks: int | None
) -> BenchmarkSpec:
    splits = split_tasks(load_instance_ids())
    limits = {"train": train_tasks, "test": test_tasks}
    return BenchmarkSpec(
        name=name,
        description="SWE-Bench Pro V2: an offline bash agent, graded by the upstream verifier on a fresh image.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            task_spec(instance_id, split)
            for split, ids in splits.items()
            for instance_id in ids[: limits[split]]
        ],
    )


def ingest(
    agent_name: str, name: str, train_tasks: int | None, test_tasks: int | None
) -> str:
    benchmark = build_benchmark(name, train_tasks, test_tasks)
    operation = start_push(
        Client(), benchmark, agent_name=agent_name, root=RUNTIME_ROOT, build_images=True
    )
    print(f"operation_id={operation.id} tasks={len(benchmark.tasks)}", flush=True)
    try:
        result = operation.result(timeout=BUILD_TIMEOUT_SECONDS)
    except BenchmarkOperationError as error:
        for failure in error.failures:
            print(failure.to_json(), flush=True)
        raise
    print(f"bench_id={result.status.bench_id}", flush=True)
    return result.status.bench_id


def benchmark_diagnostic(bench_id: str) -> None:
    client = Client()
    diagnostic_id = client.diagnostics.start_benchmark(
        bench_id=bench_id, base_model_slug=MODEL
    ).benchmark_diagnostic_id
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while (status := client.diagnostics.get_status(diagnostic_id).status) not in {
        "completed",
        "failed",
        "cancelled",
    }:
        print(f"diagnostic_status={status}", flush=True)
        time.sleep(60)

    result = client.diagnostics.get_diagnostics(diagnostic_id)
    print(result.to_json(), flush=True)
    rewards = {
        item.task_id: item.reward
        for item in client.evals.rewards.list_trajectory_rewards(result.eval_run_id)
    }
    print(f"diagnostic_rewards={rewards}", flush=True)
    if (
        result.status != "completed"
        or result.failure is not None
        or not result.tasks
        or any(task.status != "completed" or task.failure for task in result.tasks)
        or not rewards
    ):
        raise RuntimeError(
            f"Diagnostic {diagnostic_id} did not pass; inspect the report above"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default=BENCHMARK_NAME)
    parser.add_argument(
        "--train-tasks", type=int, help="Upload only the first N training tasks."
    )
    parser.add_argument(
        "--test-tasks", type=int, help="Upload only the first N test tasks."
    )
    args = parser.parse_args()
    bench_id = ingest(args.agent_name, args.name, args.train_tasks, args.test_tasks)
    benchmark_diagnostic(bench_id)


if __name__ == "__main__":
    main()
