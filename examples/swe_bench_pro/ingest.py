# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Upload SWE-Bench Pro V2 tasks with a fixed train/test split, then run a diagnostic."""

import argparse
import hashlib
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push
from trajectory.lib.benchmark_operations import BenchmarkOperationError

SOURCE_REPOSITORY = "https://github.com/scaleapi/SWE-bench_Pro-os"
SOURCE_COMMIT = "66f92766bba642462d4bbe5479e83f91f9211862"
ARCHIVE_URL = (
    f"https://codeload.github.com/scaleapi/SWE-bench_Pro-os/tar.gz/{SOURCE_COMMIT}"
)
STAGED_ARCHIVE = "swe_bench_pro_v2.tar.gz"
TASK_COUNT = 642
TEST_FRACTION = 0.15
MODEL = "Qwen/Qwen3.6-35B-A3B"
BENCHMARK_NAME = "swe-bench-pro"
RUNTIME_ROOT = Path(__file__).parent / "runtime"
BUILD_TIMEOUT_SECONDS = 45 * 60


def download_upstream(destination: Path) -> Path:
    """Unpack the pinned commit's license and v2 directory; return the checkout root."""
    with (
        urllib.request.urlopen(ARCHIVE_URL, timeout=600) as response,
        tarfile.open(fileobj=response, mode="r|gz") as archive,
    ):
        members = (
            member
            for member in archive
            if member.name.split("/")[1:2] in (["v2"], ["LICENSE"])
        )
        archive.extractall(destination, members=members, filter="data")
    return destination / f"SWE-bench_Pro-os-{SOURCE_COMMIT}"


def verify_checkout(upstream: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    if head != SOURCE_COMMIT:
        raise ValueError(f"{upstream} is at {head}; check out {SOURCE_COMMIT}")


def stage_runtime(upstream: Path, context: Path) -> list[str]:
    """Write the build context: the harness and the verified task files, minus solutions.

    Returns the release's instance IDs.
    """
    v2 = upstream / "v2"
    checksums = {
        path: digest
        for digest, path in (
            line.split(maxsplit=1)
            for line in (v2 / "SHA256SUMS").read_text().splitlines()
        )
    }
    staged = [
        path
        for path in sorted(checksums)
        if path == "tooling/configs/mini_toolcall.yaml"
        or (
            path.startswith("tasks/")
            and path.split("/")[2] in {"instruction.md", "task.toml", "tests"}
        )
    ]
    with tarfile.open(context / STAGED_ARCHIVE, "w:gz") as archive:
        for path in staged:
            if hashlib.sha256((v2 / path).read_bytes()).hexdigest() != checksums[path]:
                raise ValueError(f"v2/{path} differs from the pinned release")
            archive.add(v2 / path, arcname=f"v2/{path}")
        archive.add(v2 / "SHA256SUMS", arcname="v2/SHA256SUMS")
        archive.add(upstream / "LICENSE", arcname="LICENSE")
    for name in ("Dockerfile", "swe_bench_pro_harness.py"):
        shutil.copy(RUNTIME_ROOT / name, context / name)

    ids = sorted(
        {path.split("/")[1] for path in checksums if path.endswith("/task.toml")}
    )
    if len(ids) != TASK_COUNT:
        raise ValueError(
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
    instance_ids: list[str], name: str, train_tasks: int | None, test_tasks: int | None
) -> BenchmarkSpec:
    splits = split_tasks(instance_ids)
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
    agent_name: str,
    name: str,
    train_tasks: int | None,
    test_tasks: int | None,
    upstream: Path | None = None,
) -> str:
    with tempfile.TemporaryDirectory() as workspace:
        if upstream is None:
            upstream = download_upstream(Path(workspace))
        else:
            verify_checkout(upstream)
        context = Path(workspace) / "context"
        context.mkdir()
        benchmark = build_benchmark(
            stage_runtime(upstream, context), name, train_tasks, test_tasks
        )
        operation = start_push(
            Client(), benchmark, agent_name=agent_name, root=context, build_images=True
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
    parser.add_argument(
        "--upstream",
        type=Path,
        help="Use this SWE-bench_Pro-os checkout of the pinned commit instead of downloading it.",
    )
    args = parser.parse_args()
    bench_id = ingest(
        args.agent_name, args.name, args.train_tasks, args.test_tasks, args.upstream
    )
    benchmark_diagnostic(bench_id)


if __name__ == "__main__":
    main()
