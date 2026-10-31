# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Ingest the pinned public Harvey LAB tasks with the Trajectory runtime overlay."""

import argparse
import hashlib
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
# Task directories with a task.json in harveyai/harvey-labs at the commit the Dockerfile pins.
TASKS = (ROOT / "tasks.txt").read_text().split()


def build_benchmark(name: str) -> BenchmarkSpec:
    ordered = sorted(
        TASKS,
        key=lambda task: hashlib.sha256(f"harvey-test-v1:{task}".encode()).digest(),
    )
    test_tasks = set(ordered[:120])
    return BenchmarkSpec(
        name=name,
        family="harvey-lab",
        description="Pinned public Harvey LAB tasks with the original harness and judge.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=task,
                split="test" if task in test_tasks else "train",
                run_command=(
                    "python -m lab_core.harness.run --model trajectory/session "
                    f"--task {task} --run-id trajectory --max-turns 200 "
                    "--reasoning-effort low --judge-model gpt-5.4-mini"
                ),
                env_vars={
                    "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
                    "HARVEY_PODMAN_DISABLE_CGROUPS": "1",
                },
                env_resources=EnvResources(network_mode="public", docker_engine=True),
            )
            for task in TASKS
        ],
    )


def ingest(name: str, agent_name: str) -> str:
    with Client() as client:
        result = push(client, build_benchmark(name), agent_name=agent_name, root=ROOT)
        print(f"bench_id={result.bench_id}", flush=True)
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=60 * 60)
        return result.bench_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="harvey-lab-public-podman")
    args = parser.parse_args()
    ingest(args.name, args.agent_name)


if __name__ == "__main__":
    main()
