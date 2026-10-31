# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Ingest the pinned public Harvey LAB tasks with the Trajectory runtime overlay."""

import argparse
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
SOURCE_COMMIT = "1dd81403b2fbb60596f7aea3fcecafad7bf73143"
# Task directories with a task.json in harveyai/harvey-labs at SOURCE_COMMIT.
TASKS = (ROOT / "tasks.txt").read_text().split()
_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache")


def build_benchmark(
    name: str, judge_model: str = "gpt-5.4-mini", judge_base_url: str | None = None
) -> BenchmarkSpec:
    ordered = sorted(
        TASKS,
        key=lambda task: hashlib.sha256(f"harvey-test-v1:{task}".encode()).digest(),
    )
    test_tasks = set(ordered[:120])
    env_vars = {
        "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
        "HARVEY_PODMAN_DISABLE_CGROUPS": "1",
    }
    if judge_base_url:
        env_vars["OPENAI_BASE_URL"] = judge_base_url
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
                    f"--reasoning-effort low --judge-model {judge_model}"
                ),
                env_vars=env_vars,
                env_resources=EnvResources(network_mode="public", docker_engine=True),
            )
            for task in TASKS
        ],
    )


def git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def stage(checkout: Path, dest: Path) -> Path:
    """Copy the checkout's lab_core/, then the overlay over it, into an upload root."""
    head = git_head(checkout)
    if head != SOURCE_COMMIT:
        raise SystemExit(
            f"{checkout} is at {head}, but tasks.txt and the overlay match {SOURCE_COMMIT}. "
            f"Run `git -C {checkout} checkout {SOURCE_COMMIT}`, or omit --upstream."
        )
    # tasks/ has about 63,000 files, over the build-context limit; the image downloads it.
    shutil.copytree(
        checkout / "lab_core", dest / "runtime" / "lab_core", ignore=_IGNORE
    )
    shutil.copytree(
        ROOT / "runtime", dest / "runtime", ignore=_IGNORE, dirs_exist_ok=True
    )
    shutil.copy(ROOT / ".dockerignore", dest)
    return dest


def ingest(
    name: str,
    agent_name: str,
    judge_model: str = "gpt-5.4-mini",
    judge_base_url: str | None = None,
    upstream: Path | None = None,
) -> str:
    benchmark = build_benchmark(name, judge_model, judge_base_url)
    with tempfile.TemporaryDirectory() as tmp, Client() as client:
        root = stage(upstream, Path(tmp)) if upstream else ROOT
        result = push(client, benchmark, agent_name=agent_name, root=root)
        print(f"bench_id={result.bench_id}", flush=True)
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=60 * 60)
        return result.bench_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="harvey-lab-public-podman")
    parser.add_argument("--judge-model", default="gpt-5.4-mini")
    parser.add_argument(
        "--judge-base-url",
        help="OpenAI-compatible endpoint for the judge (OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        help=f"harveyai/harvey-labs checkout at {SOURCE_COMMIT} to take lab_core/ from",
    )
    args = parser.parse_args()
    ingest(
        args.name, args.agent_name, args.judge_model, args.judge_base_url, args.upstream
    )


if __name__ == "__main__":
    main()
