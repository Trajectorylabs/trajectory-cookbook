# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.33"]
# ///
"""Ingest the pinned public Harvey LAB tasks with the Trajectory runtime overlay."""

import argparse
import hashlib
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
SOURCE_COMMIT = "1dd81403b2fbb60596f7aea3fcecafad7bf73143"
# Task directories with a task.json in harveyai/harvey-labs at SOURCE_COMMIT.
TASKS = (ROOT / "tasks.txt").read_text().split()
# Upstream's JUDGE_MODELS in lab_core/evaluation/run_eval.py.
JUDGES = ("claude-sonnet-4-6", "gpt-5.5")
JUDGE_KEYS = {
    "claude": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


def build_benchmark(
    name: str, judges: tuple[str, ...] = JUDGES, judge_base_url: str | None = None
) -> BenchmarkSpec:
    if judge_base_url is None and any(judge.startswith("openai/") for judge in judges):
        judge_base_url = "https://openrouter.ai/api/v1"
    ordered = sorted(
        TASKS,
        key=lambda task: hashlib.sha256(f"harvey-test-v1:{task}".encode()).digest(),
    )
    test_tasks = set(ordered[:120])
    # The key each provider's client in upstream's Judge reads; other models use
    # OpenAI or the OpenAI-compatible --judge-base-url.
    secrets = {
        next(
            (key for prefix, key in JUDGE_KEYS.items() if judge.startswith(prefix)),
            "OPENAI_API_KEY",
        )
        for judge in judges
    }
    env_vars = {
        **{secret: SecretRef(secret_ref=secret) for secret in sorted(secrets)},
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
                    f"--reasoning-effort low --judges {' '.join(judges)}"
                ),
                env_vars=env_vars,
                env_resources=EnvResources(network_mode="public", docker_engine=True),
            )
            for task in TASKS
        ],
    )


def ingest(
    name: str,
    agent_name: str,
    judges: tuple[str, ...] = JUDGES,
    judge_base_url: str | None = None,
) -> str:
    benchmark = build_benchmark(name, judges, judge_base_url)
    with Client() as client:
        result = push(client, benchmark, agent_name=agent_name, root=ROOT)
        print(f"bench_id={result.bench_id}", flush=True)
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=60 * 60)
        return result.bench_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="harvey-lab-public-podman")
    parser.add_argument(
        "--judges",
        nargs="+",
        default=JUDGES,
        metavar="MODEL",
        help="One judge model, or two whose scores are averaged",
    )
    parser.add_argument(
        "--judge-base-url",
        help="OpenAI-compatible endpoint for the judge (OPENAI_BASE_URL)",
    )
    args = parser.parse_args()
    ingest(args.name, args.agent_name, tuple(args.judges), args.judge_base_url)


if __name__ == "__main__":
    main()
