# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Ingest the pinned public Tau 3 Banking tasks and build their SDK runtime."""

import argparse
from pathlib import Path

from runtime.tau3_banking.protocol import (
    EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
    MAX_ERRORS,
    MAX_STEPS,
    RETRIEVAL_CONFIG,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TAG,
    TASK_IDS,
    TASKS_SHA256,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_REASONING_EFFORT,
)
from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

ROOT = Path(__file__).parent


def build_benchmark(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        description="Pinned public Tau 3 Banking tasks and original upstream evaluator.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=task_id,
                split="train" if index < 80 else "test",
                run_command=f"cd /opt/tau3 && /opt/tau3/.venv/bin/python -u -m tau3_banking.agent {task_id}",
                env_vars={
                    "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
                    "TAU2_DATA_DIR": "/opt/tau2-source/data",
                },
                env_resources={
                    "cpus": 4,
                    "memory_mb": 8192,
                    "network_mode": "public",
                    "agent_user": "root",
                },
                spec={
                    "domain": "banking_knowledge",
                    "source_repository": SOURCE_REPOSITORY,
                    "source_tag": SOURCE_TAG,
                    "source_commit": SOURCE_COMMIT,
                    "tasks_sha256": TASKS_SHA256,
                    "retrieval": RETRIEVAL_CONFIG,
                    "user_and_judge_model": USER_AND_JUDGE_MODEL,
                    "user_and_judge_reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
                    "max_steps": MAX_STEPS,
                    "max_errors": MAX_ERRORS,
                    "empty_user_response_max_attempts": EMPTY_USER_RESPONSE_MAX_ATTEMPTS,
                },
            )
            for index, task_id in enumerate(TASK_IDS)
        ],
    )


def ingest(name: str, agent_name: str) -> str:
    with Client() as client:
        result = push(client, build_benchmark(name), agent_name=agent_name, root=ROOT)
        print(f"bench_id={result.bench_id}", flush=True)
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=45 * 60)
        return result.bench_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="tau3-banking-trajectory-sdk")
    args = parser.parse_args()
    ingest(args.name, args.agent_name)


if __name__ == "__main__":
    main()
