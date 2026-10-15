# /// script
# dependencies = ["trajectory-sdk==0.9.16"]
# ///
"""Ingest the pinned public Tau 3 Retail tasks with upstream's official train/test split."""

import argparse
import time
from pathlib import Path

from runtime.tau3_retail.protocol import (
    MAX_ERRORS,
    MAX_STEPS,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    SOURCE_TAG,
    TASK_IDS,
    TRAIN_IDS,
    USER_AND_JUDGE_MODEL,
    USER_AND_JUDGE_REASONING_EFFORT,
)
from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import (
    DockerfileBuild,
    push,
    start_task_diagnostic,
    wait_for_benchmark_images,
)
from trajectory.types.benchmarks.runtime_spec import RuntimeSpec

ROOT = Path(__file__).parent
_MODEL = "Qwen/Qwen3.5-4B"
_BUILD_TIMEOUT_SECONDS = 45 * 60


def task_spec(task_id: str, runtime: RuntimeSpec | None = None) -> TaskSpec:
    return TaskSpec(
        name=f"retail/{task_id}",
        split="train" if task_id in TRAIN_IDS else "test",
        runtime=runtime,
        run_command=(
            "cd /opt/tau3-retail && "
            f"/opt/tau3-retail/.venv/bin/python -u -m tau3_retail.agent {task_id}"
        ),
        env_vars={
            "OPENAI_API_KEY": SecretRef(secret_ref="OPENAI_API_KEY"),
            "TAU2_DATA_DIR": "/opt/tau2-source/data",
        },
        env_resources={
            "cpus": 2,
            "memory_mb": 4096,
            "network_mode": "public",
            "agent_user": "root",
        },
        spec={
            "domain": "retail",
            "source_repository": SOURCE_REPOSITORY,
            "source_tag": SOURCE_TAG,
            "source_commit": SOURCE_COMMIT,
            "user_and_judge_model": USER_AND_JUDGE_MODEL,
            "user_and_judge_reasoning_effort": USER_AND_JUDGE_REASONING_EFFORT,
            "max_steps": MAX_STEPS,
            "max_errors": MAX_ERRORS,
        },
        tags=["tau3", "tau3-retail"],
    )


def build_benchmark(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="tau3-retail",
        description=(
            "Pinned public Tau 3 Retail tasks, simulated user, and original upstream evaluator."
        ),
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[task_spec(task_id) for task_id in TASK_IDS],
    )


def ingest(name: str, agent_name: str) -> str:
    return _push_benchmark(build_benchmark(name), agent_name)


def task_diagnose(agent_name: str, task_id: str = TRAIN_IDS[0]) -> None:
    # Catch packaging, model-routing, user-simulator, and grading errors before upload.
    client = Client()
    diagnostic = start_task_diagnostic(
        client,
        task_spec(task_id, runtime=DockerfileBuild("runtime/Dockerfile")),
        agent_name=agent_name,
        root=ROOT,
        base_model_slug=_MODEL,
        timeout_seconds=_BUILD_TIMEOUT_SECONDS,
    )
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def benchmark_diagnostic(bench_id: str) -> None:
    # Check a subset of uploaded tasks before evaluation or training.
    client = Client()
    diagnostic = client.diagnostics.start_benchmark(bench_id=bench_id, base_model_slug=_MODEL)
    _wait_for_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def _push_benchmark(benchmark: BenchmarkSpec, agent_name: str) -> str:
    with Client() as client:
        result = push(client, benchmark, agent_name=agent_name, root=ROOT)
        print(f"agent_name={agent_name}", flush=True)
        print(f"bench_name={benchmark.name}", flush=True)
        print(f"bench_id={result.bench_id}", flush=True)
        wait_for_benchmark_images(
            client, result.bench_id, timeout_seconds=_BUILD_TIMEOUT_SECONDS
        )
        return result.bench_id


def _wait_for_diagnostic(client: Client, diagnostic_id: str) -> None:
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while True:
        status = client.diagnostics.get_status(diagnostic_id)
        print(f"diagnostic_status={status.status}", flush=True)
        if status.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(10)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="tau3-retail-trajectory-sdk")
    parser.add_argument(
        "--task-diagnostic",
        action="store_true",
        help="run one training task without uploading a benchmark",
    )
    args = parser.parse_args()
    if args.task_diagnostic:
        task_diagnose(args.agent_name)
    else:
        benchmark_diagnostic(ingest(args.name, args.agent_name))


if __name__ == "__main__":
    main()
