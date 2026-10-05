# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Diagnose one local task or an already uploaded benchmark."""

import argparse
import time
from pathlib import Path

from trajectory import Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_task_diagnostic

_POLL_SECONDS = 5
_TIMEOUT_SECONDS = 45 * 60
_T_FACTORY_ROOT = Path(__file__).parent / "t_factory"


def print_report(client: Client, diagnostic_id: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        status = client.diagnostics.get_status(diagnostic_id)
        if status.status in {"completed", "failed", "cancelled"}:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Diagnostic timed out: {diagnostic_id}")
        time.sleep(_POLL_SECONDS)

    report = client.diagnostics.get_diagnostics(diagnostic_id)
    print(f"status={report.status} task_counts={report.task_counts.model_dump()}")
    if report.failure:
        print(f"run_failure={report.failure}")
    for task in report.tasks:
        print(f"task_id={task.task_id} status={task.status} failure={task.failure}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-seconds", type=int, default=_TIMEOUT_SECONDS)
    modes = parser.add_subparsers(dest="mode", required=True)

    task = modes.add_parser("task", help="Upload, build, and diagnose one local task")
    task.add_argument("--agent-id", required=True)
    task.add_argument("--prompt", default="Describe playing music in a friendly way.")

    benchmark = modes.add_parser("benchmark", help="Diagnose an uploaded benchmark")
    benchmark.add_argument("--bench-id", required=True)

    args = parser.parse_args()
    client = Client()
    if args.mode == "task":
        spec = TaskSpec(
            name="t-factory/diagnostic",
            split="test",
            runtime=DockerfileBuild("runtime/Dockerfile"),
            run_command="python -u /opt/t_factory/t_factory_harness.py",
            env_vars={"USER_PROMPT": args.prompt},
        )
        started = start_task_diagnostic(
            client,
            spec,
            agent_id=args.agent_id,
            root=_T_FACTORY_ROOT,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        started = client.diagnostics.start_benchmark(bench_id=args.bench_id)

    diagnostic_id = started.benchmark_diagnostic_id
    print(f"benchmark_diagnostic_id={diagnostic_id}", flush=True)
    print_report(client, diagnostic_id, args.timeout_seconds)


if __name__ == "__main__":
    main()
