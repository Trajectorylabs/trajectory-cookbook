# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Diagnose one task or an already uploaded benchmark through the Trajectory SDK."""

import argparse
import time

from trajectory import Client, TaskSpec
from trajectory.lib import RuntimeRef

_POLL_SECONDS = 5
_TIMEOUT_SECONDS = 45 * 60


def wait_for_images(client: Client, bench_id: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        images = client.benchmarks.images.list(bench_id)
        failed = [image for image in images.images if image.build_status == "failed"]
        if failed:
            raise RuntimeError(
                f"Image build failed: {[image.failure_message for image in failed]}"
            )
        if images.images and all(image.build_status == "ready" for image in images.images):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Image build timed out for {bench_id}")
        time.sleep(_POLL_SECONDS)


def diagnose_benchmark(client: Client, bench_id: str, timeout_seconds: int) -> None:
    started = client.diagnostics.start_benchmark(bench_id=bench_id)
    diagnostic_id = started.benchmark_diagnostic_id
    print(f"benchmark_diagnostic_id={diagnostic_id}", flush=True)

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

    task = modes.add_parser("task", help="Ingest and diagnose one task")
    task.add_argument("--agent-id", required=True)
    task.add_argument("--runtime-id", required=True)
    task.add_argument("--run-command", required=True)
    task.add_argument("--task-name", default="diagnostic-task")

    benchmark = modes.add_parser("benchmark", help="Diagnose an uploaded benchmark")
    benchmark.add_argument("--bench-id", required=True)

    args = parser.parse_args()
    client = Client()
    if args.mode == "task":
        spec = TaskSpec(
            name=args.task_name,
            split="test",
            runtime=RuntimeRef(args.runtime_id),
            run_command=args.run_command,
        )
        images = client.diagnostics.ingest_task(agent_id=args.agent_id, task=spec)
        print(f"bench_id={images.bench_id}", flush=True)
        wait_for_images(client, images.bench_id, args.timeout_seconds)
        bench_id = images.bench_id
    else:
        bench_id = args.bench_id

    diagnose_benchmark(client, bench_id, args.timeout_seconds)


if __name__ == "__main__":
    main()
