# /// script
# requires-python = ">=3.12"
# dependencies = ["trajectory-sdk>=0.10.0"]
# ///
"""Upload pinned Terminal-Bench 4.0 tasks and run a diagnostic."""

import argparse
import re
import shutil
import subprocess
import tempfile
import time
import tomllib
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push

SOURCE = "https://github.com/harbor-framework/terminal-bench.git"
SOURCE_COMMIT = "452bf305c6daa62fc59061d22133a7cbc7c1572e"
EXAMPLE = Path(__file__).resolve().parent
DEFAULT_TASK = "music-harmony"
MODEL = "Qwen/Qwen3.6-35B-A3B"


def source_checkout(root: Path, tasks: list[str]) -> Path:
    checkout = root / "source"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "sparse-checkout",
            "set",
            *[f"tasks/{task}" for task in tasks],
        ],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "fetch",
            "-q",
            "--depth=1",
            "--filter=blob:none",
            SOURCE,
            SOURCE_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )
    if (
        subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        != SOURCE_COMMIT
    ):
        raise RuntimeError("Terminal-Bench source commit did not match the pin")
    return checkout


def stage(
    root: Path, tasks: list[str], timeout: int | None
) -> tuple[Path, BenchmarkSpec]:
    source = source_checkout(root, tasks)
    manifest = tomllib.loads(
        subprocess.check_output(
            ["git", "-C", str(source), "show", "HEAD:tasks/dataset.toml"], text=True
        )
    )
    published = {
        item["name"].removeprefix("terminal-bench/") for item in manifest["tasks"]
    }
    context = root / "context"
    (context / "tasks").mkdir(parents=True)
    shutil.copy2(source / "LICENSE", context / "LICENSE")
    for name in ("Dockerfile", "run_task.py"):
        shutil.copy2(EXAMPLE / "runtime" / name, context / name)

    specs = []
    for name in tasks:
        if name not in published:
            raise RuntimeError(f"Task {name} is not published in Terminal-Bench 4.0")
        source_task = source / "tasks" / name
        if not (source_task / "task.toml").is_file():
            raise RuntimeError(f"Task {name} is missing from pinned source")
        shutil.copytree(
            source_task,
            context / "tasks" / name,
            symlinks=True,
            ignore=lambda directory, _, root=source_task: (
                {"solution", "cheat"} if Path(directory) == root else set()
            ),
        )
        config = tomllib.loads((source_task / "task.toml").read_text())
        resources = config["environment"]
        if resources.get("gpus", 0):
            raise RuntimeError(
                f"{name} needs a GPU and nested Docker; this example is CPU-only"
            )
        command = f"python -u /opt/terminal_bench_4/run_task.py {name}"
        if timeout is not None:
            command += f" --agent-timeout {timeout}"
        specs.append(
            TaskSpec(
                name=f"terminal-bench-4/{name}",
                split="test",
                run_command=command,
                env_resources={
                    "docker_engine": True,
                    "network_mode": "public",
                    "cpus": max(4, resources["cpus"] + 2),
                    "memory_mb": max(8192, resources["memory_mb"] + 4096),
                },
                spec={"source_commit": SOURCE_COMMIT, "source_task": name},
            )
        )
    return context, BenchmarkSpec(
        name="terminal-bench-4",
        description="Terminal-Bench 4.0 tasks, run by Harbor with the original verifier.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=specs,
    )


def diagnose(bench_id: str, timeout: int | None) -> None:
    client = Client()
    options = {
        "reasoning_effort": "on",
        "max_turns_per_trajectory": 300,
        "max_output_tokens_per_turn": 8192,
        "max_output_tokens_per_trajectory": 131072,
        "trajectory_timeout_seconds": min(36000, (timeout or 28800) + 1800),
    }
    diagnostic_id = client.diagnostics.start_benchmark(
        bench_id=bench_id, base_model_slug=MODEL, options=options
    ).benchmark_diagnostic_id
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while (status := client.diagnostics.get_status(diagnostic_id).status) not in {
        "completed",
        "failed",
        "cancelled",
    }:
        print(f"diagnostic_status={status}", flush=True)
        time.sleep(10)
    report = client.diagnostics.get_diagnostics(diagnostic_id)
    print(report.to_json(), flush=True)
    if (
        report.status != "completed"
        or report.failure
        or not report.tasks
        or any(
            task.status != "completed" or task.failure or task.reward is None
            for task in report.tasks
        )
    ):
        raise RuntimeError(f"Diagnostic {diagnostic_id} failed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--task", action="append", metavar="NAME")
    parser.add_argument(
        "--agent-timeout",
        type=int,
        help="Cap the original 8-hour agent deadline for a smoke test",
    )
    args = parser.parse_args()
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        parser.error("--agent-timeout must be positive")
    tasks = sorted(set(args.task or [DEFAULT_TASK]))
    if any(not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", task) for task in tasks):
        parser.error("--task must be an upstream task name")
    with tempfile.TemporaryDirectory() as workspace:
        context, spec = stage(Path(workspace), tasks, args.agent_timeout)
        operation = start_push(
            Client(), spec, agent_name=args.agent_name, root=context, build_images=True
        )
        print(f"operation_id={operation.id} tasks={len(tasks)}", flush=True)
        result = operation.result(timeout=2700)
    bench_id = result.status.bench_id
    print(f"bench_id={bench_id}", flush=True)
    diagnose(bench_id, args.agent_timeout)


if __name__ == "__main__":
    main()
