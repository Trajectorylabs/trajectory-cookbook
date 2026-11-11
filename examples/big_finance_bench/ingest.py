# /// script
# requires-python = ">=3.12"
# dependencies = ["trajectory-sdk>=0.10.1"]
# ///
"""Upload pinned BigFinanceBench questions and run a diagnostic."""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, start_push

SOURCE = "https://github.com/Rogo-Technologies/big-finance-benchmark.git"
SOURCE_COMMIT = "3b13eecc4cacb4cfa13ac2b7c111d14ce127dd71"
EXAMPLE = Path(__file__).resolve().parent
DATASET = "data/big_finance_subset.jsonl"
# The credentials upstream's default tools read.
TOOL_SECRETS = ("TAVILY_API_KEY", "SEC_EDGAR_USER_AGENT")
MODEL = "Qwen/Qwen3.6-35B-A3B"
# Upstream's 50-step and 65,536-token defaults; Qwen 3.6's context bounds the total.
OPTIONS = {
    "reasoning_effort": "on",
    "max_turns_per_trajectory": 50,
    "max_output_tokens_per_turn": 65536,
    "max_output_tokens_per_trajectory": 262144,
    "trajectory_timeout_seconds": 3600,
}


def source_checkout(root: Path) -> Path:
    checkout = root / "source"
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "sparse-checkout",
            "set",
            "big_finance_harness",
            "data",
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
        raise RuntimeError("BigFinanceBench source commit did not match the pin")
    shutil.rmtree(checkout / ".git")
    return checkout


def stage(
    root: Path, tasks: list[str] | None, judge_model: str | None
) -> tuple[Path, BenchmarkSpec]:
    context = root / "context"
    context.mkdir()
    source = source_checkout(context)
    for name in ("Dockerfile", "python_exec.patch", "run_task.py"):
        shutil.copy2(EXAMPLE / "runtime" / name, context / name)
    published = [
        json.loads(line)["id"] for line in (source / DATASET).read_text().splitlines()
    ]
    if unknown := set(tasks or []) - set(published):
        raise RuntimeError(
            f"Not in the public BigFinanceBench subset: {sorted(unknown)}"
        )
    judge_secret = "OPENROUTER_API_KEY" if judge_model else "OPENAI_API_KEY"
    env_vars = {
        secret: SecretRef(secret_ref=secret) for secret in (judge_secret, *TOOL_SECRETS)
    }
    if judge_model:
        env_vars["BIG_FINANCE_BENCH_JUDGE_MODEL"] = judge_model
    specs = [
        TaskSpec(
            name=f"big-finance-bench/{name}",
            split="test",
            run_command=f"python -u /opt/big_finance_bench/run_task.py {name}",
            env_vars=env_vars,
            env_resources={"network_mode": "public"},
            spec={"source_commit": SOURCE_COMMIT, "source_id": name},
        )
        for name in sorted(tasks or published)
    ]
    return context, BenchmarkSpec(
        name="big-finance-bench",
        description="BigFinanceBench public subset, run by the upstream agent and rubric grader.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=specs,
    )


def diagnose(bench_id: str) -> None:
    client = Client()
    diagnostic_id = client.diagnostics.start_benchmark(
        bench_id=bench_id, base_model_slug=MODEL, options=OPTIONS
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
    parser.add_argument(
        "--task", action="append", metavar="ID", help="Upload only this question"
    )
    parser.add_argument(
        "--judge-model",
        metavar="openrouter/MODEL",
        help="Serve upstream's openai:gpt-5.5 judge through OpenRouter",
    )
    args = parser.parse_args()
    if args.judge_model and not args.judge_model.startswith("openrouter/"):
        parser.error("--judge-model must be an openrouter/ model")
    with tempfile.TemporaryDirectory() as workspace:
        context, spec = stage(Path(workspace), args.task, args.judge_model)
        operation = start_push(
            Client(), spec, agent_name=args.agent_name, root=context, build_images=True
        )
        print(f"operation_id={operation.id} tasks={len(spec.tasks)}", flush=True)
        result = operation.result(timeout=2700)
    bench_id = result.status.bench_id
    print(f"bench_id={bench_id}", flush=True)
    diagnose(bench_id)


if __name__ == "__main__":
    main()
