# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.9.37", "chess==1.11.2", "zstandard>=0.25.0"]
# ///
"""Upload matched reasoning and Python variants of the balanced chess dataset."""

import argparse
import json
import time
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push, start_task_diagnostic

from dataset import audit, evaluation_subset

ROOT = Path(__file__).parent
MODEL = "Qwen/Qwen3.5-4B"
OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 4000,
    "max_output_tokens_per_trajectory": 204000,
    "max_turns_per_trajectory": 104,
    "trajectory_timeout_seconds": 3600,
}


def task(record, variant):
    return TaskSpec(
        name=f"chess/{variant}/mateIn{record['mate_in']}/{record['puzzle_id']}",
        split=record["split"],
        run_command=f"python -u /opt/chess/{variant}_harness.py",
        env_vars={"CHESS_VARIANT": variant, "CHESS_PUZZLE": json.dumps(record)},
        tags=["chess", variant, f"mateIn{record['mate_in']}", record["rating_band"]],
    )


def build_benchmark(records, variant, name=None):
    if variant not in {"reasoning", "python"}:
        raise ValueError(variant)
    return BenchmarkSpec(
        name=name or f"chess-mates-{variant}",
        description=f"Lichess forcing mates in 1–5; balanced ratings 0–3000; {variant}.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[task(r, variant) for r in records],
    )


def wait_diagnostic(client, diagnostic_id):
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while client.diagnostics.get_status(diagnostic_id).status not in {
        "completed",
        "failed",
        "cancelled",
    }:
        time.sleep(5)
    report = client.diagnostics.get_diagnostics(diagnostic_id)
    print(report.to_json(), flush=True)
    if (
        report.status != "completed"
        or report.failure
        or not report.tasks
        or any(
            t.status != "completed" or t.failure or t.reward is None
            for t in report.tasks
        )
    ):
        raise RuntimeError("Diagnostic failed; inspect the report")
    return report


def diagnose_task(client, record, variant, agent_name, model=MODEL):
    one = task(record, variant)
    one.runtime = DockerfileBuild("Dockerfile")
    diagnostic = start_task_diagnostic(
        client,
        one,
        agent_name=agent_name,
        root=ROOT / "runtime",
        base_model_slug=model,
        options=OPTIONS,
        timeout_seconds=2700,
    )
    return wait_diagnostic(client, diagnostic.benchmark_diagnostic_id)


def ingest(client, records, variant, agent_name, name=None):
    audit(records)
    selected = [r for r in records if r["split"] == "train"] + evaluation_subset(
        records
    )
    audit(selected, test_per_category=20)
    operation = start_push(
        client,
        build_benchmark(selected, variant, name),
        agent_name=agent_name,
        root=ROOT / "runtime",
        build_images=True,
    )
    print(f"operation_id={operation.id}", flush=True)
    result = operation.result(timeout=2700)
    print(f"bench_id={result.status.bench_id}", flush=True)
    return result.status.bench_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "puzzles.jsonl")
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--variant", choices=["reasoning", "python"], required=True)
    args = parser.parse_args()
    records = [json.loads(line) for line in args.dataset.read_text().splitlines()]
    client = Client()
    print(client.organizations.retrieve_current(), flush=True)
    diagnose_task(
        client,
        next(r for r in records if r["split"] == "train" and r["mate_in"] == 2),
        args.variant,
        args.agent_name,
    )
    bench_id = ingest(client, records, args.variant, args.agent_name)
    diagnostic = client.diagnostics.start_benchmark(
        bench_id=bench_id, base_model_slug=MODEL, options=OPTIONS
    )
    wait_diagnostic(client, diagnostic.benchmark_diagnostic_id)


if __name__ == "__main__":
    main()
