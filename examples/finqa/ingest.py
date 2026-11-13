# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.10.13"]
# ///
"""Package FinQA locally, diagnose one task, or upload the prepared spec."""

import argparse
import hashlib
import json
import shlex
import shutil
import time
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push, start_task_diagnostic
from trajectory.lib.platform_links import PlatformLinks

from examples.finqa.prepare import ROOT, SOURCE_COMMIT, SOURCES, checked_bytes

OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 2048,
    "max_output_tokens_per_trajectory": 12288,
    "max_turns_per_trajectory": 12,
    "trajectory_timeout_seconds": 240,
}


def package_tasks(data_dir: Path, package_dir: Path) -> tuple[list[dict], dict]:
    if package_dir.exists():
        raise FileExistsError(f"Choose a new package directory: {package_dir}")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if (
        manifest["source_commit"] != SOURCE_COMMIT
        or manifest["source_sha256"] != SOURCES
    ):
        raise ValueError("Prepared data does not match the pinned upstream source")
    checked_bytes(
        data_dir / "upstream_evaluate.py", SOURCES["code/evaluate/evaluate.py"]
    )
    checked_bytes(data_dir / "SOURCE_LICENSE", SOURCES["LICENSE"])
    checked_bytes(data_dir / "private_tasks.jsonl", manifest["private_tasks_sha256"])
    records = [
        json.loads(line)
        for line in (data_dir / "private_tasks.jsonl").read_text().splitlines()
    ]
    if [
        {key: record[key] for key in ("task_id", "split", "source_id")}
        for record in records
    ] != manifest["selected"]:
        raise ValueError("Task selection differs from manifest")
    if {record["split"] for record in records} != {"train", "test"}:
        raise ValueError(
            "The managed package requires source train and public-test tasks"
        )
    package_dir.mkdir(parents=True, mode=0o700)
    for name in ("Dockerfile", "finqa_harness.py"):
        shutil.copyfile(ROOT / "runtime" / name, package_dir / name)
    for name in (
        "upstream_evaluate.py",
        "SOURCE_LICENSE",
        "private_tasks.jsonl",
    ):
        shutil.copyfile(data_dir / name, package_dir / name)
    shutil.copyfile(data_dir / "manifest.json", package_dir / "source_manifest.json")
    shutil.copyfile(ROOT / "NOTICE", package_dir / "NOTICE")
    return records, manifest


def build_benchmark(name: str, records: list[dict], manifest: dict) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="finqa",
        description=json.dumps(
            {
                "source_commit": SOURCE_COMMIT,
                "metric": manifest["metric"],
                "harness_sha256": hashlib.sha256(
                    (ROOT / "runtime/finqa_harness.py").read_bytes()
                ).hexdigest(),
                "adaptation": "structured steps compiled to native programs; bounded report tools; official execution accuracy",
            }
        ),
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"finqa/{record['source_id']}",
                split=record["split"],
                run_command=f"python -u /opt/finqa/finqa_harness.py --task-id {shlex.quote(record['task_id'])}",
                tags=["finqa", "financial-reasoning"],
            )
            for record in records
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action", choices=("package", "diagnose", "ingest"), default="package"
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--agent-name")
    parser.add_argument("--model")
    parser.add_argument("--name", default="finqa-report-agent")
    args = parser.parse_args()
    if args.action != "package" and not args.agent_name:
        parser.error("--agent-name is required for managed actions")
    if args.action == "diagnose" and not args.model:
        parser.error("--model is required for a diagnostic")
    records, manifest = package_tasks(args.data_dir, args.package_dir)
    benchmark = build_benchmark(args.name, records, manifest)
    print(
        f"package_dir={args.package_dir} tasks={len(records)} action={args.action}",
        flush=True,
    )
    if args.action == "package":
        return
    with Client() as client:
        links = PlatformLinks.from_client(client)
        agent_id = client.agents.retrieve_by_name(args.agent_name).agent_id
        if args.action == "ingest":
            operation = start_push(
                client,
                benchmark,
                agent_name=args.agent_name,
                root=args.package_dir,
                build_images=True,
            )
            print(f"operation_id={operation.id}", flush=True)
            result = operation.result(timeout=45 * 60)
            print(f"bench_id={result.status.bench_id}", flush=True)
            print(
                f"benchmark={links.spec(agent_id, result.status.bench_id)}", flush=True
            )
            return
        task = next(task for task in benchmark.tasks if task.split == "train")
        task.runtime = benchmark.runtime
        diagnostic = start_task_diagnostic(
            client,
            task,
            agent_name=args.agent_name,
            root=args.package_dir,
            base_model_slug=args.model,
            options=OPTIONS,
        )
        diagnostic_id = diagnostic.benchmark_diagnostic_id
        print(f"diagnostic_id={diagnostic_id}", flush=True)
        print(f"benchmark={links.spec(agent_id, diagnostic.bench_id)}", flush=True)
        previous = None
        while True:
            status = client.diagnostics.get_status(diagnostic_id).status
            if status != previous:
                print(f"diagnostic_status={status}", flush=True)
                previous = status
            if status in {"completed", "failed", "cancelled"}:
                break
            time.sleep(5)
        report = client.diagnostics.get_diagnostics(diagnostic_id)
        print(f"evaluation={links.evaluation(report.eval_run_id)}", flush=True)
        for result in report.tasks:
            print(
                f"task={result.status} ending={result.termination_reason} reward={result.reward} failure={result.failure}",
                flush=True,
            )
        if (
            report.status != "completed"
            or report.failure
            or not report.tasks
            or any(
                result.status != "completed"
                or result.termination_reason != "ENV_DONE"
                or result.reward is None
                or result.failure
                for result in report.tasks
            )
        ):
            raise RuntimeError(
                "FinQA diagnostic did not complete with a valid native grade"
            )


if __name__ == "__main__":
    main()
