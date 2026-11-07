# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.10.13"]
# ///
"""Package school-math tasks, diagnose one, or register a train/test spec."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push, start_task_diagnostic
from trajectory.lib.platform_links import PlatformLinks

ROOT = Path(__file__).parent
COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 1024,
    "max_turns_per_trajectory": 8,
}


def package_tasks(data_dir: Path, package_dir: Path) -> list[dict]:
    if package_dir.exists():
        raise FileExistsError(f"Choose a new package directory: {package_dir}")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    if manifest["source_commit"] != COMMIT or set(manifest["modules"]) != {
        "algebra__linear_1d",
        "polynomials__evaluate",
    }:
        raise ValueError("Unexpected source revision or modules")
    seen, seeds, rows = set(), set(), []
    for split in ("train", "validation", "test"):
        payload = (data_dir / f"{split}.jsonl").read_bytes()
        description = manifest["partitions"][split]
        if hashlib.sha256(payload).hexdigest() != description["sha256"]:
            raise ValueError(f"Data checksum mismatch: {split}")
        partition = [json.loads(line) for line in payload.decode().splitlines()]
        if len(partition) != description["count"] or not partition:
            raise ValueError(f"Data count mismatch or empty split: {split}")
        for index, row in enumerate(partition):
            question = " ".join(row["question"].lower().split())
            if (
                row["id"] != f"{split}_{index:05d}"
                or row["split"] != split
                or question in seen
                or row["generation_seed"] in seeds
            ):
                raise ValueError(
                    "Duplicate task, generation seed or incorrect partition"
                )
            if row["source_module"] not in manifest["modules"] or row[
                "source_regime"
            ] != ("train" if split == "train" else "interpolate"):
                raise ValueError("Unexpected task module/regime")
            seen.add(question)
            seeds.add(row["generation_seed"])
        if split != "validation":
            rows.extend(partition)
    license_data = (data_dir / "LICENSE-DeepMind").read_bytes()
    if hashlib.sha256(license_data).hexdigest() != manifest["license_sha256"]:
        raise ValueError("License checksum mismatch")
    package_dir.mkdir(parents=True)
    for name in ("Dockerfile", "math_harness.py", "NOTICE"):
        shutil.copyfile(ROOT / "runtime" / name, package_dir / name)
    shutil.copyfile(data_dir / "manifest.json", package_dir / "source_manifest.json")
    shutil.copyfile(data_dir / "LICENSE-DeepMind", package_dir / "LICENSE-DeepMind")
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=package_dir, delete=False
    ) as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
        temporary = handle.name
    os.replace(temporary, package_dir / "tasks.jsonl")
    return rows


def build_benchmark(name: str, rows: list[dict]) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        description=f"Generated DeepMind numeric subset; source {COMMIT}; exact rational accuracy; offline validation excluded.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"school_math/{row['id']}",
                split=row["split"],
                run_command=f"python -u /opt/school_math/math_harness.py --task-id {row['id']}",
                tags=["school-math", row["source_module"]],
            )
            for row in rows
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
    parser.add_argument("--name", default="school-math-generated-numeric")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    if args.action != "package" and not args.agent_name:
        parser.error(
            "--agent-name is required for managed actions; create or choose an agent first"
        )
    rows = package_tasks(args.data_dir, args.package_dir)
    print(f"package_dir={args.package_dir} tasks={len(rows)}", flush=True)
    if args.action == "package":
        return
    client = Client()
    links = PlatformLinks.from_client(client)
    agent_id = client.agents.retrieve_by_name(args.agent_name).agent_id
    benchmark = build_benchmark(args.name, rows)
    if args.action == "ingest":
        operation = start_push(
            client,
            benchmark,
            agent_name=args.agent_name,
            root=args.package_dir,
            build_images=True,
        )
        print(f"operation_id={operation.id}", flush=True)
        result = operation.result(timeout=1800)
        print(
            f"bench_id={result.status.bench_id} {links.spec(agent_id, result.status.bench_id)}",
            flush=True,
        )
        return
    task = benchmark.tasks[0].model_copy(
        update={"runtime": DockerfileBuild("Dockerfile")}
    )
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
    print(links.spec(agent_id, diagnostic.bench_id), flush=True)
    while client.diagnostics.get_status(diagnostic_id).status not in {
        "completed",
        "failed",
        "cancelled",
    }:
        time.sleep(5)
    report = client.diagnostics.get_diagnostics(diagnostic_id)
    print(report.model_dump_json(indent=2), flush=True)
    print(links.evaluation(report.eval_run_id), flush=True)
    if (
        report.status != "completed"
        or report.failure
        or not report.tasks
        or any(
            task.status != "completed"
            or task.termination_reason != "ENV_DONE"
            or task.reward is None
            or task.failure
            for task in report.tasks
        )
    ):
        raise RuntimeError(f"Diagnostic {report.status}; inspect the linked report")


if __name__ == "__main__":
    main()
