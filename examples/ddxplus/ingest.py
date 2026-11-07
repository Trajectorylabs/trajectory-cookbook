# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.10.13"]
# ///
"""Package DDXPlus tasks, diagnose a training task, or upload the spec."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, start_push, start_task_diagnostic
from trajectory.lib.platform_links import PlatformLinks

ROOT = Path(__file__).parent
OPTIONS = {
    "reasoning_effort": "off",
    "max_output_tokens_per_turn": 512,
    "max_turns_per_trajectory": 20,
    "max_output_tokens_per_trajectory": 10240,
    "trajectory_timeout_seconds": 600,
}


def package_tasks(data_dir: Path, package_dir: Path) -> list[dict]:
    manifest = json.loads((data_dir / "source_manifest.json").read_text())
    if (
        manifest["source"]
        != "https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585/2"
    ):
        raise ValueError("Expected pinned official English DDXPlus v2 data")
    if manifest.get("schema_version") != 2:
        raise ValueError(
            "DDXPlus recipe v2 requires re-preparing data with schema_version 2"
        )
    actor_conditions = json.loads((data_dir / "actor_conditions.json").read_text())
    official_conditions = json.loads((data_dir / "release_conditions.json").read_text())
    if (
        actor_conditions != manifest["actor_conditions"]
        or not actor_conditions
        or not set(actor_conditions) <= set(official_conditions)
    ):
        raise ValueError(
            "Actor catalog must match the manifest and official condition metadata"
        )
    for name, expected in manifest["selected_sha256"].items():
        if hashlib.sha256((data_dir / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Prepared data changed: {name}")
    rows = []
    identities = set()
    profiles = set()
    for split in ("train", "validate", "test"):
        split_rows = [
            json.loads(line)
            for line in (data_dir / f"{split}.jsonl").read_text().splitlines()
        ]
        if len(split_rows) != manifest["counts"][split] or not split_rows:
            raise ValueError("Manifest counts do not match data")
        if manifest["mode"] == "common-six":
            actual_counts = {
                name: sum(row["patient"]["pathology"] == name for row in split_rows)
                for name in actor_conditions
            }
            if set(actual_counts.values()) != {manifest["per_condition"][split]}:
                raise ValueError(
                    "Curriculum must cover every actor condition with its fixed quota"
                )
        for row in split_rows:
            if row["patient"]["pathology"] not in actor_conditions:
                raise ValueError("Patient pathology outside actor condition catalog")
            if (
                row["split"] != split
                or row["task_id"] != f"{split}-row-{row['source_row_index']}"
                or row["task_id"] in identities
            ):
                raise ValueError("Source split/identity mismatch")
            identities.add(row["task_id"])
            profile = dict(row["patient"])
            profile.pop("initial_evidence")
            profile.pop("differential")
            profile["evidences"] = sorted(profile["evidences"])
            fingerprint = hashlib.sha256(
                json.dumps(profile, sort_keys=True).encode()
            ).hexdigest()
            if fingerprint in profiles:
                raise ValueError(
                    "Duplicate patient profile in selected rows; choose a different offset"
                )
            profiles.add(fingerprint)
        if split != "validate":
            rows.extend(split_rows)
    package_dir.mkdir(parents=True, exist_ok=False)
    for name in ("Dockerfile", "ddxplus_harness.py"):
        atomic_copy(ROOT / "runtime" / name, package_dir / name)
    for name in (
        "actor_conditions.json",
        "release_evidences.json",
        "release_conditions.json",
        "source_manifest.json",
    ):
        atomic_copy(data_dir / name, package_dir / name)
    atomic_copy(ROOT / "NOTICE", package_dir / "NOTICE")
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=package_dir, delete=False
    ) as handle:
        handle.write("".join(json.dumps(row) + "\n" for row in rows))
        temporary = handle.name
    os.replace(temporary, package_dir / "tasks.jsonl")
    return rows


def atomic_copy(source: Path, target: Path) -> None:
    with NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = handle.name
        with source.open("rb") as reader:
            shutil.copyfileobj(reader, handle)
    os.replace(temporary, target)


def build_benchmark(rows: list[dict], name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        description="DDXPlus recipe v2 English synthetic closed-catalog static-evidence interview adaptation; deterministic top-1 pathology accuracy. Research only.",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"ddxplus/{row['task_id']}",
                split=row["split"],
                run_command=f"python -u /opt/ddxplus/ddxplus_harness.py --task-id {shlex.quote(row['task_id'])}",
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
    parser.add_argument("--name", default="ddxplus-common-six-v2")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    args = parser.parse_args()
    if args.action != "package" and not args.agent_name:
        parser.error("--agent-name is required for managed actions")
    rows = package_tasks(args.data_dir, args.package_dir)
    benchmark = build_benchmark(rows, args.name)
    print(f"package_dir={args.package_dir} tasks={len(rows)}", flush=True)
    if args.action == "package":
        return
    client = Client()
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
        print(f"operation_id={operation.id} agent={links.agent(agent_id)}", flush=True)
        result = operation.result(timeout=1800)
        print(
            f"bench_id={result.status.bench_id} benchmark={links.spec(agent_id, result.status.bench_id)}",
            flush=True,
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
    print(f"diagnostic_id={diagnostic.benchmark_diagnostic_id}", flush=True)
    print(f"benchmark={links.spec(agent_id, diagnostic.bench_id)}", flush=True)
    print(
        "Submitted only. Inspect the diagnostic report and its trajectory before ingestion or training.",
        flush=True,
    )


if __name__ == "__main__":
    main()
