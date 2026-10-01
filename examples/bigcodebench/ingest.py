# /// script
# dependencies = ["trajectory-sdk==0.6.20"]
# ///
"""Submit the original BigCodeBench export using the public SDK ingestion workflow."""

import argparse
import hashlib
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client
from trajectory.lib import get_operation, submit

ROOT = Path(__file__).resolve().parent


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / ".work/prepared")
    args = parser.parse_args()
    root = args.prepared.resolve()
    evidence = json.loads((root / "source-evidence.json").read_text())
    manifest_bytes = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != evidence["manifest_sha256"]:
        raise ValueError("Manifest differs from the source export")
    for relative, digest in evidence["package_files"].items():
        if hashlib.sha256((root / "package" / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Exported package changed: {relative}")
    manifest = BenchmarkSpec.model_validate_json(manifest_bytes)
    expected = {f"BigCodeBench/{i}" for i in range(1140)}
    if (len(manifest.tasks) != 1140 or {task.name for task in manifest.tasks} != expected
            or {task.split for task in manifest.tasks} != {"test"}):
        raise ValueError("Expected all 1,140 original TEST tasks")
    digest = hashlib.sha256((root / "source-evidence.json").read_bytes()).hexdigest()
    receipt = root / "ingestion.json"
    with Client() as client:
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["source_evidence_sha256"] != digest:
                raise ValueError("Source changed since submission; use a fresh output directory")
            operation = get_operation(client, saved["operation_id"])
        else:
            operation = submit(
                client, manifest, root=root / "package", build_images=True,
                idempotency_key=f"bigcodebench-{digest}",
                progress=lambda status: print(status, flush=True),
            )
            write_json(receipt, {
                "operation_id": operation.id, "source_evidence_sha256": digest,
            })
        print(f"operation_id={operation.id}", flush=True)
        result = operation.result(timeout=None, poll_interval=10)
        tasks = list(result.tasks())
        if (len(tasks) != len(expected) or {task["name"] for task in tasks} != expected
                or result.status.registered_tasks != len(expected) or result.status.failure_count):
            raise RuntimeError("Ingestion did not acknowledge all 1,140 tasks exactly once")
        if (not result.status.ready or result.status.total_runtimes != 1
                or result.status.ready_runtimes != 1):
            raise RuntimeError("The BigCodeBench runtime is not ready")
        write_json(root / "ingestion-results.json", {
            "status": result.status.model_dump(mode="json"),
            "tasks": tasks, "runtimes": list(result.runtimes()),
        })
        print(f"bench_id={result.status.bench_id}; verified {len(tasks)} tasks", flush=True)


if __name__ == "__main__":
    main()
