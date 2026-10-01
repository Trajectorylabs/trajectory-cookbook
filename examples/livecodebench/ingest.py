# /// script
# dependencies = ["trajectory-sdk==0.6.20"]
# ///
"""Upload the existing LiveCodeBench adapter's prepared package through the public SDK."""

import argparse
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client
from trajectory.lib import get_operation, submit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--agent-id")
    parser.add_argument("--idempotency-key", required=True)
    args = parser.parse_args()
    manifest = BenchmarkSpec.model_validate_json(
        (args.prepared / "manifest.json").read_text()
    )
    if len(manifest.tasks) != 400:
        raise ValueError("Expected the complete 400-task release_v1 package")

    receipt_path = args.prepared / "ingestion.json"
    with Client() as client:
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            operation = get_operation(client, receipt["operation_id"])
        else:
            operation = submit(
                client,
                manifest,
                root=args.prepared / "package",
                agent_id=args.agent_id,
                build_images=True,
                idempotency_key=args.idempotency_key,
                progress=lambda progress: print(json.dumps(progress), flush=True),
            )
            receipt_path.write_text(
                json.dumps({"operation_id": operation.id}, indent=2) + "\n"
            )
        print(f"operation_id={operation.id}", flush=True)
        try:
            result = operation.result(timeout=45 * 60, poll_interval=10)
        finally:
            if operation.status is not None:
                receipt_path.write_text(
                    operation.status.model_dump_json(indent=2) + "\n"
                )
        if result.status.registered_tasks != 400 or not result.status.ready:
            raise RuntimeError(f"Incomplete ingestion: {result.status}")
        print(f"bench_id={result.status.bench_id}", flush=True)
        print(result.status.model_dump_json(indent=2), flush=True)


if __name__ == "__main__":
    main()
