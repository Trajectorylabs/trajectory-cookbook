# /// script
# dependencies = ["trajectory-sdk==0.6.20"]
# ///
"""Preview evaluation requests; add --launch to submit through the public SDK."""

import argparse
import hashlib
import json

from trajectory import Client

MODEL = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-id", action="append", required=True)
    parser.add_argument(
        "--run-key", required=True, help="Reuse for retries; change for a new run"
    )
    parser.add_argument(
        "--max-samples", type=int, default=2, help="TEST task cap per benchmark"
    )
    parser.add_argument("--max-active-rollouts", type=int, default=2)
    parser.add_argument("--collection-timeout-seconds", type=int, default=7200)
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if (
        min(args.max_samples, args.max_active_rollouts, args.collection_timeout_seconds)
        <= 0
    ):
        parser.error("Sample, concurrency, and timeout limits must be positive")
    if len(set(args.bench_id)) != len(args.bench_id):
        parser.error("Benchmark IDs must be unique")

    requests = []
    for bench_id in args.bench_id:
        request = {
            "bench_id": bench_id,
            "model_slug": MODEL,
            "display_name": f"Nemotron Lightning / {args.run_key}",
            "execution_timeout_seconds": 7200,
            "eval_options": {
                "max_samples": args.max_samples,
                "pass_at_k": 1,
                "max_active_rollouts": args.max_active_rollouts,
                "eval_collection_timeout": args.collection_timeout_seconds,
                "max_output_tokens_per_step": 32768,
                "disable_thinking": False,
            },
        }
        digest = hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()
        request["idempotency_key"] = f"cookbook-eval-{digest}"
        requests.append(request)
    print(json.dumps(requests, indent=2), flush=True)
    if not args.launch:
        return
    with Client() as client:
        for request in requests:
            result = client.evals.start(**request)
            print(
                json.dumps(
                    {
                        "bench_id": request["bench_id"],
                        "eval_run_id": result.eval_run_id,
                        "idempotency_key": request["idempotency_key"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
