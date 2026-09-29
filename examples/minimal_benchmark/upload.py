# /// script
# dependencies = ["trajectory-sdk==0.7.1"]
# ///
"""Upload two arithmetic tasks and build their shared runtime image."""

import argparse
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images


def build_benchmark() -> BenchmarkSpec:
    return BenchmarkSpec(
        name="Minimal arithmetic benchmark",
        description="Two tasks demonstrating upload, model calls and grade reporting.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name="addition-train",
                split="train",
                run_command="python /app/harness.py",
                env_vars={"QUESTION": "What is 2 + 3?", "EXPECTED_ANSWER": "5"},
            ),
            TaskSpec(
                name="addition-test",
                split="test",
                run_command="python /app/harness.py",
                env_vars={"QUESTION": "What is 4 + 7?", "EXPECTED_ANSWER": "11"},
            ),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    args = parser.parse_args()
    client = Client()
    result = push(
        client,
        build_benchmark(),
        agent_id=args.agent_id,
        root=Path(__file__).parent,
    )
    print(f"bench_id={result.bench_id}", flush=True)
    wait_for_benchmark_images(client, result.bench_id)
    print("Runtime image ready.", flush=True)


if __name__ == "__main__":
    main()
