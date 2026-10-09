# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Upload a toy number-guessing benchmark with same-model context compaction."""

import argparse
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

ROOT = Path(__file__).parent


def build_benchmark(name: str) -> BenchmarkSpec:
    task_dir = ROOT / "runtime" / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    # Both splits deliberately share the {24, 42} distribution.
    for split, count in (("train", 32), ("test", 16)):
        for index in range(count):
            filename = f"{split}_{index:04d}.json"
            (task_dir / filename).write_text(
                json.dumps({"secret": (24, 42)[index % 2]})
            )
            tasks.append(
                TaskSpec(
                    name=f"number-guessing/{split}_{index:04d}",
                    split=split,
                    run_command=(
                        "python -u /opt/number_guessing/number_guessing_harness.py "
                        f"--task-file /opt/number_guessing/tasks/{filename}"
                    ),
                    tags=["number-guessing", "context-compaction"],
                )
            )
    return BenchmarkSpec(
        name=name,
        family="number-guessing",
        description="Guess 24 or 42 with same-model compaction after every three incorrect guesses.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    agent = parser.add_mutually_exclusive_group(required=True)
    agent.add_argument("--agent-name")
    agent.add_argument("--agent-id")
    parser.add_argument("--name", default="number-guessing")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    client = Client()
    result = push(
        client,
        build_benchmark(args.name),
        agent_name=args.agent_name,
        agent_id=args.agent_id,
        root=ROOT,
    )
    print(
        f"agent_name={args.agent_name}"
        if args.agent_name is not None
        else f"agent_id={args.agent_id}",
        flush=True,
    )
    print(f"bench_id={result.bench_id}", flush=True)
    if not args.skip_build:
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=45 * 60)


if __name__ == "__main__":
    main()
