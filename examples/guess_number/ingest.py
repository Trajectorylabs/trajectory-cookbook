# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Upload the number-guessing benchmark without context compaction."""

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
    for split, count in (("train", 32), ("test", 16)):
        for index in range(count):
            filename = f"{split}_{index:04d}.json"
            (task_dir / filename).write_text(
                json.dumps({"secret": (24, 42)[index % 2]})
            )
            tasks.append(
                TaskSpec(
                    name=f"guess-number/{split}_{index:04d}",
                    split=split,
                    run_command=(
                        "python -u /opt/guess_number/guess_number.py "
                        f"--task-file /opt/guess_number/tasks/{filename}"
                    ),
                    tags=["guess-number", "full-context"],
                )
            )
    return BenchmarkSpec(
        name=name,
        family="guess-number",
        description="Guess 24 or 42 while preserving the complete conversation.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--name", default="guess-number")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    client = Client()
    result = push(client, build_benchmark(args.name), agent_id=args.agent_id, root=ROOT)
    print(f"agent_id={args.agent_id}", flush=True)
    print(f"bench_id={result.bench_id}", flush=True)
    if not args.skip_build:
        wait_for_benchmark_images(client, result.bench_id, timeout_seconds=45 * 60)


if __name__ == "__main__":
    main()
