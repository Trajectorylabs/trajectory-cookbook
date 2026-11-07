"""Upload WeightProgramBench using the Trajectory SDK."""

import argparse
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push

RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"


def build_spec() -> BenchmarkSpec:
    rows = [json.loads(line) for line in (RUNTIME_ROOT / "tasks.jsonl").read_text().splitlines()]
    return BenchmarkSpec(
        name="WeightProgramBench",
        description=(
            "Write a neural-network policy's weights in one tool-free response. "
            "317 training games: 51 Atari-style and 266 Crisp-inspired conversions; "
            "five original MinAtar games held out. Each policy plays 1000 seeded episodes."
        ),
        visibility="private",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=row["name"],
                split=row["split"],
                run_command=f"python -u /opt/weight_program_bench/harness.py --task_id {index}",
                env_resources={"cpus": 2, "memory_mb": 2048, "gpus": 0, "network_mode": "no-network"},
                tags=["weight-programming"],
            )
            for index, row in enumerate(rows)
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True, help="An existing agent in your organization")
    args = parser.parse_args()
    with Client() as client:
        result = push(client, build_spec(), agent_name=args.agent_name,
                      root=RUNTIME_ROOT, build_images=True)
        print(json.dumps({"bench_id": result.bench_id, "task_count": result.task_count}))
