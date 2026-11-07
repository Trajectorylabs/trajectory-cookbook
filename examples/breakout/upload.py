"""Upload the fixed Breakout weight-space benchmark to your agent."""

import argparse
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push

RUNTIME_ROOT = Path(__file__).resolve().parent / "runtime"


def build_spec() -> BenchmarkSpec:
    rows = [json.loads(line) for line in (RUNTIME_ROOT / "tasks.jsonl").read_text().splitlines()]
    return BenchmarkSpec(
        name="breakout-weight-space",
        description=(
            "Write a 600-32-32-6 tanh Breakout policy in one tool-free reply. "
            "One fixed prompt, 32 training seed bundles and one held-out bundle. "
            "Reward is mean game score, declared on [0, 10000]; invalid replies score zero."
        ),
        visibility="private",
        runtime=DockerfileBuild("Dockerfile"),
        tasks=[
            TaskSpec(
                name=row["name"],
                split=row["split"],
                run_command=f"python -u /opt/breakout/breakout_harness.py --task_id {index}",
                env_resources={"cpus": 2, "memory_mb": 2048, "gpus": 0, "network_mode": "no-network"},
                tags=["weight-space", "breakout", row["split"]],
            )
            for index, row in enumerate(rows)
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True, help="An existing agent in your organization")
    args = parser.parse_args()
    with Client() as client:
        result = push(
            client,
            build_spec(),
            agent_name=args.agent_name,
            root=RUNTIME_ROOT,
            build_images=True,
        )
        print(json.dumps({"bench_id": result.bench_id, "task_count": result.task_count}))
