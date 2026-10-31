# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Upload the public Big Finance Benchmark subset with one task per question."""

import argparse
import json
import shlex
import urllib.request
from pathlib import Path

from trajectory import BenchmarkSpec, Client, SecretRef, TaskSpec
from trajectory.lib import DockerfileBuild, push
from trajectory.types.benchmarks.task_spec import EnvResources

ROOT = Path(__file__).parent
UPSTREAM_COMMIT = "bafc769bc58032f659e4117c3d839258ce0fda40"
SUBSET_URL = (
    "https://raw.githubusercontent.com/Rogo-Technologies/big-finance-benchmark/"
    f"{UPSTREAM_COMMIT}/data/big_finance_subset.jsonl"
)
RUN_COMMAND = (
    "python -u /opt/big_finance/scripts/run_eval_set.py --trajectory "
    "--dataset /opt/big_finance/data/big_finance_subset.jsonl "
    "--run-id trajectory --judge {judge}"
)


def load_task_ids() -> list[str]:
    with urllib.request.urlopen(SUBSET_URL) as response:
        lines = response.read().decode().splitlines()
    return [json.loads(line)["id"] for line in lines if line.strip()]


def build_benchmark(
    name: str, task_ids: list[str], judge: str, judge_secret: str
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        description="Public Big Finance Benchmark subset graded by the upstream rubric judge.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"big-finance/{task_id}",
                split="test",
                run_command=RUN_COMMAND.format(judge=shlex.quote(judge)),
                env_vars={
                    "BFB_TASK_ID": task_id,
                    judge_secret: SecretRef(secret_ref=judge_secret),
                    # Upstream otherwise sends Vertex calls to Provisioned Throughput only.
                    "VERTEX_DISABLE_DEDICATED": "1",
                    "TAVILY_API_KEY": SecretRef(secret_ref="TAVILY_API_KEY"),
                    "SEC_EDGAR_USER_AGENT": SecretRef(
                        secret_ref="SEC_EDGAR_USER_AGENT"
                    ),
                },
                env_resources=EnvResources(network_mode="public", agent_user="root"),
            )
            for task_id in task_ids
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="big-finance-public-subset")
    parser.add_argument(
        "--judge",
        default="vertex:gemini-3.1-pro-preview",
        help="Rubric judge as provider:model, passed to the harness's --judge.",
    )
    parser.add_argument(
        "--judge-secret",
        default="VERTEXAI_CREDENTIALS",
        help="Organization secret holding the judge provider's credential.",
    )
    args = parser.parse_args()
    benchmark = build_benchmark(
        args.name, load_task_ids(), args.judge, args.judge_secret
    )
    with Client() as client:
        result = push(
            client, benchmark, agent_name=args.agent_name, root=ROOT, build_images=True
        )
    print(f"bench_id={result.bench_id}")


if __name__ == "__main__":
    main()
