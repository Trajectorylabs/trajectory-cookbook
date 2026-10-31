# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.32"]
# ///
"""Upload the public Big Finance Benchmark subset with one task per question."""

import argparse
import io
import json
import shlex
import shutil
import subprocess
import tarfile
import tempfile
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
UPSTREAM_PATHS = (
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "big_finance_harness",
    "scripts/run_eval_set.py",
    "data/big_finance_subset.jsonl",
)
RUN_COMMAND = (
    "python -u /opt/big_finance/scripts/run_eval_set.py --trajectory "
    "--dataset /opt/big_finance/data/big_finance_subset.jsonl "
    "--run-id trajectory --max-steps 30 --judge {judge}"
)


def parse_task_ids(subset: str) -> list[str]:
    return [json.loads(line)["id"] for line in subset.splitlines() if line.strip()]


def download_task_ids() -> list[str]:
    with urllib.request.urlopen(SUBSET_URL) as response:
        return parse_task_ids(response.read().decode())


def git(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(checkout), *args], capture_output=True, check=check
    )


def stage_checkout(checkout: Path, root: Path) -> list[str]:
    """Copy the runtime and the pinned commit's harness files from `checkout` into `root`."""
    head = git(checkout, "rev-parse", "HEAD", check=False)
    if head.returncode != 0:
        raise SystemExit(
            f"{checkout} is not a git checkout of the Big Finance harness."
        )
    if (commit := head.stdout.decode().strip()) != UPSTREAM_COMMIT:
        raise SystemExit(
            f"{checkout} is at {commit}, but the overlay is pinned to {UPSTREAM_COMMIT}. "
            f"Run `git -C {checkout} checkout {UPSTREAM_COMMIT}`, or omit --upstream."
        )
    if changed := git(checkout, "status", "--porcelain", "--", *UPSTREAM_PATHS).stdout:
        print(
            f"Packaging the committed files; these local changes are left out:\n{changed.decode()}"
        )
    shutil.copytree(
        ROOT / "runtime", root / "runtime", ignore=shutil.ignore_patterns("__pycache__")
    )
    shutil.copy(ROOT / ".dockerignore", root / ".dockerignore")
    archive = git(checkout, "archive", UPSTREAM_COMMIT, *UPSTREAM_PATHS)
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        tar.extractall(root / "runtime" / "upstream", filter="data")
    subset = root / "runtime" / "upstream" / "data" / "big_finance_subset.jsonl"
    return parse_task_ids(subset.read_text())


def build_benchmark(
    name: str, task_ids: list[str], judge: str, judge_secret: str, dockerfile: str
) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        description="Public Big Finance Benchmark subset graded by the upstream rubric judge.",
        runtime=DockerfileBuild(dockerfile),
        tasks=[
            TaskSpec(
                name=f"big-finance/{task_id}",
                split="test",
                run_command=RUN_COMMAND.format(judge=shlex.quote(judge)),
                env_vars={
                    "BFB_TASK_ID": task_id,
                    judge_secret: SecretRef(secret_ref=judge_secret),
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
        default="openai:gpt-5.4-mini",
        help="Rubric judge as provider:model, passed to the harness's --judge.",
    )
    parser.add_argument(
        "--judge-secret",
        default="OPENAI_API_KEY",
        help="Organization secret holding the judge provider's API key.",
    )
    parser.add_argument(
        "--upstream",
        type=Path,
        help="Package this checkout of the harness at the pinned commit instead of downloading it.",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as staging:
        if args.upstream:
            root, dockerfile = Path(staging), "runtime/checkout.Dockerfile"
            task_ids = stage_checkout(args.upstream, root)
        else:
            root, dockerfile, task_ids = ROOT, "runtime/Dockerfile", download_task_ids()
        benchmark = build_benchmark(
            args.name, task_ids, args.judge, args.judge_secret, dockerfile
        )
        with Client() as client:
            result = push(
                client,
                benchmark,
                agent_name=args.agent_name,
                root=root,
                build_images=True,
            )
    print(f"bench_id={result.bench_id}")


if __name__ == "__main__":
    main()
