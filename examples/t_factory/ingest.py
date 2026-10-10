# /// script
# dependencies = ["trajectory-sdk>=0.9.14"]
# ///
"""Ingest the prompted T-starting-word task for evaluation and training."""

import argparse
import time
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, start_task_diagnostic, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUN_COMMAND = "python -u /opt/t_factory/t_factory_harness.py"
_BUILD_TIMEOUT_SECONDS = 45 * 60
_TRAIN_TASKS = 128
_TEST_TASKS = 64
_TOPICS = (
    "rainy weather",
    "simple arithmetic",
    "ocean animals",
    "growing a garden",
    "playing music",
    "ancient history",
    "home cooking",
    "riding bicycles",
    "public libraries",
    "healthy sleep",
    "friendly robots",
    "distant stars",
    "lasting friendship",
    "watercolor painting",
    "reading maps",
    "changing seasons",
)
_TEMPLATES = (
    "Do you enjoy {topic}? Why?",
    "Could {topic} make someone smile? Why?",
    "How can {topic} add joy?",
    "Share your view on {topic}.",
    "How do you feel about {topic}?",
    "What is your opinion of {topic}?",
    "Why do people enjoy {topic}?",
    "What would you tell a friend about {topic}?",
    "Explain why {topic} can be useful.",
    "Explain one important idea involving {topic}.",
    "Describe {topic} in a friendly way.",
    "Describe one surprising side of {topic}.",
)


def build_dataset() -> dict[str, list[str]]:
    prompts = [template.format(topic=topic) for template in _TEMPLATES for topic in _TOPICS]
    ordered = [prompts[(index * 37) % len(prompts)] for index in range(_TRAIN_TASKS + _TEST_TASKS)]
    return {
        "train": ordered[:_TRAIN_TASKS],
        "test": ordered[_TRAIN_TASKS:],
    }


def build_benchmark(name: str, small: bool = False) -> BenchmarkSpec:
    dataset = build_dataset()
    if small:
        dataset = {split: prompts[:1] for split, prompts in dataset.items()}
    return BenchmarkSpec(
        name=name,
        family="t-factory",
        description="Respond normally while maximizing the fraction of T-starting words.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"t-word-density/{split}_{index:04d}",
                split=split,
                run_command=_RUN_COMMAND,
                env_vars={"USER_PROMPT": prompt},
                tags=["t-word-density"],
            )
            for split, prompts in dataset.items()
            for index, prompt in enumerate(prompts)
        ],
    )


def ingest(name: str, agent_name: str, skip_build: bool, small: bool = False) -> str:
    client = Client()
    result = push(
        client,
        build_benchmark(name, small),
        agent_name=agent_name,
        root=_ROOT,
    )
    print(f"agent_name={agent_name}", flush=True)
    print(f"bench_id={result.bench_id}", flush=True)
    if not skip_build:
        wait_for_benchmark_images(
            client,
            result.bench_id,
            timeout_seconds=_BUILD_TIMEOUT_SECONDS,
        )
    return result.bench_id


def diagnose(agent_name: str) -> None:
    client = Client()
    benchmark = build_benchmark("t-factory-diagnostic", small=True)
    task = benchmark.tasks[0]
    task.runtime = benchmark.runtime
    diagnostic = start_task_diagnostic(
        client,
        task,
        agent_name=agent_name,
        root=_ROOT,
        base_model_slug="Qwen/Qwen3.5-4B",
        timeout_seconds=_BUILD_TIMEOUT_SECONDS,
    )
    diagnostic_id = diagnostic.benchmark_diagnostic_id
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    while True:
        status = client.diagnostics.get_status(diagnostic_id)
        print(f"diagnostic_status={status.status}", flush=True)
        if status.status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(5)

    result = client.diagnostics.get_diagnostics(diagnostic_id)
    print(result.to_json(), flush=True)
    if (
        result.status != "completed"
        or result.failure is not None
        or len(result.tasks) != 1
        or any(task.status != "completed" or task.failure is not None for task in result.tasks)
    ):
        raise RuntimeError(
            f"Task diagnostic {diagnostic_id} did not pass; inspect the report above"
        )
    evaluation = client.evals.runs.retrieve(result.eval_run_id)
    if evaluation.reward_mean is None:
        raise RuntimeError(f"Task diagnostic {diagnostic_id} completed without a recorded reward")
    print(f"diagnostic_reward={evaluation.reward_mean}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name")
    parser.add_argument(
        "--small", action="store_true", help="Upload one training task and one held-out test task"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--skip-build", action="store_true")
    mode.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Validate one task before uploading a benchmark",
    )
    args = parser.parse_args()
    if args.diagnose_only:
        diagnose(args.agent_name)
    else:
        name = args.name or ("t-factory-small" if args.small else "t-factory")
        ingest(name, args.agent_name, args.skip_build, args.small)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
