# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk>=0.9.33"]
# ///
"""Ingest the prompted T-starting-word task for evaluation and training."""

import argparse
import time
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUN_COMMAND = "python -u /opt/t_factory/t_factory_harness.py"
_BUILD_TIMEOUT_SECONDS = 45 * 60
_DIAGNOSTIC_DONE = {"completed", "failed", "cancelled"}
_MODEL = "Qwen/Qwen3.5-4B"
# The settings train.py evaluates and trains with.
_RUN_OPTIONS = {"reasoning_effort": "off", "max_output_tokens_per_step": 2_048}
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
    prompts = [
        template.format(topic=topic) for template in _TEMPLATES for topic in _TOPICS
    ]
    ordered = [
        prompts[(index * 37) % len(prompts)]
        for index in range(_TRAIN_TASKS + _TEST_TASKS)
    ]
    return {
        "train": ordered[:_TRAIN_TASKS],
        "test": ordered[_TRAIN_TASKS:],
    }


def build_benchmark(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
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
            for split, prompts in build_dataset().items()
            for index, prompt in enumerate(prompts)
        ],
    )


def ingest(client: Client, name: str, agent_name: str, skip_build: bool) -> str:
    result = push(
        client,
        build_benchmark(name),
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


def benchmark_diagnostic(client: Client, bench_id: str) -> None:
    # Run the uploaded harness on a sample of tasks before evaluation or training.
    diagnostic = client.diagnostics.start_benchmark(
        bench_id=bench_id, base_model_slug=_MODEL, options=_RUN_OPTIONS
    )
    diagnostic_id = diagnostic.benchmark_diagnostic_id
    print(f"diagnostic_id={diagnostic_id}", flush=True)
    last = None
    while True:
        status = client.diagnostics.get_status(diagnostic_id).status
        if status != last:
            print(f"diagnostic_status={status}", flush=True)
            last = status
        if status in _DIAGNOSTIC_DONE:
            break
        time.sleep(10)
    result = client.diagnostics.get_diagnostics(diagnostic_id)
    print(
        f"diagnostic_status={result.status} model={result.base_model_slug}",
        flush=True,
    )
    for task in result.tasks:
        print(
            f"task_id={task.task_id} status={task.status} "
            f"ending={task.termination_reason} reward={task.reward}",
            flush=True,
        )
        if task.failure is not None:
            print(f"task_id={task.task_id} failure={task.failure}", flush=True)
    passed = (
        result.status == "completed"
        and result.failure is None
        and result.tasks
        and all(
            task.status == "completed" and task.reward is not None
            for task in result.tasks
        )
    )
    if not passed:
        raise RuntimeError(
            f"Diagnostic {diagnostic_id} did not pass; see the task results above"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-name", required=True)
    parser.add_argument("--name", default="t-factory")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    client = Client(max_retries=20)
    bench_id = ingest(client, args.name, args.agent_name, args.skip_build)
    if not args.skip_build:
        benchmark_diagnostic(client, bench_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
