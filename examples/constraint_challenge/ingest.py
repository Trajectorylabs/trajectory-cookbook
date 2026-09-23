# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Ingest the prompted character-density task for evaluation and training."""

import argparse
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUN_COMMAND = "python -u /opt/constraint_challenge/constraint_harness.py"
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
        family="t-density",
        description="Respond normally while maximizing character-level t density.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"t-density/{split}_{index:04d}",
                split=split,
                run_command=_RUN_COMMAND,
                env_vars={"USER_PROMPT": prompt},
                tags=["t-density"],
            )
            for split, prompts in build_dataset().items()
            for index, prompt in enumerate(prompts)
        ],
    )


def ingest(name: str, skip_build: bool) -> str:
    client = Client()
    agent_id = client.agents.get_default().agent_id
    result = push(client, build_benchmark(name), agent_id=agent_id, root=_ROOT)
    print(f"agent_id={agent_id}", flush=True)
    print(f"bench_id={result.bench_id}", flush=True)
    if not skip_build:
        wait_for_benchmark_images(
            client,
            result.bench_id,
            timeout_seconds=_BUILD_TIMEOUT_SECONDS,
        )
    return result.bench_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="t-density")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    ingest(args.name, args.skip_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
