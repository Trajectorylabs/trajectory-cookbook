# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Ingest the Trajectory Word train/test tasks through the Trajectory SDK."""

import argparse
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUNTIME_DOCKERFILE = "runtime/Dockerfile"
_RUN_COMMAND = "python -u /opt/trajectory_word/trajectory_word_harness.py"
_BUILD_TIMEOUT_SECONDS = 45 * 60
_TRAIN_TASKS = 128
_TEST_TASKS = 64
_TOPICS = (
    "the weather",
    "basic arithmetic",
    "the ocean",
    "gardening",
    "music",
    "history",
    "cooking",
    "bicycles",
    "libraries",
    "healthy sleep",
    "robots",
    "the stars",
    "friendship",
    "painting",
    "maps",
    "the seasons",
)
_QUESTION_TEMPLATES = (
    "What is one interesting fact about {topic}?",
    "Explain {topic} in one sentence.",
    "How would you describe {topic} to a child?",
    "Why might someone learn about {topic}?",
    "Give one practical tip related to {topic}.",
    "What question would you ask about {topic}?",
    "Write a short sentence about {topic}.",
    "What is a common misconception about {topic}?",
    "How does {topic} affect daily life?",
    "Name one benefit associated with {topic}.",
    "What makes {topic} worth discussing?",
    "Summarize {topic} briefly.",
)


def build_dataset() -> dict[str, list[str]]:
    prompts = [
        "How is the weather?",
        "What is 1 + 1?",
        *(
            template.format(topic=topic)
            for template in _QUESTION_TEMPLATES
            for topic in _TOPICS
        ),
    ][: _TRAIN_TASKS + _TEST_TASKS]
    prompts = [prompts[(index * 37) % len(prompts)] for index in range(len(prompts))]
    return {
        "train": prompts[:_TRAIN_TASKS],
        "test": prompts[_TRAIN_TASKS:],
    }


def build_benchmark(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="trajectory-word",
        description="Follow a system instruction to include the word trajectory.",
        runtime=DockerfileBuild(_RUNTIME_DOCKERFILE),
        tasks=[
            TaskSpec(
                name=f"trajectory-word/{split}_{index:04d}",
                split=split,
                run_command=_RUN_COMMAND,
                env_vars={"TRAJECTORY_WORD_PROMPT": prompt},
                tags=["trajectory-word", "instruction-following"],
            )
            for split, prompts in build_dataset().items()
            for index, prompt in enumerate(prompts)
        ],
    )


def ingest(name: str, skip_build: bool) -> str:
    client = Client()
    result = push(client, build_benchmark(name), root=_ROOT)
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
    parser.add_argument("--name", default="trajectory-word-instruction")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    ingest(args.name, args.skip_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
