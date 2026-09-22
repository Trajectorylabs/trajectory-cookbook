# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Ingest three constraint-following tasks for evaluation and training."""

import argparse
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

_ROOT = Path(__file__).parent
_RUN_COMMAND = "python -u /opt/constraint_challenge/constraint_harness.py"
_BUILD_TIMEOUT_SECONDS = 45 * 60
_TRAIN_COUNTS = {
    "no_y": 43,
    "no_p": 43,
    "no_m": 42,
}
_TEST_COUNTS = {
    "no_y": 21,
    "no_p": 21,
    "no_m": 22,
}
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
    "What is one surprising fact about {topic}?",
    "Explain {topic} to a curious child.",
    "Give one useful tip about {topic}.",
    "Why do people care about {topic}?",
    "Describe {topic} in a memorable way.",
    "What makes {topic} interesting?",
)
_KIND_OFFSETS = {"no_m": 77, "no_p": 110, "no_y": 165}


def _natural_prompts(kind: str) -> tuple[list[str], list[str]]:
    prompts = [
        template.format(topic=topic) for template in _TEMPLATES for topic in _TOPICS
    ]
    kind_offset = _KIND_OFFSETS[kind]
    ordered = [
        prompts[(index * 37 + kind_offset) % len(prompts)] for index in range(64)
    ]
    extra_train = _TRAIN_COUNTS[kind] - 32
    extra_test = _TEST_COUNTS[kind] - 16
    train = ordered[:32] + ordered[48 : 48 + extra_train]
    test = ordered[32:48] + ordered[48 + extra_train : 48 + extra_train + extra_test]
    return train, test


def build_dataset() -> dict[str, list[tuple[str, str]]]:
    rows: dict[str, list[tuple[str, str]]] = {"train": [], "test": []}
    for kind in ("no_y", "no_p", "no_m"):
        train, test = _natural_prompts(kind)
        rows["train"].extend((kind, prompt) for prompt in train)
        rows["test"].extend((kind, prompt) for prompt in test)
    return rows


def build_benchmark(name: str) -> BenchmarkSpec:
    return BenchmarkSpec(
        name=name,
        family="constraint-challenge",
        description="Three simple, deterministic instruction-following challenges.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=[
            TaskSpec(
                name=f"constraint-challenge/{kind}/{split}_{index:04d}",
                split=split,
                run_command=_RUN_COMMAND,
                env_vars={
                    "TASK_KIND": kind,
                    "USER_PROMPT": prompt,
                },
                tags=[kind],
            )
            for split, split_rows in build_dataset().items()
            for index, (kind, prompt) in enumerate(split_rows)
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
    parser.add_argument("--name", default="constraint-challenge")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()
    ingest(args.name, args.skip_build)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
