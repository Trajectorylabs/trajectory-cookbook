# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Upload a toy benchmark that rewards emoji density in short writing tasks."""

import argparse
import json
from pathlib import Path

from trajectory import BenchmarkSpec, Client, TaskSpec
from trajectory.lib import DockerfileBuild, push, wait_for_benchmark_images

ROOT = Path(__file__).parent
TRAIN_TOPICS = [
    "your favorite season",
    "a day at the beach",
    "learning to cook",
    "a rainy afternoon",
    "the first day of school",
    "walking a dog",
    "a birthday party",
    "visiting a farm",
    "riding a bicycle",
    "a camping trip",
    "baking bread",
    "a city at night",
    "planting a garden",
    "a road trip",
    "your morning routine",
    "a snow day",
    "a trip to the zoo",
    "playing soccer",
    "a picnic in the park",
    "moving to a new home",
    "a thunderstorm",
    "a visit to the library",
    "making new friends",
    "a summer festival",
    "watching the sunrise",
    "a family dinner",
    "learning a new language",
    "a visit to the museum",
    "a lazy Sunday",
    "a game night",
    "a hike in the mountains",
    "the ocean",
]
TEST_TOPICS = [
    "a weekend getaway",
    "a favorite meal",
    "the first snowfall",
    "a concert",
    "starting a new job",
    "a cat napping",
    "fall leaves",
    "a farmers market",
    "stargazing",
    "a long flight",
    "a surprise gift",
    "a spring morning",
    "an old friend",
    "a coffee shop",
    "a holiday celebration",
    "a quiet evening at home",
]


def build_benchmark(name: str) -> BenchmarkSpec:
    task_dir = ROOT / "runtime" / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    # Test topics are unseen during training, so a learned emoji habit must generalize.
    for split, topics in (("train", TRAIN_TOPICS), ("test", TEST_TOPICS)):
        for index, topic in enumerate(topics):
            filename = f"{split}_{index:04d}.json"
            (task_dir / filename).write_text(json.dumps({"topic": topic}))
            tasks.append(
                TaskSpec(
                    name=f"emoji-game/{split}_{index:04d}",
                    split=split,
                    run_command=(
                        "python -u /opt/emoji_game/emoji_game_harness.py "
                        f"--task-file /opt/emoji_game/tasks/{filename}"
                    ),
                    tags=["emoji-game"],
                )
            )
    return BenchmarkSpec(
        name=name,
        family="emoji-game",
        description="Short writing prompts rewarded by emojis per output token minus 0.3.",
        runtime=DockerfileBuild("runtime/Dockerfile"),
        tasks=tasks,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--name", default="emoji-game")
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
