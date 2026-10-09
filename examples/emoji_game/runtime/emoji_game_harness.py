"""Answer a short writing prompt; reward is emoji density in the reply."""

import argparse
import json
from pathlib import Path

import emoji
from trajectory import Client

PROMPT = "Write a short paragraph about {topic}."
BASELINE_DENSITY = 0.3
MAX_TOKENS = 512


def score(text: str, completion_tokens: int) -> float:
    """Emojis per output token, offset so density above 30% is positive."""
    if completion_tokens <= 0:
        return -BASELINE_DENSITY
    return emoji.emoji_count(text) / completion_tokens - BASELINE_DENSITY


def play(client: Client, tid: str, topic: str, model: str) -> dict:
    # The prompt never mentions emojis; the model learns the objective from reward.
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PROMPT.format(topic=topic)}],
        max_tokens=MAX_TOKENS,
        temperature=1.0,
        x_trajectory_id=tid,
    )
    if response.usage is None:
        raise RuntimeError("The response did not report completion tokens")
    text = response.choices[0].message.content or ""
    completion_tokens = response.usage.completion_tokens
    return {
        "emojis": emoji.emoji_count(text),
        "completion_tokens": completion_tokens,
        "reward": score(text, completion_tokens),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--model", default="emoji-game")
    args = parser.parse_args()
    topic = json.loads(args.task_file.read_text())["topic"]
    client = Client(max_retries=20)
    tid = client.trajectories.create().tid
    try:
        result = play(client, tid, topic, args.model)
        client.trajectories.log_event(
            tid, event_id="emoji-density", name="emoji_game_density", payload=result
        )
        client.trajectories.log_reward(
            tid,
            reward_id="emoji-density",
            name="reward_emoji_density",
            value=result["reward"],
        )
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(tid, termination_reason="ENV_DONE")
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")
    print(json.dumps({"trajectory_id": tid, **result}), flush=True)


if __name__ == "__main__":
    main()
