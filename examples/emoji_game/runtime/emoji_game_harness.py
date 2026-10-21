"""Answer a short writing prompt; reward is allowed-emoji density in the reply."""

import argparse
import json
from pathlib import Path

import emoji
from trajectory import Client

PROMPT = "Write a short paragraph about {topic}. Use plenty of emojis."
BASELINE_DENSITY = 0.3
MAX_TOKENS = 512
# Very common emojis; the first 8 are single Qwen3.5 tokens, the rest are two.
# The prompt doesn't name them, so the model learns the set from reward alone.
ALLOWED_EMOJIS = frozenset("✅ ⭐ ❤ 😊 😀 😉 🙂 ✨ 😂 😍 😭 😎 💯 💪 😁".split())


def count_emojis(text: str) -> tuple[int, int]:
    """Return (allowed, other) emoji counts, ignoring variation selectors."""
    found = [match["emoji"].replace("\ufe0f", "") for match in emoji.emoji_list(text)]
    allowed = sum(e in ALLOWED_EMOJIS for e in found)
    return allowed, len(found) - allowed


def score(text: str, completion_tokens: int) -> float:
    """Net allowed emojis per output token, offset so density above 30% is positive."""
    if completion_tokens <= 0:
        return -BASELINE_DENSITY
    allowed, other = count_emojis(text)
    return (allowed - other) / completion_tokens - BASELINE_DENSITY


def play(client: Client, tid: str, topic: str, model: str) -> dict:
    # The prompt asks for emojis but not which ones; reward only favors ALLOWED_EMOJIS.
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
    allowed, other = count_emojis(text)
    return {
        "allowed_emojis": allowed,
        "other_emojis": other,
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
