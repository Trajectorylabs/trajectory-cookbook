"""Guess a fake user's number while preserving the complete conversation."""

import argparse
import json
import re
from pathlib import Path

from trajectory import Client

GUESS_PROMPT = (
    "Guess the user's secret integer from 1 to 100 inclusive. "
    "Use their feedback. Output only one integer per turn."
)
MAX_GUESSES = 12
MAX_TOKENS = 8_192


def run_game(client: Client, tid: str, secret: int, model: str) -> dict:
    """Keep the secret and guess count in the environment, outside model messages."""
    if not 1 <= secret <= 100:
        raise ValueError("secret must be in 1–100")
    history = [
        {"role": "user", "content": "I've picked a number from 1 to 100. Guess it."}
    ]
    for guesses in range(1, MAX_GUESSES + 1):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": GUESS_PROMPT}, *history],
            max_tokens=MAX_TOKENS,
            temperature=1.0,
            x_trajectory_id=tid,
        )
        answer = response.choices[0].message.content or ""
        history.append({"role": "assistant", "content": answer})
        guess = (
            int(answer.strip()) if re.fullmatch(r"[0-9]{1,3}", answer.strip()) else None
        )
        if guess is None or not 1 <= guess <= 100:
            feedback = "Invalid guess. Output one integer from 1 to 100."
        elif guess == secret:
            return {
                "solved": True,
                "guesses": guesses,
                "reward": 1 / max(guesses - 1, 1),
            }
        else:
            feedback = "Higher." if guess < secret else "Lower."
        history.append({"role": "user", "content": feedback})

    return {"solved": False, "guesses": MAX_GUESSES, "reward": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    parser.add_argument("--model", default="guess-number")
    args = parser.parse_args()
    secret = json.loads(args.task_file.read_text())["secret"]
    client = Client(max_retries=20)
    tid = client.trajectories.create().tid
    try:
        result = run_game(client, tid, secret, args.model)
        client.trajectories.log_reward(
            tid,
            reward_id="guess-efficiency",
            name="reward_guess_efficiency",
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
