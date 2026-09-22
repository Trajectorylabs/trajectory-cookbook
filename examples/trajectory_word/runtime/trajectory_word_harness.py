"""Run and grade one Trajectory Word instruction-following task."""

import os
import re

from trajectory import Client

_SYSTEM_PROMPT = "Always include the word trajectory in your response."


def includes_trajectory(text: str) -> bool:
    return re.search(r"\btrajectory\b", text, flags=re.IGNORECASE) is not None


def main() -> None:
    prompt = os.environ["WORD_PROMPT"]

    client = Client()
    response = client.chat.completions.create(
        model="trajectory-word",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=32_768,
        temperature=1.0,
        top_p=0.95,
    )
    answer = response.choices[0].message.content
    reward = float(includes_trajectory(answer))

    client.trajectories.log_reward(
        reward_id="trajectory-word-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete()
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
