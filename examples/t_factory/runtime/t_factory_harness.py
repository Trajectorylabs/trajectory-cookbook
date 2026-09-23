"""Run and grade one T-starting-word task."""

import os
import re

from trajectory import APIError, Client

_SYSTEM_PROMPT = (
    "try to respond normally but with as many words starting with T as possible"
)


def t_word_density(answer: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", answer)
    return (
        sum(word.lower().startswith("t") for word in words) / len(words)
        if words
        else 0.0
    )


def main() -> None:
    prompt = os.environ["USER_PROMPT"]
    client = Client()
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model="t-factory",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=32_768,
            temperature=1.0,
            top_p=0.95,
            extra_headers={"X-Trajectory-Id": trajectory_id},
        )
    except APIError:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise

    answer = response.choices[0].message.content
    reward = t_word_density(answer)
    client.trajectories.log_reward(
        trajectory_id,
        reward_id="t-word-density",
        name="reward_t_word_density",
        value=reward,
    )
    completed = client.trajectories.complete(
        trajectory_id, termination_reason="ENV_DONE"
    )
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
