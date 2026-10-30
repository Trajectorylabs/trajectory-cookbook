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
    client = Client(max_retries=20)
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model="t-factory",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2_048,  # Managed runs use max_output_tokens_per_step instead.
            temperature=1.0,
            top_p=0.95,
            x_trajectory_id=trajectory_id,
        )
    except APIError:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise

    choice = response.choices[0]
    answer = choice.message.content or ""
    reward = t_word_density(answer)
    client.trajectories.log_reward(
        trajectory_id,
        name="reward_t_word_density",
        value=reward,
    )
    cut_off = choice.finish_reason in ("length", "max_tokens")
    completed = client.trajectories.complete(
        trajectory_id, termination_reason="TRUNCATION" if cut_off else "ENV_DONE"
    )
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
