"""Run and grade one constraint-following task."""

import os
import re

from trajectory import Client

_SYSTEM_PROMPTS = {
    kind: (
        f"Answer the user in exactly four words. Do not use the letter {kind[-1]}, "
        "uppercase or lowercase."
    )
    for kind in ("no_y", "no_p", "no_m")
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", text)


def follows_constraint(kind: str, answer: str) -> bool:
    if kind in _SYSTEM_PROMPTS:
        forbidden_letter = kind[-1]
        return len(_words(answer)) == 4 and forbidden_letter not in answer.lower()
    raise ValueError(f"unknown task kind: {kind}")


def main() -> None:
    kind = os.environ["TASK_KIND"]
    prompt = os.environ["USER_PROMPT"]
    client = Client()
    response = client.chat.completions.create(
        model="constraint-challenge",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPTS[kind]},
            {"role": "user", "content": prompt},
        ],
        max_tokens=32_768,
        temperature=1.0,
        top_p=0.95,
    )
    answer = response.choices[0].message.content
    reward = float(follows_constraint(kind, answer))
    client.trajectories.log_reward(
        reward_id=f"{kind}-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete()
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
