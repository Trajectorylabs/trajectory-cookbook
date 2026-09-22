"""Run and grade one response-density task."""

import os
import re

from trajectory import Client

_SYSTEM_PROMPTS = {
    "t_density": "You are a helpful assistant.",
    "t_density_prompted": "try to respond normally but with as many t as possible",
    "t_word_density_prompted": (
        "try to respond normally but with as many words starting with T as possible"
    ),
}


def t_density(answer: str) -> float:
    return answer.lower().count("t") / len(answer) if answer else 0.0


def t_word_density(answer: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", answer)
    return (
        sum(word.lower().startswith("t") for word in words) / len(words)
        if words
        else 0.0
    )


def score_response(kind: str, answer: str) -> float:
    if kind in {"t_density", "t_density_prompted"}:
        return t_density(answer)
    if kind == "t_word_density_prompted":
        return t_word_density(answer)
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
    reward = score_response(kind, answer)
    client.trajectories.log_reward(
        reward_id=kind.replace("_", "-"),
        name=f"reward_{kind.removesuffix('_prompted')}",
        value=reward,
    )
    completed = client.trajectories.complete()
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
