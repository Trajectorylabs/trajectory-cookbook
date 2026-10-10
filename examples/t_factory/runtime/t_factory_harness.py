# /// script
# dependencies = ["trajectory-sdk>=0.9.14"]
# ///
"""Run and grade one T-starting-word task."""

import argparse
import os
import re

from trajectory import APIError, Client

_SYSTEM_PROMPT = "try to respond normally but with as many words starting with T as possible"


def t_word_density(answer: str) -> float:
    words = re.findall(r"[A-Za-z0-9']+", answer)
    return sum(word.lower().startswith("t") for word in words) / len(words) if words else 0.0


def run_task(client: Client, prompt: str, model: str) -> None:
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2_048,
            temperature=1.0,
            top_p=0.95,
            x_trajectory_id=trajectory_id,
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
    completed = client.trajectories.complete(trajectory_id, termination_reason="ENV_DONE")
    print(f"prompt={prompt}", flush=True)
    print(f"response={answer}", flush=True)
    print(f"trajectory_id={trajectory_id}", flush=True)
    print(f"reward={reward:.6f}", flush=True)
    print(f"status={completed.status}", flush=True)
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt", default=os.environ.get("USER_PROMPT", "Do you enjoy rainy weather? Why?")
    )
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()
    run_task(Client(max_retries=20), args.prompt, args.model)


if __name__ == "__main__":
    main()
