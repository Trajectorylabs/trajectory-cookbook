"""Run and grade one response-density task."""

import os

from trajectory import APIError, Client

_SYSTEM_PROMPT = "try to respond normally but with as many t as possible"


def t_density(answer: str) -> float:
    return answer.lower().count("t") / len(answer) if answer else 0.0


def main() -> None:
    prompt = os.environ["USER_PROMPT"]
    client = Client()
    trajectory_id = client.trajectories.create().tid
    try:
        response = client.chat.completions.create(
            model="constraint-challenge",
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
    reward = t_density(answer)
    client.trajectories.log_reward(
        trajectory_id,
        reward_id="t-density",
        name="reward_t_density",
        value=reward,
    )
    completed = client.trajectories.complete(
        trajectory_id, termination_reason="ENV_DONE"
    )
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
