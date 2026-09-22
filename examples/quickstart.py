# /// script
# dependencies = ["trajectory-sdk"]
# ///
"""Capture and inspect one task through the Trajectory SDK."""

from trajectory import Client


def main() -> None:
    client = Client()
    tid = client.trajectories.create().tid

    response = client.chat.completions.create(
        model="openai/gpt-5.4-mini",
        messages=[
            {
                "role": "user",
                "content": "What is 6 × 7? Reply with only the number.",
            }
        ],
        extra_headers={"X-Trajectory-Id": tid},
    )
    answer = response.choices[0].message.content
    reward = float(answer.strip() == "42")

    client.trajectories.log_reward(
        tid,
        reward_id="correctness",
        name="reward_accuracy",
        value=reward,
    )
    client.trajectories.complete(tid)

    trajectory = client.trajectories.retrieve(tid, include_steps=True)
    print(
        f"trajectory_id={trajectory.trajectory_id} "
        f"status={trajectory.status} reward={trajectory.reward} steps={trajectory.num_steps}"
    )


if __name__ == "__main__":
    main()
