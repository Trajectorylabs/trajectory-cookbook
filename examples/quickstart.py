# /// script
# dependencies = ["trajectory-sdk==0.6.16"]
# ///
"""Run and grade one platform-managed task through the Trajectory SDK."""

from trajectory import Client


def main() -> None:
    client = Client()

    with client.responses.create(
        model="openai/gpt-5.4-mini",
        input="What is 6 × 7? Reply with only the number.",
        stream=True,
    ) as stream:
        answer = "".join(
            event.delta or ""
            for event in stream
            if event.type == "response.output_text.delta"
        )
    reward = float(answer.strip() == "42")

    client.trajectories.log_reward(
        reward_id="correctness",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete(termination_reason="ENV_DONE")
    print(f"answer={answer.strip()} reward={reward} status={completed.status}")


if __name__ == "__main__":
    main()
