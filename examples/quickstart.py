# /// script
# dependencies = ["trajectory-sdk==0.6.16"]
# ///
"""Capture and inspect one trajectory through the Trajectory SDK."""

from uuid import uuid4

from trajectory import Client


def main() -> None:
    client = Client()
    agents = list(client.agents.list(limit=1))
    agent = agents[0] if agents else client.agents.create(name="Quickstart Agent")
    created = client.agents.trajectories.create(
        agent.agent_id,
        idempotency_key=f"quickstart-{uuid4()}",
    )

    with client.responses.create(
        model="openai/gpt-5.4-mini",
        x_trajectory_id=created.tid,
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
        created.tid,
        reward_id="correctness",
        name="reward_accuracy",
        value=reward,
    )
    client.trajectories.complete(created.tid, termination_reason="ENV_DONE")

    trajectory = client.trajectories.retrieve(created.tid, include_steps=True)
    print(
        f"trajectory_id={trajectory.trajectory_id} "
        f"status={trajectory.status} reward={trajectory.reward} steps={trajectory.num_steps}"
    )


if __name__ == "__main__":
    main()
