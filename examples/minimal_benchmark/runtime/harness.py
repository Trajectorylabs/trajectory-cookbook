import os

from trajectory import APIError, Client

question = os.environ["QUESTION"]
expected = os.environ["EXPECTED_ANSWER"]

client = Client()
tid = client.trajectories.create().tid
try:
    response = client.chat.completions.create(
        model="policy",
        messages=[
            {"role": "system", "content": "Return only the numeric answer."},
            {"role": "user", "content": question},
        ],
        max_tokens=128,
        x_trajectory_id=tid,
    )
    answer = response.choices[0].message.content or ""
    reward = float(answer.strip() == expected)
    client.trajectories.log_reward(
        tid,
        reward_id="answer-accuracy",
        name="reward_accuracy",
        value=reward,
    )
except Exception as error:
    try:
        client.trajectories.complete(tid, termination_reason="ERROR")
    except APIError as completion_error:
        error.add_note(f"Error completion failed: {type(completion_error).__name__}")
    raise
else:
    client.trajectories.complete(tid)
