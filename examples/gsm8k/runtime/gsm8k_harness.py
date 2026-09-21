"""Run and grade one GSM8K task."""

import os
import re
from decimal import Decimal, InvalidOperation

from openai import OpenAI
from trajectory import Client

_PROMPT = """Solve this grade-school math problem carefully. Show your reasoning, then put the
final numeric answer on the last line in the exact form `#### <number>`.

{question}"""


def extract_number(text: str) -> Decimal | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def _runtime_client(
    base_url: str,
    token: str,
    headers: dict[str, str] | None = None,
) -> Client:
    return Client(
        trajectory_token=token,
        base_url=base_url,
        default_headers=headers,
        max_retries=5,
        timeout=180,
    )


def main() -> None:
    trajectory_id = os.environ["TRAJECTORY_TID"]
    question = os.environ["GSM8K_QUESTION"]
    expected = extract_number(os.environ["GSM8K_ANSWER"])
    model_id = os.environ["MODEL_ENDPOINT_ID"]

    client = OpenAI(
        api_key=os.environ["MODEL_ENDPOINT_ACCESS_TOKEN"],
        base_url=os.environ["MODEL_ENDPOINT_URL"],
        default_headers={
            "X-Trajectory-Id": trajectory_id,
            "X-Model-Request-Id": "gsm8k-model-request",
        },
        max_retries=5,
        timeout=180,
    )
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": _PROMPT.format(question=question)}],
        max_tokens=1024,
        temperature=1.0,
        top_p=0.95,
    )
    answer = response.choices[0].message.content or ""
    submitted = extract_number(answer)
    reward = float(
        submitted is not None and expected is not None and submitted == expected
    )

    trajectories = _runtime_client(
        os.environ["TRAJECTORY_BASE_URL"],
        os.environ["TRAJECTORY_TOKEN"],
    ).trajectories
    trajectories.log_event(
        trajectory_id,
        event_id="gsm8k-answer",
        name="gsm8k_answer",
        payload={
            "answer": answer,
            "submitted": str(submitted) if submitted is not None else None,
        },
    )
    trajectories.log_reward(
        trajectory_id,
        reward_id="gsm8k-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = trajectories.complete(trajectory_id, termination_reason="ENV_DONE")
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
