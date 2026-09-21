"""Run and grade one GSM8K task."""

import os
import re
from decimal import Decimal, InvalidOperation

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


def main() -> None:
    question = os.environ["GSM8K_QUESTION"]
    expected = extract_number(os.environ["GSM8K_ANSWER"])

    client = Client()
    with client.responses.create(
        model="gsm8k",  # Any model name.
        input=_PROMPT.format(question=question),
        stream=True,
        extra_body={
            "max_output_tokens": 1024,
            "temperature": 1.0,
            "top_p": 0.95,
        },
    ) as stream:
        answer = "".join(
            event.delta or ""
            for event in stream
            if event.type == "response.output_text.delta"
        )
    submitted = extract_number(answer)
    reward = float(
        submitted is not None and expected is not None and submitted == expected
    )

    client.trajectories.log_reward(
        reward_id="gsm8k-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete(termination_reason="ENV_DONE")
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
