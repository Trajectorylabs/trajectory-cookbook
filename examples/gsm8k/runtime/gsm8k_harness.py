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
    response = client.chat.completions.create(
        model="gsm8k",  # Any model name.
        messages=[
            {"role": "user", "content": _PROMPT.format(question=question)},
        ],
        max_tokens=32_768,
        temperature=1.0,
        top_p=0.95,
    )
    answer = response.choices[0].message.content or ""
    submitted = extract_number(answer)
    reward = float(
        submitted is not None and expected is not None and submitted == expected
    )

    client.trajectories.log_reward(
        reward_id="gsm8k-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete()
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
