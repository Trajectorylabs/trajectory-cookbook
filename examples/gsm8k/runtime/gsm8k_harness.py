"""Run and grade one GSM8K task."""

import json
import os
import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation

from trajectory import Client

_PROMPT = """Solve this grade-school math problem carefully. Show your reasoning, then submit
the final numeric answer by calling the `submit_answer` tool. Do not state the final answer only
as text -- you must call `submit_answer` for your answer to count.

{question}"""
_SUBMIT_ANSWER_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit the final numeric answer to the math problem.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final numeric answer.",
                }
            },
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}


def extract_number(text: str) -> Decimal | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not matches:
        return None
    try:
        return Decimal(matches[-1])
    except InvalidOperation:
        return None


def extract_submitted_answer(
    tool_calls: list[Mapping[str, object]] | None,
) -> Decimal | None:
    for tool_call in reversed(tool_calls or []):
        function = tool_call.get("function")
        if not isinstance(function, Mapping) or function.get("name") != "submit_answer":
            continue
        try:
            arguments = json.loads(str(function["arguments"]))
            return extract_number(str(arguments["answer"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return None


def main() -> None:
    question = os.environ["GSM8K_QUESTION"]
    expected = extract_number(os.environ["GSM8K_ANSWER"])

    client = Client()
    trajectory_id = client.trajectories.create().tid
    response = client.chat.completions.create(
        extra_headers={"X-Trajectory-Id": trajectory_id},
        model="gsm8k",  # Any model name.
        messages=[
            {"role": "user", "content": _PROMPT.format(question=question)},
        ],
        extra_body={"tools": [_SUBMIT_ANSWER_TOOL]},
        max_tokens=32_768,
        temperature=1.0,
        top_p=0.95,
    )
    submitted = extract_submitted_answer(
        getattr(response.choices[0].message, "tool_calls", None)
    )
    reward = float(
        submitted is not None and expected is not None and submitted == expected
    )

    client.trajectories.log_reward(
        trajectory_id,
        reward_id="gsm8k-accuracy",
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete(trajectory_id)
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
