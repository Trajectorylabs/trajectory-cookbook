"""Run and grade one GSM8K task."""

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from trajectory import Client
from trajectory.types.inference.chat_completion_response import (
    ChatCompletionResponseToolCall,
)

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
    tool_calls: list[ChatCompletionResponseToolCall] | None,
) -> Decimal | None:
    for tool_call in reversed(tool_calls or []):
        if tool_call.function.name != "submit_answer":
            continue
        try:
            arguments = json.loads(tool_call.function.arguments)
            return extract_number(str(arguments["answer"]))
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", required=True, type=Path)
    args = parser.parse_args()
    task = json.loads(args.task_file.read_text())
    question = task["question"]
    expected = extract_number(task["expected_answer"])

    client = Client(max_retries=20)
    trajectory_id = client.trajectories.create().tid
    response = client.chat.completions.create(
        model="gsm8k",  # Any model name.
        messages=[
            {"role": "user", "content": _PROMPT.format(question=question)},
        ],
        extra_body={"tools": [_SUBMIT_ANSWER_TOOL]},
        max_tokens=2_048,
        temperature=1.0,
        top_p=0.95,
        x_trajectory_id=trajectory_id,
    )
    submitted = extract_submitted_answer(
        getattr(response.choices[0].message, "tool_calls", None)
    )
    reward = float(
        submitted is not None and expected is not None and submitted == expected
    )

    client.trajectories.log_reward(
        trajectory_id,
        name="reward_accuracy",
        value=reward,
    )
    completed = client.trajectories.complete(trajectory_id)
    if completed.status != "completed":
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
