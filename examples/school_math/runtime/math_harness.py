"""Run a bounded tool-using actor and grade one numeric mathematics task."""

import argparse
import ast
import json
import re
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

from trajectory import Client

NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:/[+-]?[0-9]+)?")
MAX_TURNS = 8
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {argument: {"type": "string"}},
                "required": [argument],
                "additionalProperties": False,
            },
        },
    }
    for name, description, argument in (
        (
            "calculate",
            "Evaluate bounded rational arithmetic using + - * / ** and parentheses; no variables or functions.",
            "expression",
        ),
        (
            "submit_answer",
            "Submit one final integer, decimal or fraction, without units or prose.",
            "answer",
        ),
    )
]


def parse_number(value: str) -> Fraction:
    if (
        not isinstance(value, str)
        or len(value) > 120
        or NUMBER.fullmatch(value.strip()) is None
    ):
        raise ValueError("Use one integer, decimal or fraction, without units or prose")
    try:
        number = Fraction(value.strip())
    except ZeroDivisionError as error:
        raise ValueError("The denominator must be nonzero") from error
    if max(number.numerator.bit_length(), number.denominator.bit_length()) > 512:
        raise ValueError("Number exceeds the size limit")
    return number


def calculate(expression: str) -> str:
    if not isinstance(expression, str) or len(expression) > 256:
        raise ValueError("Expression must be a string of at most 256 characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid arithmetic expression") from error
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression exceeds the node limit")

    def evaluate(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            result = parse_number(ast.get_source_segment(expression, node))
        elif isinstance(node, ast.UnaryOp) and isinstance(
            node.op, (ast.UAdd, ast.USub)
        ):
            result = evaluate(node.operand) * (
                -1 if isinstance(node.op, ast.USub) else 1
            )
        elif isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                result = left + right
            elif isinstance(node.op, ast.Sub):
                result = left - right
            elif isinstance(node.op, ast.Mult):
                result = left * right
            elif isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError("Division by zero")
                result = left / right
            elif (
                isinstance(node.op, ast.Pow)
                and right.denominator == 1
                and abs(right) <= 12
            ):
                if left == 0 and right < 0:
                    raise ValueError("Division by zero")
                result = left ** int(right)
            else:
                raise ValueError(
                    "Only + - * / and integer powers from -12 to 12 are allowed"
                )
        else:
            raise ValueError(
                "Variables, calls, attributes and indexing are not allowed"
            )
        if max(result.numerator.bit_length(), result.denominator.bit_length()) > 512:
            raise ValueError("Result exceeds the size limit")
        return result

    return str(evaluate(tree.body))


def actor_messages(task: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "Solve this school-math question. You may use calculate for exact rational arithmetic. "
                "Call submit_answer with a single integer, decimal or fraction. Do not include prose or units. "
                "You have at most eight assistant turns.\n\n" + task["question"]
            ),
        }
    ]


def run_task(client: Client, task: dict, max_turns: int = MAX_TURNS) -> str:
    tid = client.trajectories.create().tid
    try:
        expected = parse_number(task["answer"])
        messages = actor_messages(task)
        for _ in range(max_turns):
            response = client.chat.completions.create(
                model="policy",
                messages=messages,
                extra_body={"tools": TOOLS},
                max_tokens=1024,
                x_trajectory_id=tid,
            )
            choice = response.choices[0]
            if choice.finish_reason in ("length", "max_tokens"):
                client.trajectories.complete(tid, termination_reason="TRUNCATION")
                return tid
            message = choice.message
            expected_finish = "tool_calls" if message.tool_calls else "stop"
            if choice.finish_reason != expected_finish:
                raise ValueError(
                    f"Unexpected model finish reason: {choice.finish_reason}"
                )
            messages.append(message.model_dump(exclude_none=True))
            calls = message.tool_calls or []
            submitted = None
            for call in calls:
                arguments = json.loads(call.function.arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("Tool arguments must be a JSON object")
                try:
                    name = call.function.name
                    if name == "calculate" and set(arguments) == {"expression"}:
                        result = {"result": calculate(arguments["expression"])}
                    elif name == "submit_answer" and set(arguments) == {"answer"}:
                        if submitted is not None:
                            raise ValueError("Submit exactly one answer per turn")
                        submitted = parse_number(arguments["answer"])
                        result = {"submitted": True}
                    else:
                        raise ValueError("Unknown tool or invalid argument keys")
                except (ValueError, TypeError) as error:
                    result = {"error": str(error)}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(result),
                    }
                )
            if submitted is not None:
                client.trajectories.log_reward(
                    tid,
                    name="numeric_accuracy",
                    value=float(submitted == expected),
                    min_value=0,
                    max_value=1,
                    explanation=json.dumps(
                        {
                            "submitted": str(submitted),
                            "expected": str(expected),
                            "source_module": task["source_module"],
                        }
                    ),
                )
                client.trajectories.complete(tid, termination_reason="ENV_DONE")
                return tid
            if not calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Use submit_answer to finish. A prose answer does not submit an answer.",
                    }
                )
        client.trajectories.complete(tid, termination_reason="TRUNCATION")
    except Exception:
        client.trajectories.log_event(
            tid,
            event_id=str(uuid4()),
            name="harness_error",
            payload={"category": "math_harness_failure"},
        )
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    return tid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    tasks = [
        json.loads(line)
        for line in Path(__file__).with_name("tasks.jsonl").read_text().splitlines()
    ]
    task = next(row for row in tasks if row["id"] == args.task_id)
    run_task(Client(), task)


if __name__ == "__main__":
    main()
