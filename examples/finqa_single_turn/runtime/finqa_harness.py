"""Answer one FinQA question with a FinQA program, graded by FinQA's own execution accuracy."""

import argparse
import json
import re
import sys
from pathlib import Path

from trajectory import APIError, Client

SOURCE = Path("/opt/finqa-src")
FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
# Train rows shown as worked examples in every prompt; never uploaded as tasks.
FEW_SHOT_TRAIN_ROWS = (0, 3, 1)
sys.path.insert(0, str(SOURCE / "code" / "evaluate"))
from evaluate import eval_program, program_tokenization  # noqa: E402

INSTRUCTIONS = """Answer the question about the financial report below by writing a program in
the FinQA language, then submit it with the `submit_program` tool.

A program is a comma-separated sequence of steps. Each step is one operation with two arguments:
  add(a, b)  subtract(a, b)  multiply(a, b)  divide(a, b)  exp(a, b)  greater(a, b)
  table_sum(row, none)  table_average(row, none)  table_max(row, none)  table_min(row, none)
Arguments are numbers copied from the report (e.g. 5829, 23.6%), constants (const_100,
const_1000, const_1000000, const_2, const_m1 for -1, and so on), or #N for the result of step N
(counting from 0). Table operations take a row's first cell exactly as written. The program's
final result is the answer, so follow the conventions in the examples (for instance, a
"percentage change" is usually left as a fraction unless the question asks to multiply by 100).

Worked examples (contexts omitted):
{examples}"""


def _row_text(row: list[str]) -> str:
    return " | ".join(cell.strip() for cell in row)


def render_context(item: dict) -> str:
    table = "\n".join(_row_text(row) for row in item["table"])
    return (
        "Text before the table:\n"
        + "\n".join(item["pre_text"])
        + "\n\nTable:\n"
        + table
        + "\n\nText after the table:\n"
        + "\n".join(item["post_text"])
    )


def few_shot(train: list[dict]) -> str:
    lines = []
    for row in FEW_SHOT_TRAIN_ROWS:
        qa = train[row]["qa"]
        lines.append(f"Q: {qa['question']}\nProgram: {qa['program']}\nResult: {qa['exe_ans']}")
    return "\n\n".join(lines)


SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_program",
        "description": "Submit the final FinQA program, e.g. 'subtract(5829, 5735), divide(#0, 5735)'.",
        "parameters": {
            "type": "object",
            "properties": {"program": {"type": "string"}},
            "required": ["program"],
            "additionalProperties": False,
        },
    },
}


_STEP = re.compile(r"([a-z_]+)\s*\((.*?)\)(?=\s*,|\s*$)", re.DOTALL)
_SPACED_COMMA = re.compile(r"\s*,\s+|\s+,\s*")


def _two_args(inner: str) -> str:
    inner = " ".join(inner.split())
    if _SPACED_COMMA.search(inner):
        first, second = _SPACED_COMMA.split(inner, maxsplit=1)
    else:
        commas = [i for i, c in enumerate(inner) if c == ","]
        if not commas:
            return inner
        # Every step takes two arguments; skip commas that look like thousands separators.
        splits = [i for i in commas if not re.match(r"\d{3}(?![\d.])", inner[i + 1 :])]
        cut = (splits or commas)[0]
        first, second = inner[:cut], inner[cut + 1 :]
    return f"{first.strip()}, {second.strip()}"


def normalize(program: str) -> str:
    """Rewrite spacing to the gold programs' form so FinQA's tokenizer splits steps and arguments."""
    program = program.strip().strip("`").strip()
    steps = _STEP.findall(program)
    if not steps:
        return program
    return ", ".join(f"{op}({_two_args(args)})" for op, args in steps)


def grade(program: str | None, item: dict) -> tuple[float, str]:
    if not program:
        return 0.0, "n/a"
    invalid, result = eval_program(program_tokenization(normalize(program)), item["table"])
    return float(invalid == 0 and result == item["qa"]["exe_ans"]), str(result)


def lenient_match(result: str, gold: object) -> float:
    """Diagnostic only: within 1%, allowing a x100 percentage-scale mismatch."""
    try:
        value, target = float(result), float(gold)
    except (TypeError, ValueError):
        return float(result == str(gold))
    return float(
        any(
            abs(value * scale - target) <= 0.01 * max(abs(target), 1e-9) for scale in (1, 100, 0.01)
        )
    )


def main(split: str, row: int) -> None:
    train = json.loads((SOURCE / "dataset" / "train.json").read_text())
    data = train if split == "train" else json.loads((SOURCE / "dataset" / "test.json").read_text())
    item = data[row]

    client = Client(max_retries=20)
    trajectory_id = client.trajectories.create().tid
    prompt = (
        INSTRUCTIONS.format(examples=few_shot(train))
        + "\n\n"
        + render_context(item)
        + f"\n\nQuestion: {item['qa']['question']}"
    )
    try:
        response = client.chat.completions.create(
            model="finqa",  # Any model name.
            messages=[{"role": "user", "content": prompt}],
            extra_body={"tools": [SUBMIT_TOOL]},
            # Managed runs apply the run's output limit and sampling settings instead.
            max_tokens=8_192,
            x_trajectory_id=trajectory_id,
        )
    except APIError:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise

    choice = response.choices[0]
    program = None
    for call in reversed(choice.message.tool_calls or []):
        if call.function.name == "submit_program":
            try:
                program = str(json.loads(call.function.arguments)["program"])
            except (json.JSONDecodeError, KeyError, TypeError):
                program = None
            break

    reward, result = grade(program, item)
    client.trajectories.log_reward(
        trajectory_id,
        name="execution_accuracy",
        value=reward,
        explanation=f"program={program!r} result={result} gold={item['qa']['exe_ans']}",
    )
    client.trajectories.log_reward(
        trajectory_id,
        name="lenient_answer_match",
        value=lenient_match(result, item["qa"]["exe_ans"]),
        weight=0.0,
    )
    client.trajectories.log_event(
        trajectory_id,
        event_id="finqa-protocol",
        name="finqa_protocol",
        payload={
            "id": item["id"],
            "split": split,
            "row": row,
            "finqa_commit": FINQA_COMMIT,
            "program": program,
            "result": result,
            "gold_program": item["qa"]["program"],
            "gold_result": item["qa"]["exe_ans"],
        },
    )
    cut_off = choice.finish_reason in ("length", "max_tokens")
    completed = client.trajectories.complete(
        trajectory_id, termination_reason="TRUNCATION" if cut_off else "ENV_DONE"
    )
    if completed.status not in ("completed", "finalizing"):
        raise RuntimeError(f"trajectory completion failed: {completed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--row", type=int, required=True)
    args = parser.parse_args()
    main(args.split, args.row)
