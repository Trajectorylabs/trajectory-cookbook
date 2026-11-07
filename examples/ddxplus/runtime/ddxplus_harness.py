"""Run a bounded, static-evidence synthetic DDXPlus diagnostic interview."""

import argparse
import json
from pathlib import Path

from trajectory import Client

QUESTION_LIMIT = 15
TURN_LIMIT = 20
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ask_evidence",
            "description": "Ask one catalog evidence question. Reveal its parent first when it has one. Maximum 15 distinct questions.",
            "parameters": {
                "type": "object",
                "properties": {"evidence_id": {"type": "string"}},
                "required": ["evidence_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_differential",
            "description": "Finish with a ranked list of 1 to 5 distinct catalog condition names. First is your diagnosis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 5,
                    }
                },
                "required": ["conditions"],
                "additionalProperties": False,
            },
        },
    },
]


class Interview:
    def __init__(self, patient: dict, evidences: dict, conditions: list[str]):
        self.patient = patient
        self.evidences = evidences
        self.conditions = conditions
        self.profile = {}
        for token in patient["evidences"]:
            evidence_id, separator, value = token.partition("_@_")
            self.profile.setdefault(evidence_id, []).append(
                value if separator else True
            )
        self.revealed = {patient["initial_evidence"].partition("_@_")[0]}
        self.questions = set()

    def values(self, evidence_id: str) -> list:
        metadata = self.evidences[evidence_id]
        default = metadata["default_value"]
        return self.profile.get(
            evidence_id,
            [bool(default) if metadata["data_type"] == "B" else str(default)],
        )

    def positive(self, evidence_id: str) -> bool:
        default = self.evidences[evidence_id]["default_value"]
        return (
            any(str(value) != str(default) for value in self.values(evidence_id))
            if self.evidences[evidence_id]["data_type"] != "B"
            else bool(self.values(evidence_id)[0])
        )

    def answer(self, evidence_id: str) -> dict:
        metadata = self.evidences[evidence_id]
        values = self.values(evidence_id)
        meanings = metadata["value_meaning"]
        return {
            "evidence_id": evidence_id,
            "question": metadata["question_en"],
            "data_type": metadata["data_type"],
            "values": values,
            "meanings": [
                meanings[value]["en"] if value in meanings else value
                for value in values
            ],
        }

    def ask(self, arguments: dict) -> dict:
        if set(arguments) != {"evidence_id"} or not isinstance(
            arguments["evidence_id"], str
        ):
            return {"error": "Supply only a string evidence_id"}
        evidence_id = arguments["evidence_id"]
        if evidence_id not in self.evidences:
            return {"error": "Unknown evidence_id"}
        if evidence_id in self.revealed:
            return self.answer(evidence_id)
        if len(self.questions) >= QUESTION_LIMIT:
            return {"error": "Question budget exhausted; submit your differential"}
        parent = self.evidences[evidence_id]["code_question"]
        if (
            parent in self.evidences
            and parent != evidence_id
            and parent not in self.revealed
        ):
            return {"error": "Ask parent evidence first", "parent_evidence_id": parent}
        self.questions.add(evidence_id)
        self.revealed.add(evidence_id)
        answer = self.answer(evidence_id)
        if (
            parent in self.evidences
            and parent != evidence_id
            and not self.positive(parent)
        ):
            if self.positive(evidence_id):
                raise ValueError(
                    "Source profile contains active evidence under an inactive parent"
                )
            answer["not_applicable"] = True
        return answer

    def prompt(self) -> str:
        catalog = [
            {
                "id": key,
                "question": item["question_en"],
                "type": item["data_type"],
                "parent": item["code_question"],
            }
            for key, item in self.evidences.items()
        ]
        return (
            "Research simulation with synthetic patients. Ask catalog questions and submit a ranked differential. No real clinical advice. Only use the provided tools. You have 15 distinct follow-up questions; duplicates do not cost extra.\n"
            + json.dumps(
                {
                    "age": self.patient["age"],
                    "sex": self.patient["sex"],
                    "initial_evidence": self.answer(
                        self.patient["initial_evidence"].partition("_@_")[0]
                    ),
                    "evidence_catalog": catalog,
                    "condition_catalog": self.conditions,
                }
            )
        )

    def grade(self, arguments: dict) -> dict:
        ranked = arguments.get("conditions")
        valid = (
            set(arguments) == {"conditions"}
            and isinstance(ranked, list)
            and 1 <= len(ranked) <= 5
        )
        valid = valid and all(
            isinstance(item, str) and item in self.conditions for item in ranked
        )
        valid = valid and len(set(ranked)) == len(ranked)
        if not valid:
            return {
                "accuracy": 0.0,
                "valid_submission": False,
                "questions": len(self.questions),
                "differential_recall": 0.0,
            }
        expected = {
            label
            for label, probability in self.patient["differential"]
            if probability > 0
        }
        return {
            "accuracy": float(ranked[0] == self.patient["pathology"]),
            "valid_submission": True,
            "questions": len(self.questions),
            "differential_recall": len(set(ranked) & expected) / len(expected),
        }


def run(interview: Interview, client: Client, tid: str) -> None:
    try:
        run_turns(interview, client, tid)
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise


def run_turns(interview: Interview, client: Client, tid: str) -> None:
    messages = [{"role": "user", "content": interview.prompt()}]
    for _ in range(TURN_LIMIT):
        response = client.chat.completions.create(
            model="ddxplus-policy",
            messages=messages,
            extra_body={"tools": TOOLS},
            max_tokens=512,
            x_trajectory_id=tid,
        )
        choice = response.choices[0]
        if choice.finish_reason in {"length", "max_tokens"}:
            client.trajectories.complete(tid, termination_reason="TRUNCATION")
            return
        if choice.finish_reason not in {"tool_calls", "stop"}:
            client.trajectories.complete(tid, termination_reason="ERROR")
            return
        calls = choice.message.tool_calls
        if calls and choice.finish_reason != "tool_calls":
            client.trajectories.complete(tid, termination_reason="ERROR")
            return
        if not calls:
            client.trajectories.complete(tid, termination_reason="TRUNCATION")
            return
        messages.append(
            {
                "role": "assistant",
                "content": choice.message.content,
                "tool_calls": [call.model_dump(exclude_none=True) for call in calls],
            }
        )
        for call in calls:
            arguments = json.loads(call.function.arguments)
            if not isinstance(arguments, dict):
                raise TypeError("Tool arguments must be a JSON object")
            if call.function.name == "submit_differential":
                metrics = interview.grade(arguments)
                client.trajectories.log_reward(
                    tid,
                    name="diagnosis_accuracy",
                    value=metrics["accuracy"],
                    explanation=json.dumps(metrics),
                )
                client.trajectories.complete(tid, termination_reason="ENV_DONE")
                return
            if call.function.name == "ask_evidence" and isinstance(arguments, dict):
                result = interview.ask(arguments)
            else:
                result = {"error": "Unknown tool or invalid JSON object"}
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)}
            )
    client.trajectories.complete(tid, termination_reason="TRUNCATION")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()
    client = Client()
    tid = client.trajectories.create().tid
    try:
        root = Path(__file__).parent
        task = next(
            row
            for line in (root / "tasks.jsonl").read_text().splitlines()
            if (row := json.loads(line))["task_id"] == args.task_id
        )
        evidences = json.loads((root / "release_evidences.json").read_text())
        conditions = list(json.loads((root / "release_conditions.json").read_text()))
        interview = Interview(task["patient"], evidences, conditions)
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    run(interview, client, tid)


if __name__ == "__main__":
    main()
