# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.9.37", "chess==1.11.2"]
# ///
"""Replay a stored best move through the reasoning harness without calling a model."""

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "harness"))
from chess_harness import play


def reference_trace(record):
    submitted = iter(record["solution"])
    transcript = []
    turn = 0

    def respond(**request):
        nonlocal turn
        turn += 1
        move = next(submitted)
        call = {
            "id": f"move_{turn}",
            "type": "function",
            "function": {
                "name": "submit_move",
                "arguments": json.dumps({"move": move}),
            },
        }
        assistant = {"role": "assistant", "tool_calls": [call]}
        transcript[:] = copy.deepcopy(request["messages"]) + [assistant]
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    message=SimpleNamespace(model_dump=lambda **_: assistant),
                )
            ]
        )

    # This client returns source moves. It has no HTTP client or API credentials.
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=respond))
    )
    reward, ending, metrics = play(client, "reference_replay", record, "reasoning")
    return {
        "kind": "reference_replay",
        "model_calls": 0,
        "position_id": record["position_id"],
        "messages": transcript,
        "result": {"reward": reward, "ending": ending, **metrics},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position-id", default=None)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "visuals" / "reasoning-trace.json"
    )
    args = parser.parse_args()
    record = next(
        json.loads(line)
        for line in (ROOT / "data" / "positions.jsonl").read_text().splitlines()
        if (
            args.position_id is None
            and json.loads(line)["split"] == "test"
            and json.loads(line)["category"] == "tactics"
        )
        or json.loads(line)["position_id"] == args.position_id
    )
    result = reference_trace(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
