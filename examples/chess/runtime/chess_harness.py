"""Play and grade a Lichess puzzle, with optional isolated Python calculations."""

import json
import os

import chess
from trajectory import Client

from python_tool import execute_chess_python


def tool(name, description, argument):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {argument: {"type": "string"}},
                       "required": [argument], "additionalProperties": False},
    }}


SUBMIT = tool("submit_move", "Play your next puzzle move in UCI, for example e2e4 or a7a8q.", "move")
PYTHON = tool("python_chess", "Run Python with chess and board at the current FEN. Use print(). Each call starts fresh. No files, network, engines, or reference answers.", "code")


def play(client, tid, puzzle, variant):
    if variant not in {"reasoning", "python"}:
        raise ValueError(f"Unknown variant: {variant}")
    board = chess.Board(puzzle["fen"])
    solution = puzzle["solution"]
    index = 0
    max_steps = 10 if variant == "python" else puzzle["mate_in"]
    system = (
        "Solve this chess puzzle. Play one move at a time with submit_move in UCI notation. "
        "The environment plays the opponent's reply. Solve the whole puzzle to earn reward. "
        "Think through checks and the opponent's defenses before you play. "
        "Make exactly one tool call per turn. "
    )
    system += (
        "You have at most 10 model turns total, including Python calls and move submissions. "
        "Use python_chess to calculate with python-chess. Each call gets a fresh board at the current position. "
        "Reserve enough turns to submit every move."
        if variant == "python" else "Use your own reasoning. No calculation tool is available."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content":
                f"Mate in {puzzle['mate_in']}. You play {'White' if board.turn else 'Black'}.\n"
                f"FEN: {board.fen()}\n{board}"}]
    metrics = {"model_turns": 0, "python_calls": 0, "correct_moves": 0}
    for step in range(max_steps):
        metrics["model_turns"] += 1
        response = client.chat.completions.create(
            model="chess", messages=messages, x_trajectory_id=tid, max_tokens=4096,
            extra_body={"tools": [SUBMIT, PYTHON] if variant == "python" else [SUBMIT],
                        "parallel_tool_calls": False},
        )
        choice = response.choices[0]
        # Never dispatch a tool from a response that hit the output limit.
        if choice.finish_reason in {"length", "max_tokens"}:
            return 0.0, "TRUNCATION", metrics
        assistant = choice.message.model_dump(mode="json", exclude_none=True)
        calls = assistant.get("tool_calls") or []
        if not calls:
            return 0.0, "MISSING_SUBMIT", metrics
        if len(calls) != 1:
            return 0.0, "ENV_DONE", metrics
        call = calls[0]
        function = call["function"]
        try:
            arguments = json.loads(function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            if function["name"] == "submit_move":
                uci = arguments["move"]
                move = board.parse_uci(uci)
                board.push(move)
                # Any legal final mating move wins, including alternate mates in one.
                if board.is_checkmate():
                    metrics["correct_moves"] += 1
                    return 1.0, "ENV_DONE", metrics
                if uci != solution[index]:
                    return 0.0, "ENV_DONE", metrics
                metrics["correct_moves"] += 1
                index += 1
                if index >= len(solution):
                    raise RuntimeError("Reference solution did not end in mate")
                opponent = solution[index]
                board.push_uci(opponent)
                index += 1
                observation = {"opponent_move": opponent, "fen": board.fen(), "board": str(board)}
            elif function["name"] == "python_chess" and variant == "python":
                metrics["python_calls"] += 1
                observation = execute_chess_python(board.fen(), {}, arguments["code"])
                if not observation["isolation_active"]:
                    raise RuntimeError(f"Python isolation failed: {observation.get('error')}")
            else:
                raise ValueError("Unknown tool")
        except (ValueError, TypeError, KeyError):
            return 0.0, "ENV_DONE", metrics
        messages.append(assistant)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation)})
    return 0.0, "MAX_STEPS", metrics


def main():
    client = Client(max_retries=2)
    tid = client.trajectories.create().tid
    try:
        puzzle = json.loads(os.environ["CHESS_PUZZLE"])
        reward, reason, metrics = play(client, tid, puzzle, os.environ["CHESS_VARIANT"])
        client.trajectories.log_event(tid, event_id="chess-result", name="chess_result", payload={
            "puzzle_id": puzzle["puzzle_id"], "mate_in": puzzle["mate_in"],
            "rating_band": puzzle["rating_band"], "variant": os.environ["CHESS_VARIANT"],
            **metrics,
        })
        client.trajectories.log_reward(tid, name="reward_accuracy", value=reward)
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(tid, termination_reason=reason)
    if completed.status not in {"completed", "finalizing"}:
        raise RuntimeError(f"Trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
