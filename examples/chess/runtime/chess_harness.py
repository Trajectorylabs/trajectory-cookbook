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


def position(board, moves_remaining):
    return {"fen": board.fen(), "board": str(board), "moves_remaining": moves_remaining,
            "instruction": f"You have {moves_remaining} {'move' if moves_remaining == 1 else 'moves'} to make with submit_move."}


def play(client, tid, puzzle, variant):
    if variant not in {"reasoning", "python"}:
        raise ValueError(f"Unknown variant: {variant}")
    board = chess.Board(puzzle["fen"])
    solution = puzzle["solution"]
    index = 0
    max_steps = 10 if variant == "python" else puzzle["mate_in"]
    system = (
        "Solve this chess puzzle by reproducing its full reference mating continuation. "
        f"{position(board, puzzle['mate_in'])['instruction']} "
        "Play only your own moves, one at a time, with submit_move in UCI notation. "
        "After each correct non-final move, the environment plays the opponent's reference reply "
        "and returns the updated board. Do not submit the opponent's moves. "
        "Reward is 1.0 only when every submitted move matches the reference and you complete "
        "the whole continuation. Any different move scores 0.0, even another legal checkmate. "
        "Think through checks and the opponent's defenses before you play. "
        "Make exactly one tool call per turn. Each response allows at most 4000 output tokens. "
    )
    system += (
        "You have at most 10 model turns total, including Python calls and move submissions. "
        "Use python_chess to calculate with python-chess. Each call gets a fresh board at the current position. "
        "Reserve enough turns to submit every move."
        if variant == "python" else "Use your own reasoning. No calculation tool is available."
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content":
                f"Mate in {puzzle['mate_in']}. You play {'White' if board.turn else 'Black'}.\n"
                f"{position(board, puzzle['mate_in'])['instruction']}\n"
                f"FEN: {board.fen()}\n{board}"}]
    metrics = {"model_turns": 0, "python_calls": 0, "correct_moves": 0, "correct_half_moves": 0,
               "final_fen": board.fen(), "final_board": str(board), "moves_remaining": puzzle["mate_in"]}
    for step in range(max_steps):
        metrics["model_turns"] += 1
        response = client.chat.completions.create(
            model="chess", messages=messages, x_trajectory_id=tid, max_tokens=4000,
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
                # Exact continuation only: even another legal mating move scores zero.
                if uci != solution[index]:
                    return 0.0, "ENV_DONE", metrics
                move = board.parse_uci(uci)
                board.push(move)
                metrics["correct_moves"] += 1
                metrics["correct_half_moves"] += 1
                moves_remaining = puzzle["mate_in"] - metrics["correct_moves"]
                metrics.update(final_fen=board.fen(), final_board=str(board), moves_remaining=moves_remaining)
                index += 1
                if index == len(solution):
                    if not board.is_checkmate():
                        raise RuntimeError("Reference solution did not end in mate")
                    return 1.0, "ENV_DONE", metrics
                opponent = solution[index]
                board.push_uci(opponent)
                index += 1
                metrics["correct_half_moves"] += 1
                metrics.update(final_fen=board.fen(), final_board=str(board))
                observation = {"your_move": uci, "opponent_move": opponent, **position(board, moves_remaining)}
            elif function["name"] == "python_chess" and variant == "python":
                metrics["python_calls"] += 1
                observation = execute_chess_python(board.fen(), {}, arguments["code"])
                if not observation["isolation_active"]:
                    raise RuntimeError(f"Python isolation failed: {observation.get('error')}")
                observation.update(position(board, puzzle["mate_in"] - metrics["correct_moves"]))
            else:
                raise ValueError("Unknown tool")
        except (ValueError, TypeError, KeyError):
            return 0.0, "ENV_DONE", metrics
        messages.append(assistant)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(observation)})
    return 0.0, "MAX_STEPS", metrics


def main(variant=None):
    client = Client(max_retries=2)
    tid = client.trajectories.create().tid
    try:
        puzzle = json.loads(os.environ["CHESS_PUZZLE"])
        variant = variant or os.environ["CHESS_VARIANT"]
        reward, reason, metrics = play(client, tid, puzzle, variant)
        client.trajectories.log_event(tid, event_id="chess-result", name="chess_result", payload={
            "puzzle_id": puzzle["puzzle_id"], "mate_in": puzzle["mate_in"],
            "rating_band": puzzle["rating_band"], "variant": variant,
            **metrics,
        })
        client.trajectories.log_reward(tid, name="reward_accuracy", value=reward)
        client.trajectories.log_reward(tid, name="correct_half_moves", value=metrics["correct_half_moves"],
                                       min_value=0, max_value=len(puzzle["solution"]), weight=0)
        client.trajectories.log_reward(tid, name="correct_solver_moves", value=metrics["correct_moves"],
                                       min_value=0, max_value=puzzle["mate_in"], weight=0)
    except Exception:
        client.trajectories.complete(tid, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(tid, termination_reason=reason)
    if completed.status not in {"completed", "finalizing"}:
        raise RuntimeError(f"Trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
