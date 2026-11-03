"""Select and grade one best chess move, with optional isolated Python calculations."""

import json
import os

import chess
from trajectory import Client

from python_tool import execute_chess_python

MAX_OUTPUT_TOKENS = 4_000
PYTHON_MAX_TURNS = 100


def tool(name, description, argument):
    return {
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


SUBMIT_MOVE_TOOL = tool(
    "submit_move",
    "Submit your best move in UCI, for example e2e4 or a7a8q.",
    "move",
)
PYTHON_CHESS_TOOL = tool(
    "python_chess",
    "Run Python with chess and board at the current FEN. Use print(). Each call starts fresh. No files, network, engines, or reference answers.",
    "code",
)


def position(board, moves_remaining):
    move_word = "move" if moves_remaining == 1 else "moves"
    return {
        "fen": board.fen(),
        "board": str(board),
        "moves_remaining": moves_remaining,
        "instruction": f"You have {moves_remaining} {move_word} to make with submit_move.",
    }


def build_messages(variant, board):
    """Give the model the position and rules, without the reference moves."""
    instructions = (
        "Play the best move in this chess position. You have 1 move to make with submit_move. "
        "Submit one legal move in UCI notation. Reward is 1.0 only if your move matches "
        "the stored Stockfish reference move; any other move scores 0.0. "
        "Consider candidate moves, forcing lines, and the opponent's strongest replies. "
        f"Make exactly one tool call per turn. Each response allows at most {MAX_OUTPUT_TOKENS} output tokens. "
    )
    if variant == "python":
        instructions += (
            f"You have at most {PYTHON_MAX_TURNS} model turns total, including Python calls and move submissions. "
            "Use python_chess to calculate with python-chess. Each call gets a fresh board at the current position. "
            "Reserve a turn to submit your final move."
        )
    else:
        instructions += "Use your own reasoning. No calculation tool is available."

    return [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": f"Play the best move. You play {'White' if board.turn else 'Black'}.\n"
            f"{position(board, 1)['instruction']}\n"
            f"FEN: {board.fen()}\n{board}",
        },
    ]


def play(client, trajectory_id, record, variant):
    """Allow calculation, then grade one submitted move against the frozen label."""
    if variant not in {"reasoning", "python"}:
        raise ValueError(f"Unknown variant: {variant}")

    board = chess.Board(record["fen"])
    solution = record["solution"]
    if len(solution) != 1:
        raise ValueError("Best-move tasks require one reference move")
    board.parse_uci(solution[0])
    max_steps = PYTHON_MAX_TURNS if variant == "python" else 1
    messages = build_messages(variant, board)
    tools = [SUBMIT_MOVE_TOOL]
    if variant == "python":
        tools.append(PYTHON_CHESS_TOOL)

    metrics = {
        "model_turns": 0,
        "python_calls": 0,
        "correct_moves": 0,
        "final_fen": board.fen(),
        "final_board": str(board),
        "moves_remaining": 1,
    }
    for _ in range(max_steps):
        metrics["model_turns"] += 1
        response = client.chat.completions.create(
            model="chess",
            messages=messages,
            x_trajectory_id=trajectory_id,
            max_tokens=MAX_OUTPUT_TOKENS,
            extra_body={
                "tools": tools,
                "parallel_tool_calls": False,
            },
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
        tool_call = calls[0]
        tool_function = tool_call["function"]
        try:
            arguments = json.loads(tool_function["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be an object")
            if tool_function["name"] == "submit_move":
                submitted_move = arguments["move"]
                move = board.parse_uci(submitted_move)
                correct = submitted_move == solution[0]
                board.push(move)
                metrics.update(
                    correct_moves=int(correct),
                    final_fen=board.fen(),
                    final_board=str(board),
                    moves_remaining=0,
                )
                return float(correct), "ENV_DONE", metrics
            elif tool_function["name"] == "python_chess" and variant == "python":
                metrics["python_calls"] += 1
                observation = execute_chess_python(board.fen(), {}, arguments["code"])
                if not observation["isolation_active"]:
                    raise RuntimeError(
                        f"Python isolation failed: {observation.get('error')}"
                    )
                observation.update(position(board, 1))
            else:
                raise ValueError("Unknown tool")
        except (ValueError, TypeError, KeyError):
            return 0.0, "ENV_DONE", metrics
        messages.append(assistant)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(observation),
            }
        )
    return 0.0, "MAX_STEPS", metrics


def main(variant=None):
    client = Client(max_retries=2)
    trajectory_id = client.trajectories.create().tid
    try:
        record = json.loads(os.environ["CHESS_POSITION"])
        variant = variant or os.environ["CHESS_VARIANT"]
        reward, reason, metrics = play(client, trajectory_id, record, variant)
        client.trajectories.log_event(
            trajectory_id,
            event_id="chess-result",
            name="chess_result",
            payload={
                "position_id": record["position_id"],
                "category": record["category"],
                "rating_band": record["rating_band"],
                "variant": variant,
                **metrics,
            },
        )
        client.trajectories.log_reward(
            trajectory_id, name="reward_accuracy", value=reward
        )
        for name in ("python_calls", "model_turns"):
            client.trajectories.log_reward(
                trajectory_id,
                name=name,
                value=metrics[name],
                min_value=0,
                max_value=PYTHON_MAX_TURNS,
                weight=0,
            )
    except Exception:
        client.trajectories.complete(trajectory_id, termination_reason="ERROR")
        raise
    completed = client.trajectories.complete(trajectory_id, termination_reason=reason)
    if completed.status not in {"completed", "finalizing"}:
        raise RuntimeError(f"Trajectory completion failed: {completed}")


if __name__ == "__main__":
    main()
