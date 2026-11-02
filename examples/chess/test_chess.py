"""Check puzzle semantics, actor limits, and private Python boundaries."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import httpx2
from trajectory import Client

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runtime"))

import dataset
import chess_harness as harness
import ingest


@pytest.fixture
def puzzle():
    record = dataset.validate_puzzle({
        "PuzzleId": "00sHx", "FEN": "q3k1nr/1pp1nQpp/3p4/1P2p3/4P3/B1PP1b2/B5PP/5K2 b k - 0 17",
        "Moves": "e8d7 a2e6 d7d8 f7f8", "Rating": "1760", "Themes": "mate mateIn2 middlegame short",
        "GameUrl": "https://lichess.org/yyznGmXs/black#34",
    })
    assert record
    record["split"] = "train"
    return record


def response(name=None, arguments=None, finish="tool_calls"):
    calls = [] if name is None else [{"id": "call_1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments),
    }}]
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(
        model_dump=lambda **_: {"role": "assistant", "content": "", "tool_calls": calls},
    ))])


def client_for(responses):
    requests = []
    iterator = iter(responses)

    def create(**kwargs):
        requests.append(kwargs)
        return next(iterator)

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), requests


def test_setup_move_and_mate_length(puzzle):
    board = chess.Board(puzzle["source_fen"])
    board.push_uci(puzzle["source_moves"][0])
    assert puzzle["fen"] == board.fen()
    assert board.turn == chess.WHITE
    for uci in puzzle["solution"]:
        board.push_uci(uci)
    assert board.is_checkmate()
    assert dataset.game_id("https://lichess.org/game/yyznGmXs/black#34") == dataset.game_id(puzzle["game_url"])


@pytest.mark.parametrize("rating,band", [(-1,None),(0,"0-999"),(999,"0-999"),
    (1000,"1000-1499"),(1499,"1000-1499"),(1500,"1500-1999"),(1999,"1500-1999"),
    (2000,"2000-3000"),(3000,"2000-3000"),(3001,None)])
def test_rating_edges(rating, band):
    assert dataset.rating_band(rating) == band


@pytest.mark.parametrize("variant", ["reasoning", "python"])
def test_whole_puzzle_must_be_solved_and_no_solution_is_in_prompt(puzzle, variant):
    client, requests = client_for([response("submit_move", {"move": "a2e6"}),
                                   response("submit_move", {"move": "f7f8"})])
    reward, reason, metrics = harness.play(client, "tid", puzzle, variant)
    assert (reward, reason) == (1.0, "ENV_DONE")
    assert metrics["correct_moves"] == 2
    initial = json.dumps(requests[0]["messages"][0:2])
    assert "a2e6" not in initial and "f7f8" not in initial
    offered = [t["function"]["name"] for t in requests[0]["extra_body"]["tools"]]
    assert ("python_chess" in offered) == (variant == "python")


@pytest.mark.parametrize("first,ending", [
    (response("submit_move", {"move": "a2b3"}), "ENV_DONE"),  # Legal, wrong.
    (response("submit_move", {"move": "a1a8"}), "ENV_DONE"),  # Illegal.
    (response("submit_move", {"move": None}), "ENV_DONE"),
    (response(finish="stop"), "MISSING_SUBMIT"),
    (response("python_chess", {"code": "print(board)"}, finish="length"), "TRUNCATION"),
])
def test_wrong_missing_and_cut_off_moves(puzzle, first, ending, monkeypatch):
    monkeypatch.setattr(harness, "execute_chess_python", lambda *_: pytest.fail("Tool must not run"))
    client, _ = client_for([first])
    assert harness.play(client, "tid", puzzle, "reasoning")[:2] == (0.0, ending)


def test_python_ten_step_limit_and_no_reference_payload(puzzle, monkeypatch):
    calls = []

    def execute(fen, root_moves, code):
        calls.append((fen, root_moves, code))
        return {"isolation_active": True, "stdout": "ok", "exit_code": 0}

    monkeypatch.setattr(harness, "execute_chess_python", execute)
    client, requests = client_for([response("python_chess", {"code": "print(board)"}) for _ in range(10)])
    reward, ending, metrics = harness.play(client, "tid", puzzle, "python")
    assert (reward, ending) == (0.0, "MAX_STEPS")
    assert len(requests) == metrics["python_calls"] == 10
    assert calls == [(puzzle["fen"], {}, "print(board)")] * 10


def test_isolation_failure_is_an_error(puzzle, monkeypatch):
    monkeypatch.setattr(harness, "execute_chess_python", lambda *_: {"isolation_active": False})
    client, _ = client_for([response("python_chess", {"code": "print(board)"})])
    with pytest.raises(RuntimeError, match="isolation failed"):
        harness.play(client, "tid", puzzle, "python")


def test_variants_have_matching_source_tasks(puzzle):
    left = ingest.build_benchmark([puzzle], "reasoning")
    right = ingest.build_benchmark([puzzle], "python")
    assert left.tasks[0].env_vars["CHESS_PUZZLE"] == right.tasks[0].env_vars["CHESS_PUZZLE"]
    assert left.tasks[0].split == right.tasks[0].split == "train"
    assert (ROOT / "runtime" / left.runtime.source.dockerfile_path).exists()


def test_frozen_dataset():
    path = ROOT / "puzzles.jsonl"
    if not path.exists():
        pytest.skip("Build the dataset to audit all selected puzzles")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    dataset.audit(records)
    assert len(records) == 5500


def test_balancing_when_a_cell_is_scarce():
    available = {(mate, band): 1000 for mate in range(1, 6) for band in dataset.BANDS}
    available[5, "0-999"] = 9
    allocation = dataset.allocate(available, 1000)
    assert allocation[5, "0-999"] <= 9
    assert all(sum(allocation[m, b] for b in dataset.BANDS) == 1000 for m in range(1, 6))
    assert all(sum(allocation[m, b] for m in range(1, 6)) == 1250 for b in dataset.BANDS)
    with pytest.raises(ValueError, match="Not enough"):
        dataset.allocate({}, 1000)


@pytest.mark.parametrize("ending", ["ENV_DONE", "TRUNCATION", "ERROR"])
def test_runtime_uses_published_sdk_reward_event_and_completion(puzzle, monkeypatch, ending):
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path.endswith("/trajectories"):
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path.endswith("/complete"):
            return httpx2.Response(200, json={"trajectory_id": "tid_test", "status": "completed"})
        return httpx2.Response(200, json={})

    def play(*_):
        if ending == "ERROR":
            raise RuntimeError("Actor unavailable")
        return (1.0 if ending == "ENV_DONE" else 0.0), ending, {}

    monkeypatch.setenv("CHESS_PUZZLE", json.dumps(puzzle))
    monkeypatch.setenv("CHESS_VARIANT", "reasoning")
    monkeypatch.setattr(harness, "play", play)
    with Client(api_key="test", base_url="https://example.com", max_retries=0,
                http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
        monkeypatch.setattr(harness, "Client", lambda **_: client)
        if ending == "ERROR":
            with pytest.raises(RuntimeError, match="Actor unavailable"):
                harness.main()
        else:
            harness.main()
    assert requests[-1][1] == {"termination_reason": ending}
    reward_requests = [body for path, body in requests if path.endswith("/reward")]
    if ending == "ERROR":
        assert not reward_requests
    else:
        assert len(requests) == 4
        assert any(body.get("event_id") == "chess-result" for _, body in requests)
        assert any(body.get("name") == "reward_accuracy" for _, body in requests)
