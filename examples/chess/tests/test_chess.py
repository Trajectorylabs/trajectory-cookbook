"""Check puzzle semantics, actor limits, and private Python boundaries."""

import json
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import chess
import pytest
import httpx2
from trajectory import Client

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "harness"))

import dataset
import chess_harness as harness
import ingest
import train
import evaluate


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
        requests.append(copy.deepcopy(kwargs))
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
    assert metrics["correct_half_moves"] == 3
    expected = chess.Board(puzzle["fen"])
    expected.push_uci("a2e6")
    expected.push_uci("d7d8")
    observation = json.loads(requests[1]["messages"][-1]["content"])
    assert observation["opponent_move"] == "d7d8"
    assert observation["fen"] == expected.fen()
    assert observation["moves_remaining"] == 1
    assert observation["instruction"] == "You have 1 move to make with submit_move."
    assert "You have 2 moves to make with submit_move." in requests[0]["messages"][1]["content"]
    assert all(r["max_tokens"] == 4000 for r in requests)
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


def test_python_fifty_step_limit_and_no_reference_payload(puzzle, monkeypatch):
    calls = []

    def execute(fen, root_moves, code):
        calls.append((fen, root_moves, code))
        return {"isolation_active": True, "stdout": "ok", "exit_code": 0}

    monkeypatch.setattr(harness, "execute_chess_python", execute)
    client, requests = client_for([response("python_chess", {"code": "print(board)"}) for _ in range(50)])
    reward, ending, metrics = harness.play(client, "tid", puzzle, "python")
    assert (reward, ending) == (0.0, "MAX_STEPS")
    assert len(requests) == metrics["python_calls"] == 50
    assert calls == [(puzzle["fen"], {}, "print(board)")] * 50
    assert "at most 50 model turns total" in requests[0]["messages"][0]["content"]


def test_isolation_failure_is_an_error(puzzle, monkeypatch):
    monkeypatch.setattr(harness, "execute_chess_python", lambda *_: {"isolation_active": False})
    client, _ = client_for([response("python_chess", {"code": "print(board)"})])
    with pytest.raises(RuntimeError, match="isolation failed"):
        harness.play(client, "tid", puzzle, "python")


def test_python_can_complete_on_the_fiftieth_turn(puzzle, monkeypatch):
    monkeypatch.setattr(harness, "execute_chess_python", lambda *_: {"isolation_active": True})
    responses = [response("python_chess", {"code": "print(board)"}) for _ in range(48)]
    responses += [response("submit_move", {"move": "a2e6"}), response("submit_move", {"move": "f7f8"})]
    client, requests = client_for(responses)
    reward, ending, metrics = harness.play(client, "tid", puzzle, "python")
    assert (reward, ending) == (1.0, "ENV_DONE")
    assert len(requests) == metrics["model_turns"] == 50
    assert metrics["python_calls"] == 48


def test_variants_have_matching_source_tasks(puzzle):
    left = ingest.build_benchmark([puzzle], "reasoning")
    right = ingest.build_benchmark([puzzle], "python")
    assert left.tasks[0].env_vars["CHESS_PUZZLE"] == right.tasks[0].env_vars["CHESS_PUZZLE"]
    assert left.tasks[0].split == right.tasks[0].split == "train"
    assert (ROOT / "harness" / left.runtime.source.dockerfile_path).exists()


def test_frozen_dataset():
    path = ROOT / "data" / "puzzles.jsonl"
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
        return (1.0 if ending == "ENV_DONE" else 0.0), ending, {
            "correct_half_moves": 3 if ending == "ENV_DONE" else 0,
            "correct_moves": 2 if ending == "ENV_DONE" else 0,
        }

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
        assert len(requests) == 6
        assert any(body.get("event_id") == "chess-result" for _, body in requests)
        assert any(body.get("name") == "reward_accuracy" for _, body in requests)
        for name in ("correct_half_moves", "correct_solver_moves"):
            metric = next(body for _, body in requests if body.get("name") == name)
            assert metric["weight"] == 0


def test_another_legal_checkmate_is_rejected():
    puzzle = {"fen": "7k/5K2/6Q1/8/8/8/8/8 w - - 0 1", "mate_in": 1, "solution": ["g6g7"]}
    for uci in ("g6g7", "g6g8"):
        board = chess.Board(puzzle["fen"])
        board.push_uci(uci)
        assert board.is_checkmate()
    client, _ = client_for([response("submit_move", {"move": "g6g8"})])
    reward, ending, metrics = harness.play(client, "tid", puzzle, "reasoning")
    assert (reward, ending, metrics["correct_moves"]) == (0.0, "ENV_DONE", 0)


@pytest.mark.parametrize("variant", ["reasoning", "python"])
def test_five_move_continuation_plays_every_opponent_reply(variant):
    puzzle = next(json.loads(line) for line in (ROOT / "data" / "puzzles.jsonl").read_text().splitlines()
                  if json.loads(line)["puzzle_id"] == "eYtxb")
    line = puzzle["solution"]
    client, requests = client_for([response("submit_move", {"move": uci}) for uci in line[::2]])
    reward, ending, metrics = harness.play(client, "tid", puzzle, variant)
    assert (reward, ending) == (1.0, "ENV_DONE")
    assert metrics["correct_moves"] == 5 and metrics["correct_half_moves"] == 9
    board = chess.Board(puzzle["fen"])
    for index, request in enumerate(requests[1:]):
        board.push_uci(line[index * 2])
        board.push_uci(line[index * 2 + 1])
        observation = json.loads(request["messages"][-1]["content"])
        assert observation["opponent_move"] == line[index * 2 + 1]
        assert observation["fen"] == board.fen()


def test_wrong_second_move_only_keeps_the_correct_prefix(puzzle):
    client, _ = client_for([response("submit_move", {"move": "a2e6"}),
                            response("submit_move", {"move": "f7f6"})])
    reward, ending, metrics = harness.play(client, "tid", puzzle, "python")
    assert reward == 0.0 and ending == "ENV_DONE"
    assert metrics["correct_moves"] == 1 and metrics["correct_half_moves"] == 2


def test_shared_balanced_hundred_puzzle_subset():
    records = [json.loads(line) for line in (ROOT / "data" / "puzzles.jsonl").read_text().splitlines()]
    selected = dataset.evaluation_subset(records)
    assert selected == dataset.evaluation_subset(list(reversed(records)))
    assert len(selected) == 100
    assert all(r["split"] == "test" for r in selected)
    from collections import Counter
    assert Counter(r["mate_in"] for r in selected) == {m: 20 for m in range(1, 6)}
    assert Counter(r["rating_band"] for r in selected) == {b: 25 for b in dataset.BANDS}
    dataset.audit([r for r in records if r["split"] == "train"] + selected, test_per_category=20)


@pytest.mark.parametrize("c,k,expected", [(0,1,0), (4,4,1), (1,1,.25), (1,2,.5), (1,3,.75), (1,4,1), (2,2,5/6)])
def test_pass_at_k(c, k, expected):
    assert train.pass_at_k(4, c, k) == pytest.approx(expected)


def test_scores_and_prefixes_are_averaged_by_puzzle_and_category():
    tasks = {f"t{i}": {"mate_in": i % 5 + 1, "rating_band": dataset.BANDS[i // 25]} for i in range(100)}
    outcomes = [{"task_id": tid, "solved": tid == "t4" and attempt == 0,
                 "correct_half_moves": 9 if tid == "t4" and attempt == 0 else 0,
                 "correct_solver_moves": 5 if tid == "t4" and attempt == 0 else 0}
                for tid in tasks for attempt in range(4)]
    report = train.summarize(outcomes, tasks)
    assert report["pass_at_k"]["1"] == pytest.approx(.0025)
    assert report["pass_at_k"]["4"] == pytest.approx(.01)
    assert report["mean_correct_half_moves"] == pytest.approx(9/400)
    assert report["breakdowns"]["mate_in"]["5"]["mean_correct_half_moves"] == pytest.approx(9/80)
    assert report["breakdowns"]["rating_band"]["0-999"]["mean_correct_half_moves"] == pytest.approx(9/100)
    with pytest.raises(ValueError, match="exactly four"):
        train.summarize(outcomes[:-1], tasks)


def test_twenty_jobs_are_preflighted_before_the_first_evaluation(monkeypatch, tmp_path):
    checked, submitted = [], []
    def tasks(_, bid):
        return {f"{bid}-{i}": {"puzzle_id": str(i), "variant": bid} for i in range(100)}
    def check(*args, **kwargs):
        checked.append(args)
    def submit(*args):
        assert len(checked) == 20
        submitted.append(args)
    monkeypatch.setattr(evaluate, "evaluation_tasks", tasks)
    monkeypatch.setattr(evaluate, "model_options", check)
    monkeypatch.setattr(evaluate, "evaluate", submit)
    evaluate.run_matrix(object(), {"reasoning": "reasoning", "python": "python"}, tmp_path)
    assert len(submitted) == 20
    assert len(evaluate.OPEN_MODELS) == 6 and len(evaluate.CLOSED_MODELS) == 4
    assert train.EVAL_OPTIONS["val_num_tasks"] * train.EVAL_OPTIONS["val_trajectories_per_task"] == 400


def test_reference_replay_is_labeled_and_contains_all_visible_turns(puzzle):
    from trace import reference_trace
    record = reference_trace(puzzle)
    assert record["kind"] == "reference_replay" and record["model_calls"] == 0
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert json.loads(record["messages"][3]["content"])["moves_remaining"] == 1
    assert record["result"]["reward"] == 1.0
    assert record["result"]["moves_remaining"] == 0
    assert chess.Board(record["result"]["final_fen"]).is_checkmate()
