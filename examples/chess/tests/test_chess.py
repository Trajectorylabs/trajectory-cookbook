"""Check record semantics, actor limits, and private Python boundaries."""

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
def record():
    return next(
        json.loads(line)
        for line in (ROOT / "data/positions.jsonl").read_text().splitlines()
        if json.loads(line)["category"] == "tactics"
        and json.loads(line)["split"] == "test"
    )


def response(name=None, arguments=None, finish="tool_calls"):
    calls = (
        []
        if name is None
        else [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ]
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish,
                message=SimpleNamespace(
                    model_dump=lambda **_: {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": calls,
                    },
                ),
            )
        ]
    )


def client_for(responses):
    requests = []
    iterator = iter(responses)

    def create(**kwargs):
        requests.append(copy.deepcopy(kwargs))
        return next(iterator)

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    ), requests


def test_frozen_annotations_agree_and_are_legal(record):
    assert dataset.validate_position(record) is record
    assert record["eval_recheck"]["limit_nodes"] == 2_000_000
    assert record["eval_recheck"]["pv"][0] == record["solution"][0]


@pytest.mark.parametrize("variant", ["reasoning", "python"])
def test_one_best_move_is_graded_without_label_leakage(record, variant):
    client, requests = client_for(
        [response("submit_move", {"move": record["solution"][0]})]
    )
    reward, reason, metrics = harness.play(client, "tid", record, variant)
    assert (reward, reason) == (1.0, "ENV_DONE")
    assert metrics["correct_moves"] == 1 and metrics["moves_remaining"] == 0
    assert metrics["model_turns"] == 1
    initial = json.dumps(requests[0]["messages"])
    assert record["solution"][0] not in initial
    assert "filter_passes" not in initial and "eval_recheck" not in initial
    assert "You have 1 move to make with submit_move." in initial
    assert requests[0]["max_tokens"] == 4000
    offered = [t["function"]["name"] for t in requests[0]["extra_body"]["tools"]]
    assert ("python_chess" in offered) == (variant == "python")
    board = chess.Board(record["fen"])
    board.push_uci(record["solution"][0])
    assert (
        metrics["final_fen"] == board.fen()
    )  # No automatic opponent reply for a one-move task.


@pytest.mark.parametrize(
    "first,ending",
    [
        (response("submit_move", {"move": "a2b3"}), "ENV_DONE"),  # Legal, wrong.
        (response("submit_move", {"move": "a1a8"}), "ENV_DONE"),  # Illegal.
        (response("submit_move", {"move": None}), "ENV_DONE"),
        (response(finish="stop"), "MISSING_SUBMIT"),
        (
            response("python_chess", {"code": "print(board)"}, finish="length"),
            "TRUNCATION",
        ),
    ],
)
def test_wrong_missing_and_cut_off_moves(record, first, ending, monkeypatch):
    monkeypatch.setattr(
        harness, "execute_chess_python", lambda *_: pytest.fail("Tool must not run")
    )
    client, _ = client_for([first])
    assert harness.play(client, "tid", record, "reasoning")[:2] == (0.0, ending)


def test_python_hundred_step_limit_and_no_reference_payload(record, monkeypatch):
    calls = []

    def execute(fen, root_moves, code):
        calls.append((fen, root_moves, code))
        return {"isolation_active": True, "stdout": "ok", "exit_code": 0}

    monkeypatch.setattr(harness, "execute_chess_python", execute)
    client, requests = client_for(
        [response("python_chess", {"code": "print(board)"}) for _ in range(100)]
    )
    reward, ending, metrics = harness.play(client, "tid", record, "python")
    assert (reward, ending) == (0.0, "MAX_STEPS")
    assert len(requests) == metrics["python_calls"] == 100
    assert calls == [(record["fen"], {}, "print(board)")] * 100
    assert "at most 100 model turns total" in requests[0]["messages"][0]["content"]


def test_isolation_failure_is_an_error(record, monkeypatch):
    monkeypatch.setattr(
        harness, "execute_chess_python", lambda *_: {"isolation_active": False}
    )
    client, _ = client_for([response("python_chess", {"code": "print(board)"})])
    with pytest.raises(RuntimeError, match="isolation failed"):
        harness.play(client, "tid", record, "python")


def test_python_can_complete_on_the_hundredth_turn(record, monkeypatch):
    monkeypatch.setattr(
        harness, "execute_chess_python", lambda *_: {"isolation_active": True}
    )
    responses = [response("python_chess", {"code": "print(board)"}) for _ in range(99)]
    responses += [response("submit_move", {"move": record["solution"][0]})]
    client, requests = client_for(responses)
    reward, ending, metrics = harness.play(client, "tid", record, "python")
    assert (reward, ending) == (1.0, "ENV_DONE")
    assert len(requests) == metrics["model_turns"] == 100
    assert metrics["python_calls"] == 99


def test_variants_have_matching_source_tasks(record):
    left = ingest.build_benchmark([record], "reasoning")
    right = ingest.build_benchmark([record], "python")
    assert (
        left.tasks[0].env_vars["CHESS_POSITION"]
        == right.tasks[0].env_vars["CHESS_POSITION"]
    )
    assert left.tasks[0].split == right.tasks[0].split == "test"
    assert (ROOT / "harness" / left.runtime.source.dockerfile_path).exists()


def test_frozen_dataset():
    path = ROOT / "data" / "positions.jsonl"
    if not path.exists():
        pytest.skip("Build the dataset to audit all selected puzzles")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    dataset.audit(records)
    assert len(records) == 5100


@pytest.mark.parametrize(
    "mutation", ["best_move", "illegal_pv", "missing_recheck", "rating"]
)
def test_corrupt_annotations_are_rejected(record, mutation):
    broken = copy.deepcopy(record)
    if mutation == "best_move":
        broken["filter_passes"][0]["lines"][0]["move"] = "a1a2"
    elif mutation == "illegal_pv":
        broken["filter_passes"][0]["lines"][0]["pv"].append("a1a1")
    elif mutation == "missing_recheck":
        del broken["eval_recheck"]
    else:
        broken["player_rating"] += 100
    with pytest.raises(ValueError):
        dataset.validate_position(broken)


@pytest.mark.parametrize("ending", ["ENV_DONE", "TRUNCATION", "ERROR"])
def test_runtime_uses_published_sdk_reward_event_and_completion(
    record, monkeypatch, ending
):
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path.endswith("/trajectories"):
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path.endswith("/complete"):
            return httpx2.Response(
                200, json={"trajectory_id": "tid_test", "status": "completed"}
            )
        return httpx2.Response(200, json={})

    def play(*_):
        if ending == "ERROR":
            raise RuntimeError("Actor unavailable")
        return (
            (1.0 if ending == "ENV_DONE" else 0.0),
            ending,
            {
                "python_calls": 0,
                "model_turns": 1,
            },
        )

    monkeypatch.setenv("CHESS_POSITION", json.dumps(record))
    monkeypatch.setenv("CHESS_VARIANT", "reasoning")
    monkeypatch.setattr(harness, "play", play)
    with Client(
        api_key="test",
        base_url="https://example.com",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
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
        for name in ("python_calls", "model_turns"):
            metric = next(body for _, body in requests if body.get("name") == name)
            assert metric["weight"] == 0


def test_legal_alternative_scores_zero(record):
    board = chess.Board(record["fen"])
    alternative = next(
        move.uci() for move in board.legal_moves if move.uci() != record["solution"][0]
    )
    client, _ = client_for([response("submit_move", {"move": alternative})])
    reward, ending, metrics = harness.play(client, "tid", record, "reasoning")
    assert (reward, ending, metrics["correct_moves"]) == (0.0, "ENV_DONE", 0)


def test_shared_hundred_position_subset():
    records = [
        json.loads(line)
        for line in (ROOT / "data/positions.jsonl").read_text().splitlines()
    ]
    selected = dataset.evaluation_subset(records)
    assert selected == dataset.evaluation_subset(list(reversed(records)))
    assert len(selected) == 100 and all(row["split"] == "test" for row in selected)
    from collections import Counter

    assert Counter(row["category"] for row in selected) == {
        category: 25 for category in dataset.CATEGORIES
    }
    dataset.audit(records)
    duplicate = copy.deepcopy(records[0])
    with pytest.raises(ValueError, match="Duplicate"):
        dataset.audit(records + [duplicate])
    overlap = copy.deepcopy(records)
    train_game = next(
        row["source"]["game"] for row in overlap if row["split"] == "train"
    )
    next(row for row in overlap if row["split"] == "test")["source"]["game"] = (
        train_game
    )
    with pytest.raises(ValueError, match="overlaps"):
        dataset.audit(overlap)


@pytest.mark.parametrize(
    "c,k,expected",
    [
        (0, 1, 0),
        (4, 4, 1),
        (1, 1, 0.25),
        (1, 2, 0.5),
        (1, 3, 0.75),
        (1, 4, 1),
        (2, 2, 5 / 6),
    ],
)
def test_pass_at_k(c, k, expected):
    assert train.pass_at_k(4, c, k) == pytest.approx(expected)


def test_scores_and_usage_are_averaged_by_position_and_category():
    tasks = {
        f"t{i}": {"category": dataset.CATEGORIES[i // 25], "rating_band": "2000-2199"}
        for i in range(100)
    }
    outcomes = [
        {
            "task_id": tid,
            "solved": tid == "t4" and attempt == 0,
            "python_calls": 2,
            "model_turns": 3,
        }
        for tid in tasks
        for attempt in range(4)
    ]
    report = train.summarize(outcomes, tasks)
    assert report["pass_at_k"]["1"] == pytest.approx(0.0025)
    assert report["pass_at_k"]["4"] == pytest.approx(0.01)
    assert report["mean_python_calls"] == 2
    assert report["mean_model_turns"] == 3
    assert report["breakdowns"]["category"]["openings"]["pass_at_k"][
        "4"
    ] == pytest.approx(0.04)
    with pytest.raises(ValueError, match="exactly four"):
        train.summarize(outcomes[:-1], tasks)


def test_twenty_jobs_are_preflighted_before_the_first_evaluation(monkeypatch, tmp_path):
    checked, submitted = [], []

    def tasks(_, bid):
        return {
            f"{bid}-{i}": {"position_id": str(i), "variant": bid} for i in range(100)
        }

    def check(*args, **kwargs):
        checked.append(args)

    def submit(*args):
        assert len(checked) == 20
        submitted.append(args)

    monkeypatch.setattr(evaluate, "evaluation_tasks", tasks)
    monkeypatch.setattr(evaluate, "model_options", check)
    monkeypatch.setattr(evaluate, "evaluate", submit)
    evaluate.run_matrix(
        object(), {"reasoning": "reasoning", "python": "python"}, tmp_path
    )
    assert len(submitted) == 20
    assert len(evaluate.OPEN_MODELS) == 6 and len(evaluate.CLOSED_MODELS) == 4
    assert (
        train.EVAL_OPTIONS["val_num_tasks"]
        * train.EVAL_OPTIONS["val_trajectories_per_task"]
        == 400
    )


def test_reference_replay_is_labeled_and_contains_one_submission(record):
    from trace import reference_trace

    record = reference_trace(record)
    assert record["kind"] == "reference_replay" and record["model_calls"] == 0
    assert [message["role"] for message in record["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert record["result"]["reward"] == 1.0
    assert record["result"]["moves_remaining"] == 0
    assert record["result"]["model_turns"] == 1


def test_old_multi_move_tasks_are_rejected(record):
    old = copy.deepcopy(record)
    old["solution"] *= 2
    client, requests = client_for([])
    with pytest.raises(ValueError, match="one reference move"):
        harness.play(client, "tid", old, "reasoning")
    assert not requests


def test_python_exploration_preserves_root_board(record, monkeypatch):
    def calculate(fen, private, code):
        board = chess.Board(fen)
        board.push(next(iter(board.legal_moves)))
        return {"isolation_active": True, "stdout": board.fen()}

    monkeypatch.setattr(harness, "execute_chess_python", calculate)
    client, requests = client_for(
        [
            response(
                "python_chess", {"code": "board.push(next(iter(board.legal_moves)))"}
            ),
            response("submit_move", {"move": record["solution"][0]}),
        ]
    )
    assert harness.play(client, "tid", record, "python")[:2] == (1.0, "ENV_DONE")
    observation = json.loads(requests[1]["messages"][-1]["content"])
    assert observation["fen"] == record["fen"]
    assert observation["moves_remaining"] == 1


def test_selected_file_matches_manifest():
    import hashlib

    data = (ROOT / "data/positions.jsonl").read_bytes()
    manifest = json.loads((ROOT / "data/positions.manifest.json").read_text())
    assert hashlib.sha256(data).hexdigest() == manifest["sha256"]
    assert manifest["count"] == 5100
    assert all(
        value["counts"] == {"train": 50000, "test": 250}
        for value in manifest["source_files"].values()
    )


def test_uploaded_eval_tasks_retain_four_category_split():
    records = [
        json.loads(line)
        for line in (ROOT / "data/positions.jsonl").read_text().splitlines()
    ]
    records = dataset.evaluation_subset(records)
    spec = ingest.build_benchmark(records, "reasoning")
    tasks = [
        SimpleNamespace(task_id=f"t{i}", env_vars=task.env_vars)
        for i, task in enumerate(spec.tasks)
    ]
    client = SimpleNamespace(
        benchmarks=SimpleNamespace(tasks=SimpleNamespace(list=lambda *_, **__: tasks))
    )
    metadata = train.evaluation_tasks(client, "benchmark")
    assert len(metadata) == 100
    assert {row["category"] for row in metadata.values()} == set(dataset.CATEGORIES)
    assert all(row["variant"] == "reasoning" for row in metadata.values())


@pytest.mark.parametrize("missing_metric", [False, True])
def test_eval_collection_reads_new_usage_metrics(missing_metric):
    tasks = {
        f"t{i}": {
            "position_id": str(i),
            "category": dataset.CATEGORIES[i // 25],
            "rating_band": "2000-2199",
        }
        for i in range(100)
    }
    attempts = [
        SimpleNamespace(
            task_id=task_id,
            trajectory_id=f"{task_id}-{index}",
            rollout=SimpleNamespace(status="COMPLETED"),
        )
        for task_id in tasks
        for index in range(4)
    ]
    components = [
        SimpleNamespace(name="reward_accuracy", value=1.0),
        SimpleNamespace(name="model_turns", value=2),
    ]
    if not missing_metric:
        components.append(SimpleNamespace(name="python_calls", value=1))
    runs = SimpleNamespace(
        progress=lambda _: SimpleNamespace(
            status="completed", terminal_rollouts=400, total_rollouts=400
        ),
        retrieve=lambda _: SimpleNamespace(reward_mean=1.0),
    )
    trajectories = SimpleNamespace(
        list=lambda **_: attempts,
        retrieve=lambda *_, **__: SimpleNamespace(
            termination_reason="ENV_DONE", reward_components=components
        ),
    )
    client = SimpleNamespace(
        evals=SimpleNamespace(runs=runs), trajectories=trajectories
    )
    if missing_metric:
        with pytest.raises(RuntimeError, match="tool-usage metrics"):
            train.wait_evaluation(client, "eval", tasks)
    else:
        report = train.wait_evaluation(client, "eval", tasks)
        assert report["graded"] == 400
        assert report["mean_python_calls"] == 1 and report["mean_model_turns"] == 2
        assert report["breakdowns"]["category"]["tactics"]["positions"] == 25
        assert report["breakdowns"]["category"]["tactics"]["attempts"] == 100


def test_training_uses_same_labels_options_and_requested_checkpoint(
    monkeypatch, tmp_path
):
    evaluations, creates, checkpoints = [], [], []

    def evaluate_checkpoint(_, bench_id, model, path, checkpoint_id=None):
        evaluations.append(checkpoint_id)
        return {
            "position_ids": ["same-position"],
            "variant": "reasoning",
            "options": {"temperature": 1.0, "val_num_tasks": 100},
            "report": {"pass_at_k": {str(k): 0.5 for k in range(1, 5)}},
        }

    def create(**kwargs):
        creates.append(kwargs)
        return SimpleNamespace(training_run_id="training")

    def checkpoint(run_id, step):
        checkpoints.append((run_id, step))
        return SimpleNamespace(checkpoint_id="checkpoint-100")

    client = SimpleNamespace(
        training=SimpleNamespace(
            create=create, checkpoints=SimpleNamespace(retrieve=checkpoint)
        )
    )
    monkeypatch.setattr(train, "evaluate", evaluate_checkpoint)
    monkeypatch.setattr(
        train,
        "model_options",
        lambda *_, **options: {
            key: value for key, value in options.items() if key != "training"
        },
    )
    monkeypatch.setattr(train, "wait_training", lambda *_, **__: 100)
    state = train.run(client, "new-benchmark", tmp_path / "comparison.json")
    assert evaluations == [None, "checkpoint-100"]
    assert checkpoints == [("training", 100)]
    options = creates[0]["options"]
    assert (options["num_steps"], options["batch_size"], options["group_size"]) == (
        100,
        16,
        8,
    )
    assert state["before"]["position_ids"] == state["after"]["position_ids"]
    assert set(state["pass_at_k_delta"].values()) == {0.0}
