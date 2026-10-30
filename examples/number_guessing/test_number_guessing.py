"""Offline checks of the game protocol; no model calls or credentials required."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx2
import pytest
from trajectory import Client

ROOT = Path(__file__).parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_module(
    "number_guessing_harness", ROOT / "runtime" / "number_guessing_harness.py"
)
ingest = load_module("number_guessing_ingest", ROOT / "ingest.py")


class ScriptedClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        content, finish_reason = (
            output if isinstance(output, tuple) else (output, "stop")
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason=finish_reason,
                )
            ]
        )


def test_compaction_replaces_history_and_does_not_count_as_guess():
    summary = "The secret is in 38–49. Three guesses completed."
    client = ScriptedClient(["50", "25", "37", summary, "43", "42"])
    result = harness.run_game(client, "tid-test", 42, "same-model")
    assert result == {
        "solved": True,
        "guesses": 5,
        "compactions": 1,
        "reward": 0.25,
        "truncated": False,
    }
    assert len(client.calls) == 6
    assert all(
        call["model"] == "same-model" and call["x_trajectory_id"] == "tid-test"
        for call in client.calls
    )
    assert client.calls[3]["messages"][0]["content"] == harness.COMPACT_PROMPT
    assert client.calls[3]["max_tokens"] == 32_768
    assert client.calls[4]["max_tokens"] == 8_192
    assert "user: Higher." in client.calls[3]["messages"][1]["content"]
    assert client.calls[4]["messages"] == [
        {"role": "system", "content": harness.GUESS_PROMPT},
        {
            "role": "user",
            "content": f"Summary of the previous conversation:\n{summary}\n\nContinue guessing.",
        },
    ]


@pytest.mark.parametrize(
    "outputs, guesses, reward",
    [
        (["42"], 1, 1),
        (["24", "42"], 2, 1),
        (["50", "25", "42"], 3, 0.5),
        (["invalid", "24", "42"], 3, 0.5),
    ],
)
def test_reward_and_stop_before_compaction(outputs, guesses, reward):
    client = ScriptedClient(outputs)
    result = harness.run_game(client, "tid-test", 42, "same-model")
    assert result["guesses"] == guesses
    assert result["reward"] == reward
    assert result["compactions"] == 0
    assert len(client.calls) == guesses


def test_multiple_compactions_keep_environment_count():
    client = ScriptedClient(
        ["50", "25", "37", "First summary", "38", "39", "40", "Second summary", "42"]
    )
    result = harness.run_game(client, "tid-test", 42, "same-model")
    assert result == {
        "solved": True,
        "guesses": 7,
        "compactions": 2,
        "reward": 1 / 6,
        "truncated": False,
    }
    transcript = client.calls[7]["messages"][1]["content"]
    assert "Completed guesses according to the environment: 6." in transcript
    assert "First summary" in transcript


def test_unsolved_game_has_zero_reward_and_no_final_compaction():
    outputs = []
    for block in range(4):
        outputs.extend(["1"] * 3)
        if block < 3:
            outputs.append("Still in 2–100.")
    client = ScriptedClient(outputs)
    assert harness.run_game(client, "tid-test", 24, "same-model") == {
        "solved": False,
        "guesses": 12,
        "compactions": 3,
        "reward": 0.0,
        "truncated": False,
    }
    assert len(client.calls) == 15


@pytest.mark.parametrize(
    "outputs, guesses",
    [
        ([("4", "length")], 1),
        (["50", "25", "37", ("The secret is in", "max_tokens")], 3),
    ],
)
def test_cut_off_reply_ends_game_without_grading_it(outputs, guesses):
    client = ScriptedClient(outputs)
    result = harness.run_game(client, "tid-test", 42, "same-model")
    assert result == {
        "solved": False,
        "guesses": guesses,
        "compactions": 0,
        "reward": 0.0,
        "truncated": True,
    }
    assert len(client.calls) == len(outputs)


@pytest.mark.parametrize(
    "finish_reason, termination_reason",
    [("stop", "ENV_DONE"), ("length", "TRUNCATION")],
)
@pytest.mark.parametrize("completion_status", ["completed", "finalizing"])
def test_main_logs_reward_and_completes_with_sdk(
    tmp_path, monkeypatch, finish_reason, termination_reason, completion_status
):
    requests = []

    def respond(request):
        body = json.loads(request.content or b"{}")
        requests.append((request.url.path, body))
        if request.url.path == "/api/v1/trajectories":
            return httpx2.Response(200, json={"tid": "tid_test"})
        if request.url.path == "/v1/chat/completions":
            message = {"role": "assistant", "content": "42"}
            choice = {"index": 0, "finish_reason": finish_reason, "message": message}
            return httpx2.Response(
                200,
                json={
                    "id": "chat_test",
                    "model": "number-guessing",
                    "object": "chat.completion",
                    "created": 0,
                    "choices": [choice],
                },
            )
        if request.url.path.endswith("/rewards"):
            return httpx2.Response(
                200, json={"ok": True, "trajectory_id": "tid_test", "status": "active"}
            )
        if request.url.path.endswith("/complete"):
            return httpx2.Response(
                200, json={"trajectory_id": "tid_test", "status": completion_status}
            )
        raise AssertionError(request.url.path)

    task = tmp_path / "task.json"
    task.write_text(json.dumps({"secret": 42}))
    monkeypatch.setattr(sys, "argv", ["harness", "--task-file", str(task)])
    with Client(
        api_key="test",
        base_url="https://example.com",
        http_client=httpx2.Client(transport=httpx2.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(harness, "Client", lambda **kwargs: client)
        harness.main()
    assert len(requests) == 4
    assert requests[2][1]["value"] == (1.0 if finish_reason == "stop" else 0.0)
    assert requests[3][1] == {"termination_reason": termination_reason}


def test_dataset_is_balanced_and_secrets_stay_in_task_files(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    benchmark = ingest.build_benchmark("test")
    for split, expected_count in (("train", 32), ("test", 16)):
        tasks = [task for task in benchmark.tasks if task.split == split]
        assert len(tasks) == expected_count
        secrets = [
            json.loads(
                (
                    tmp_path / "runtime" / "tasks" / f"{split}_{index:04d}.json"
                ).read_text()
            )["secret"]
            for index in range(expected_count)
        ]
        assert secrets.count(24) == secrets.count(42) == expected_count // 2
        assert all(not task.env_vars for task in tasks)
