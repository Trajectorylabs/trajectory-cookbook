"""Offline checks of the emoji reward; no model calls or credentials required."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from trajectory.resources.trajectories.trajectories import Trajectories

ROOT = Path(__file__).parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_module("emoji_game_harness", ROOT / "runtime" / "emoji_game_harness.py")
ingest = load_module("emoji_game_ingest", ROOT / "ingest.py")


class ScriptedClient:
    def __init__(self, output, completion_tokens):
        self.output = output
        self.usage = (
            None
            if completion_tokens is None
            else SimpleNamespace(completion_tokens=completion_tokens)
        )
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.output))],
            usage=self.usage,
        )


@pytest.mark.parametrize(
    "text, tokens, reward",
    [
        ("A sunny day at the beach.", 7, -0.3),
        ("😀✨😊", 3, 0.7),
        ("Beach day 😎", 4, -0.05),
        ("Love ❤️ and ❤", 6, 2 / 6 - 0.3),  # The variation selector is ignored.
        ("Beach day 🏖️", 4, -0.55),  # Emojis outside the allowed set are penalized.
        ("😀 🌊", 4, -0.3),
        ("👨‍👩‍👧 dinner", 8, -1 / 8 - 0.3),  # A ZWJ sequence counts as one emoji.
        ("Fine™", 2, -0.8),
        ("", 0, -0.3),
    ],
)
def test_score(text, tokens, reward):
    assert harness.score(text, tokens) == pytest.approx(reward)


def test_play_uses_completion_tokens_and_hides_objective():
    client = ScriptedClient("Sun 😎 and sea 🌊 ✨", 10)
    result = harness.play(client, "tid-test", "the ocean", "same-model")
    assert result == {
        "allowed_emojis": 2,
        "other_emojis": 1,
        "completion_tokens": 10,
        "reward": pytest.approx(-0.2),
    }
    (call,) = client.calls
    assert call["x_trajectory_id"] == "tid-test"
    assert call["messages"] == [
        {"role": "user", "content": "Write a short paragraph about the ocean. Use plenty of emojis."}
    ]
    assert not any(e in harness.PROMPT for e in harness.ALLOWED_EMOJIS)


def test_missing_usage_is_an_error():
    with pytest.raises(RuntimeError, match="completion tokens"):
        harness.play(ScriptedClient("🌞", None), "tid-test", "the ocean", "m")


def test_splits_use_disjoint_topics(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)
    benchmark = ingest.build_benchmark("test")
    for split, expected in (
        ("train", ingest.TRAIN_TOPICS),
        ("test", ingest.TEST_TOPICS),
    ):
        tasks = [task for task in benchmark.tasks if task.split == split]
        topics = [
            json.loads(
                (tmp_path / "runtime" / "tasks" / f"{split}_{i:04d}.json").read_text()
            )["topic"]
            for i in range(len(tasks))
        ]
        assert topics == expected
    assert len(ingest.TRAIN_TOPICS) == 32 and len(ingest.TEST_TOPICS) == 16
    assert not set(ingest.TRAIN_TOPICS) & set(ingest.TEST_TOPICS)


def test_main_reports_through_real_sdk_signatures(tmp_path, monkeypatch):
    """Autospec rejects keyword arguments the installed SDK does not accept."""
    task_file = tmp_path / "task.json"
    task_file.write_text(json.dumps({"topic": "the ocean"}))
    client = ScriptedClient("Sun 😎 and sea ✨", 10)
    client.trajectories = mock.create_autospec(Trajectories, instance=True)
    client.trajectories.create.return_value = SimpleNamespace(tid="tid-test")
    client.trajectories.complete.return_value = SimpleNamespace(status="completed")
    monkeypatch.setattr(harness, "Client", lambda **_: client)
    monkeypatch.setattr("sys.argv", ["harness", "--task-file", str(task_file)])

    harness.main()

    client.trajectories.log_reward.assert_called_once_with(
        "tid-test", name="reward_emoji_density", value=pytest.approx(-0.1)
    )
    client.trajectories.complete.assert_called_once_with(
        "tid-test", termination_reason="ENV_DONE"
    )
