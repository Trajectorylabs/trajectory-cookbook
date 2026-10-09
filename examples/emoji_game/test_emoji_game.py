"""Offline checks of the emoji reward; no model calls or credentials required."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        ("🌞🏖️🌊", 3, 0.7),
        ("Beach day 🏖️", 4, -0.05),
        ("👨‍👩‍👧 dinner", 8, 1 / 8 - 0.3),  # A ZWJ sequence counts as one emoji.
        ("", 0, -0.3),
    ],
)
def test_score(text, tokens, reward):
    assert harness.score(text, tokens) == pytest.approx(reward)


def test_play_uses_completion_tokens_and_hides_objective():
    client = ScriptedClient("Sun 🌞 and sea 🌊", 10)
    result = harness.play(client, "tid-test", "the ocean", "same-model")
    assert result == {
        "emojis": 2,
        "completion_tokens": 10,
        "reward": pytest.approx(-0.1),
    }
    (call,) = client.calls
    assert call["x_trajectory_id"] == "tid-test"
    assert call["messages"] == [
        {"role": "user", "content": "Write a short paragraph about the ocean."}
    ]
    assert "emoji" not in harness.PROMPT.lower()


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
