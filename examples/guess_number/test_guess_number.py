import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = load_module("guess_number", ROOT / "runtime" / "guess_number.py")
ingest = load_module("guess_number_ingest", ROOT / "ingest.py")


class FakeCompletions:
    def __init__(self, outputs: list[str]):
        self.outputs = iter(outputs)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content=next(self.outputs)))
            ]
        )


class FakeClient:
    def __init__(self, outputs: list[str]):
        self.chat = SimpleNamespace(completions=FakeCompletions(outputs))


@pytest.mark.parametrize(
    ("outputs", "guesses", "reward"),
    [
        (["42"], 1, 1.0),
        (["50", "42"], 2, 1.0),
        (["invalid", "60", "42"], 3, 0.5),
    ],
)
def test_game_preserves_history_without_compaction(outputs, guesses, reward):
    client = FakeClient(outputs)

    result = harness.run_game(client, "trj_test", 42, "guess-number")

    assert result == {"solved": True, "guesses": guesses, "reward": reward}
    assert len(client.chat.completions.calls) == guesses
    assert all(
        call["x_trajectory_id"] == "trj_test" for call in client.chat.completions.calls
    )
    assert all(
        call["max_tokens"] == harness.MAX_TOKENS
        for call in client.chat.completions.calls
    )
    message_counts = [len(call["messages"]) for call in client.chat.completions.calls]
    assert message_counts == list(range(2, 2 * guesses + 1, 2))


def test_ingest_builds_both_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "ROOT", tmp_path)

    benchmark = ingest.build_benchmark("guess-number-test")

    assert len(benchmark.tasks) == 48
    assert sum(task.split == "train" for task in benchmark.tasks) == 32
    assert sum(task.split == "test" for task in benchmark.tasks) == 16
    assert all("context-compaction" not in task.tags for task in benchmark.tasks)
