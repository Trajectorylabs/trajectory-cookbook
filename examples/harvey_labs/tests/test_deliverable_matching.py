import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from lab_core.evaluation.judge import Judge
from lab_core.evaluation.scoring import _match_deliverables


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    judge = Judge(model="gpt-6-luna")
    judge.client = MagicMock()
    return judge


@pytest.fixture
def output_dir(tmp_path):
    for name in ("final-v2.txt", "scratch.txt"):
        (tmp_path / name).write_text(f"{name} contents")
    return tmp_path


def test_renamed_deliverable_is_matched_through_the_judge(judge, output_dir):
    judge.client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"memo.txt": "final-v2.txt"})
    )

    resolved = _match_deliverables(
        {"memo.txt": "memo.txt"}, ["final-v2.txt", "scratch.txt"], output_dir, judge
    )

    assert resolved == {"memo.txt": "final-v2.txt"}
    request = judge.client.responses.create.call_args.kwargs
    assert request["model"] == "gpt-6-luna"
    assert request["text"]["format"]["schema"]["required"] == ["memo.txt"]


def test_judge_failure_fails_matching_with_a_diagnostic(judge, output_dir, capsys):
    judge.client.responses.create.side_effect = RuntimeError("provider down")

    with pytest.raises(RuntimeError, match="provider down"):
        _match_deliverables(
            {"memo.txt": "memo.txt"}, ["final-v2.txt", "scratch.txt"], output_dir, judge
        )

    assert '"event": "judge_exhausted"' in capsys.readouterr().err


def test_matching_without_a_judge_raises(output_dir):
    with pytest.raises(ValueError, match="needs a judge"):
        _match_deliverables(
            {"memo.txt": "memo.txt"}, ["final-v2.txt", "scratch.txt"], output_dir
        )
