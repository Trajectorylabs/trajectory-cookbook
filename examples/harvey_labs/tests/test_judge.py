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
    (tmp_path / "final-v2.txt").write_text("{not a template field}")
    (tmp_path / "scratch.txt").write_text("scratch")
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
    assert "Preview: {not a template field}" in request["input"]
    assert request["text"]["format"]["schema"]["required"] == ["memo.txt"]


def test_judge_failure_fails_matching(judge, output_dir):
    judge.client.responses.create.side_effect = RuntimeError("provider down")

    with pytest.raises(ValueError, match="provider down"):
        _match_deliverables(
            {"memo.txt": "memo.txt"}, ["final-v2.txt", "scratch.txt"], output_dir, judge
        )


def test_base_url_sends_prefixed_model_ids_to_the_openai_client(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    judge = Judge(model="openai/gpt-6-luna")
    judge.client = MagicMock()
    judge.client.responses.create.return_value = SimpleNamespace(
        output_text=json.dumps({"reasoning": "ok", "verdict": "pass"})
    )

    assert judge.evaluate("Grade {x}", {"x": "this"})["verdict"] == "pass"
    assert (
        judge.client.responses.create.call_args.kwargs["model"] == "openai/gpt-6-luna"
    )


def test_base_url_keeps_claude_judges_on_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    assert Judge(model="claude-sonnet-4-6").provider == "anthropic"


def test_without_base_url_unknown_prefixes_are_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="Unknown judge provider"):
        Judge(model="openai/gpt-6-luna")
