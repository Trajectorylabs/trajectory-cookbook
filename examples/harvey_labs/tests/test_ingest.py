import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", EXAMPLE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


@pytest.mark.parametrize(
    ("flag", "target", "agent_id", "agent_name"),
    [
        ("--agent-id", "agt_test", "agt_test", None),
        ("--agent-name", "test-agent", None, "test-agent"),
    ],
)
def test_cli_uploads_to_the_selected_agent(
    monkeypatch, flag, target, agent_id, agent_name
):
    client_factory = create_autospec(ingest.Client)
    client = client_factory.return_value.__enter__.return_value
    upload = create_autospec(
        ingest.push, return_value=SimpleNamespace(bench_id="bmk_test")
    )
    wait = create_autospec(ingest.wait_for_benchmark_images)
    monkeypatch.setattr(ingest, "Client", client_factory)
    monkeypatch.setattr(ingest, "push", upload)
    monkeypatch.setattr(ingest, "wait_for_benchmark_images", wait)
    monkeypatch.setattr(sys, "argv", ["ingest.py", flag, target])

    ingest.main()

    upload.assert_called_once()
    assert upload.call_args.args[0] is client
    assert upload.call_args.kwargs["agent_id"] == agent_id
    assert upload.call_args.kwargs["agent_name"] == agent_name
    wait.assert_called_once_with(client, "bmk_test", timeout_seconds=3600)


@pytest.mark.parametrize(
    "flags", [[], ["--agent-id", "agt_test", "--agent-name", "test-agent"]]
)
def test_cli_requires_exactly_one_agent_selector(monkeypatch, flags):
    client_factory = create_autospec(ingest.Client)
    monkeypatch.setattr(ingest, "Client", client_factory)
    monkeypatch.setattr(sys, "argv", ["ingest.py", *flags])

    with pytest.raises(SystemExit) as error:
        ingest.main()

    assert error.value.code == 2
    client_factory.assert_not_called()


def test_default_judges_are_the_standard_pair_with_both_keys():
    task = ingest.build_benchmark("harvey").tasks[0]

    assert task.run_command.endswith("--judges claude-sonnet-4-6 gpt-5.5")
    assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} <= task.env_vars.keys()


def test_openrouter_judge_needs_only_the_openai_key_and_model_flag():
    task = ingest.build_benchmark("harvey", ("openai/gpt-6-luna",)).tasks[0]

    assert task.run_command.endswith("--judges openai/gpt-6-luna")
    assert "ANTHROPIC_API_KEY" not in task.env_vars
    assert task.env_vars["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


@pytest.mark.parametrize(
    "judges",
    [
        ("openai/gpt-6-luna", "gpt-5.5"),
        ("openai/gpt-6-luna", "claude-sonnet-4-6"),
        ("meta-llama/llama-3.3-70b-instruct", "claude-sonnet-4-6"),
    ],
)
def test_provider_qualified_judge_cannot_run_in_dual_mode(judges):
    with pytest.raises(ValueError, match="supported only in single-judge runs"):
        ingest.build_benchmark("harvey", judges, judge_base_url="https://openrouter.ai/api/v1")


@pytest.mark.parametrize("judges", [(), ("gpt-5.5",) * 2, ("gpt-5.5", "claude-sonnet-4-6", "gemini-3-flash-preview")])
def test_invalid_judge_selection_fails_before_upload(judges):
    with pytest.raises(ValueError, match="--judges"):
        ingest.build_benchmark("harvey", judges)


def test_unknown_bare_judge_requires_a_base_url():
    with pytest.raises(ValueError, match="--judge-base-url"):
        ingest.build_benchmark("harvey", ("llama-3.3",))

    task = ingest.build_benchmark(
        "harvey", ("llama-3.3",), judge_base_url="https://example.com/v1"
    ).tasks[0]
    assert task.env_vars["OPENAI_BASE_URL"] == "https://example.com/v1"


@pytest.mark.parametrize("base_url", ["", "  "])
def test_empty_judge_base_url_fails_before_upload(base_url):
    with pytest.raises(ValueError, match="--judge-base-url must not be empty"):
        ingest.build_benchmark("harvey", ("openai/gpt-6-luna",), base_url)


def test_judge_id_is_shell_quoted():
    task = ingest.build_benchmark(
        "harvey", ("gpt-5.5; echo unsafe",), judge_base_url="https://example.com/v1"
    ).tasks[0]
    assert task.run_command.endswith("--judges 'gpt-5.5; echo unsafe'")


def test_dockerfile_downloads_source_commit():
    dockerfile = (EXAMPLE / "runtime" / "Dockerfile").read_text()

    assert f"harvey-labs/tar.gz/{ingest.SOURCE_COMMIT}" in dockerfile


def test_each_judge_provider_gets_its_key():
    task = ingest.build_benchmark(
        "harvey", ("gemini-3-flash-preview", "mistral-medium-3.5")
    ).tasks[0]
    assert {"GEMINI_API_KEY", "MISTRAL_API_KEY"} <= task.env_vars.keys()
    assert "OPENAI_API_KEY" not in task.env_vars


def test_base_url_keeps_native_claude_key_alongside_openai_key():
    task = ingest.build_benchmark(
        "harvey", ("claude-sonnet-4-6", "gpt-5.5"), judge_base_url="https://example.com/v1"
    ).tasks[0]
    assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} <= task.env_vars.keys()
    assert task.env_vars["OPENAI_BASE_URL"] == "https://example.com/v1"
