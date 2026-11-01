import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", EXAMPLE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


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
