import importlib.util
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", EXAMPLE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def test_default_judges_are_the_standard_pair_with_both_keys():
    task = ingest.build_benchmark("harvey").tasks[0]

    assert task.run_command.endswith("--judges claude-sonnet-4-6 gpt-5.5")
    assert {"ANTHROPIC_API_KEY", "OPENAI_API_KEY"} <= task.env_vars.keys()


def test_compatible_endpoint_judge_needs_only_the_openai_key():
    task = ingest.build_benchmark(
        "harvey", ("openai/gpt-6-luna",), "https://openrouter.ai/api/v1"
    ).tasks[0]

    assert task.run_command.endswith("--judges openai/gpt-6-luna")
    assert "ANTHROPIC_API_KEY" not in task.env_vars
    assert task.env_vars["OPENAI_BASE_URL"] == "https://openrouter.ai/api/v1"


def test_dockerfile_downloads_source_commit():
    dockerfile = (EXAMPLE / "runtime" / "Dockerfile").read_text()

    assert f"harvey-labs/tar.gz/{ingest.SOURCE_COMMIT}" in dockerfile
