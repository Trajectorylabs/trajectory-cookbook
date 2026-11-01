"""Check judge credentials on the uploaded task spec without contacting a provider."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location("big_finance_ingest", ROOT / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


def test_default_vertex_judge_uses_vertex_credentials():
    task = ingest.build_benchmark(
        "finance", ["question"], "vertex:gemini-3.1-pro-preview"
    ).tasks[0]
    assert task.env_vars["VERTEXAI_CREDENTIALS"].secret_ref == "VERTEXAI_CREDENTIALS"


def test_one_flag_openrouter_judge_uses_openrouter_credentials():
    task = ingest.build_benchmark(
        "finance", ["question"], "openai:openrouter/openai/gpt-6-luna"
    ).tasks[0]
    assert task.env_vars["OPENROUTER_API_KEY"].secret_ref == "OPENROUTER_API_KEY"


def test_custom_secret_name_still_uses_provider_variable():
    task = ingest.build_benchmark(
        "finance", ["question"], "vertex:gemini-3.1-pro-preview", "MY_VERTEX_KEY"
    ).tasks[0]
    assert task.env_vars["VERTEXAI_CREDENTIALS"].secret_ref == "MY_VERTEX_KEY"
    assert "MY_VERTEX_KEY" not in task.env_vars
