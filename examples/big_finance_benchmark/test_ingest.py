"""Check how ingest.py stages an existing checkout of the harness."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
SUBSET = '{"id": "bf-1"}\n{"id": "bf-2"}\n'


@pytest.fixture
def ingest():
    spec = importlib.util.spec_from_file_location(
        "big_finance_ingest", ROOT / "ingest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(checkout, *args):
    command = ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@t"]
    return subprocess.run([*command, *args], check=True, capture_output=True, text=True)


@pytest.fixture
def checkout(tmp_path):
    files = {
        "LICENSE": "Apache",
        "README.md": "readme",
        "pyproject.toml": "[project]",
        "big_finance_harness/__init__.py": "",
        "big_finance_harness/tools/python_exec.py": "upstream",
        "scripts/run_eval_set.py": "upstream",
        "scripts/build_plots.py": "",
        "data/big_finance_subset.jsonl": SUBSET,
        "data/held_back.jsonl": "secret",
        "grades/run.jsonl": "grades",
    }
    repo = tmp_path / "upstream"
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "pinned")
    return repo


def test_stages_the_pinned_files_and_the_overlay(
    ingest, checkout, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ingest, "UPSTREAM_COMMIT", git(checkout, "rev-parse", "HEAD").stdout.strip()
    )
    (checkout / "data/big_finance_subset.jsonl").write_text('{"id": "bf-local"}\n')
    (checkout / "big_finance_harness/local.py").write_text("untracked")
    stage = tmp_path / "stage"

    task_ids = ingest.stage_checkout(checkout, stage)

    upstream = stage / "runtime/upstream"
    staged = sorted(
        p.relative_to(upstream).as_posix() for p in upstream.rglob("*") if p.is_file()
    )
    assert staged == [
        "LICENSE",
        "README.md",
        "big_finance_harness/__init__.py",
        "big_finance_harness/tools/python_exec.py",
        "data/big_finance_subset.jsonl",
        "pyproject.toml",
        "scripts/run_eval_set.py",
    ]
    assert task_ids == ["bf-1", "bf-2"]
    assert (stage / "runtime/checkout.Dockerfile").is_file()
    assert (
        stage / "runtime/overlay/big_finance_harness/models/trajectory.py"
    ).is_file()
    assert (stage / ".dockerignore").read_text() == (ROOT / ".dockerignore").read_text()


def test_refuses_a_checkout_at_another_commit(ingest, checkout, tmp_path):
    with pytest.raises(SystemExit, match=ingest.UPSTREAM_COMMIT):
        ingest.stage_checkout(checkout, tmp_path / "stage")


def test_refuses_a_directory_that_is_not_a_checkout(ingest, tmp_path):
    with pytest.raises(SystemExit, match="not a git checkout"):
        ingest.stage_checkout(tmp_path, tmp_path / "stage")
