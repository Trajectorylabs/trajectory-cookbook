import importlib.util
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ingest", EXAMPLE / "ingest.py")
ingest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ingest)


@pytest.fixture
def checkout(tmp_path):
    root = tmp_path / "harvey-labs"
    (root / "lab_core" / "harness").mkdir(parents=True)
    (root / "lab_core" / "harness" / "tools.py").write_text("upstream tools\n")
    (root / "lab_core" / "harness" / "run.py").write_text("upstream run\n")
    (root / "lab_core" / "__pycache__").mkdir()
    (root / "lab_core" / "__pycache__" / "x.pyc").write_bytes(b"")
    (root / "tasks" / "area" / "task").mkdir(parents=True)
    (root / "tasks" / "area" / "task" / "task.json").write_text("{}")
    return root


def test_stage_overlays_checkout_without_tasks(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "git_head", lambda path: ingest.SOURCE_COMMIT)

    root = ingest.stage(checkout, tmp_path / "stage")

    runtime = root / "runtime"
    overlay_run = EXAMPLE / "runtime" / "lab_core" / "harness" / "run.py"
    assert (
        runtime / "lab_core" / "harness" / "tools.py"
    ).read_text() == "upstream tools\n"
    assert (
        runtime / "lab_core" / "harness" / "run.py"
    ).read_text() == overlay_run.read_text()
    assert (runtime / "lab_core" / "harness" / "adapters" / "trajectory.py").is_file()
    assert (runtime / "Dockerfile").read_text() == (
        EXAMPLE / "runtime" / "Dockerfile"
    ).read_text()
    assert (root / ".dockerignore").is_file()
    assert not (root / "tasks").exists()
    assert not list(root.rglob("__pycache__"))


def test_stage_rejects_other_commit(checkout, tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "git_head", lambda path: "0" * 40)

    with pytest.raises(SystemExit, match=ingest.SOURCE_COMMIT):
        ingest.stage(checkout, tmp_path / "stage")


def test_dockerfile_downloads_source_commit():
    dockerfile = (EXAMPLE / "runtime" / "Dockerfile").read_text()

    assert f"harvey-labs/tar.gz/{ingest.SOURCE_COMMIT}" in dockerfile
