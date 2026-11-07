import json
from pathlib import Path

import pytest

from examples.writing_agent.ingest import build_benchmark, package_tasks
from examples.writing_agent.tests.test_runtime import writing_task


def prepared_data(root: Path) -> tuple[Path, Path]:
    data = root / "data"
    data.mkdir()
    (data / "voice_examples.jsonl").write_text(
        json.dumps({"id": "voice_001", "text": "My voice."}) + "\n"
    )
    for split in ("train", "test", "validation"):
        (data / f"{split}_seeds.jsonl").write_text(
            json.dumps({"id": f"post_{split}", "text": "Private seed."}) + "\n"
        )
    (data / "private_archive.zip").write_text("Unrelated raw archive")
    tasks = []
    for split in ("train", "test"):
        task = writing_task()
        task.update(task_id=f"{split}_001", split=split)
        task["private_grading"]["source_post_ids"] = [f"post_{split}"]
        tasks.append(task)
    path = root / "reviewed.jsonl"
    path.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    return data, path


def test_package_contains_only_selected_runtime_and_reviewed_data(
    tmp_path: Path,
) -> None:
    data, reviewed = prepared_data(tmp_path)
    package = tmp_path / "package"
    tasks = package_tasks(data, reviewed, package)
    assert {path.name for path in package.iterdir()} == {
        "Dockerfile",
        "writing_harness.py",
        "tasks.jsonl",
        "voice_examples.jsonl",
    }
    manifest = build_benchmark(
        "writing-test", tasks, "fixed-judge", "voice_hash_fixture"
    )
    assert {task.split for task in manifest.tasks} == {"train", "test"}
    assert all(
        task.env_vars["OPENAI_API_KEY"].secret_ref == "OPENAI_API_KEY"
        for task in manifest.tasks
    )
    assert all(
        "--judge-model fixed-judge" in task.run_command for task in manifest.tasks
    )
    assert "OPENAI_API_KEY" not in (package / "tasks.jsonl").read_text()


def test_openrouter_judge_uses_its_own_runtime_secret(tmp_path: Path) -> None:
    data, reviewed = prepared_data(tmp_path)
    tasks = package_tasks(data, reviewed, tmp_path / "package")
    manifest = build_benchmark(
        "writing-test", tasks, "openai/fixed-judge", "voice_hash_fixture", "openrouter"
    )
    assert all(
        task.env_vars["OPENAI_API_KEY"].secret_ref == "OPENROUTER_API_KEY"
        for task in manifest.tasks
    )
    assert all(
        "--judge-provider openrouter" in task.run_command for task in manifest.tasks
    )


@pytest.mark.parametrize(
    "mistake", ["wrong_split", "unreviewed", "duplicate", "unsafe_id", "private_input"]
)
def test_invalid_tasks_are_rejected_before_creating_package(
    tmp_path: Path, mistake: str
) -> None:
    data, reviewed = prepared_data(tmp_path)
    tasks = [json.loads(line) for line in reviewed.read_text().splitlines()]
    if mistake == "wrong_split":
        tasks[0]["private_grading"]["source_post_ids"] = ["post_test"]
    elif mistake == "unreviewed":
        tasks[0]["private_grading"]["author_reviewed"] = False
    elif mistake == "duplicate":
        tasks[1]["task_id"] = tasks[0]["task_id"]
    elif mistake == "unsafe_id":
        tasks[0]["task_id"] = "$(unexpected)"
    elif mistake == "private_input":
        tasks[0]["input"]["private_grading"] = tasks[0]["private_grading"]
    reviewed.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    with pytest.raises(ValueError):
        package_tasks(data, reviewed, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_agent_prepared_tasks_require_explicit_mode_and_keep_review_provenance(
    tmp_path: Path,
) -> None:
    data, reviewed = prepared_data(tmp_path)
    tasks = [json.loads(line) for line in reviewed.read_text().splitlines()]
    for task in tasks:
        task["private_grading"]["author_reviewed"] = False
        task["private_grading"]["brief_origin"] = "synthetic_agent_prepared"
    reviewed.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    with pytest.raises(ValueError, match="not been author reviewed"):
        package_tasks(data, reviewed, tmp_path / "blocked")
    packaged = package_tasks(
        data, reviewed, tmp_path / "smoke", allow_agent_prepared=True
    )
    assert all(task["private_grading"]["author_reviewed"] is False for task in packaged)
    assert all(
        task["private_grading"]["brief_origin"] == "synthetic_agent_prepared"
        for task in packaged
    )


def test_agent_prepared_mode_cannot_bypass_review_of_other_task_origins(
    tmp_path: Path,
) -> None:
    data, reviewed = prepared_data(tmp_path)
    tasks = [json.loads(line) for line in reviewed.read_text().splitlines()]
    tasks[0]["private_grading"]["author_reviewed"] = False
    reviewed.write_text("".join(json.dumps(task) + "\n" for task in tasks))
    with pytest.raises(ValueError, match="not been author reviewed"):
        package_tasks(data, reviewed, tmp_path / "blocked", allow_agent_prepared=True)
