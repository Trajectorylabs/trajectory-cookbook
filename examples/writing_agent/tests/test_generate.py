import json
from pathlib import Path

import pytest

from examples.writing_agent import generate
from examples.writing_agent.tests.test_runtime import writing_task


def test_new_ideas_need_no_reference_and_package_no_other_private_files(
    tmp_path: Path,
) -> None:
    task = writing_task()
    public = {"task_id": task["task_id"], "input": task["input"]}
    inputs = tmp_path / "inputs.jsonl"
    voice = tmp_path / "voice.jsonl"
    inputs.write_text(json.dumps(public) + "\n")
    voice.write_text(json.dumps({"id": "voice_test", "text": "small things :)"}) + "\n")
    (tmp_path / "private_notes.txt").write_text("not for the model")
    package = tmp_path / "package"
    assert generate.package_inputs(inputs, voice, package) == [public]
    assert not (package / "private_notes.txt").exists()
    assert "private_grading" not in (package / "tasks.jsonl").read_text()
    assert (package / "writing_harness.py").exists()


def test_draft_review_rejects_private_reference_fields(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs.jsonl"
    voice = tmp_path / "voice.jsonl"
    inputs.write_text(json.dumps(writing_task()) + "\n")
    voice.write_text('{"text":"example"}\n')
    with pytest.raises(ValueError, match="no reference posts"):
        generate.package_inputs(inputs, voice, tmp_path / "package")
