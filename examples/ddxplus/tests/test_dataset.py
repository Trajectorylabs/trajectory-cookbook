import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from examples.ddxplus.ingest import build_benchmark, package_tasks
from examples.ddxplus.prepare_dataset import SOURCE, select_rows


def prepared_data(root: Path):
    root.mkdir()
    names = []
    for number, split in enumerate(("train", "validate", "test")):
        row = {
            "task_id": f"{split}-row-0",
            "source_row_index": 0,
            "split": split,
            "patient": {
                "age": 20 + number,
                "sex": "F",
                "initial_evidence": "symptom",
                "evidences": ["symptom"],
                "pathology": "condition_a",
                "differential": [["condition_a", 1]],
            },
        }
        name = f"{split}.jsonl"
        (root / name).write_text(json.dumps(row) + "\n")
        names.append(name)
    for name in ("release_evidences.json", "release_conditions.json"):
        (root / name).write_text("{}")
        names.append(name)
    manifest = {
        "source": SOURCE,
        "counts": {"train": 1, "validate": 1, "test": 1},
        "selected_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        },
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_only_native_train_and_test_are_packaged(tmp_path):
    data = tmp_path / "data"
    prepared_data(data)
    package = tmp_path / "package"
    rows = package_tasks(data, package)
    assert [row["split"] for row in rows] == ["train", "test"]
    assert not (package / "validate.jsonl").exists()
    assert "validate-row-0" not in (package / "tasks.jsonl").read_text()
    benchmark = build_benchmark(rows, "test-ddxplus")
    assert [task.split for task in benchmark.tasks] == ["train", "test"]


def test_modified_patient_data_is_rejected_before_packaging(tmp_path):
    data = tmp_path / "data"
    prepared_data(data)
    (data / "train.jsonl").write_text("{}")
    with pytest.raises(ValueError, match="Prepared data changed"):
        package_tasks(data, tmp_path / "package")
    assert not (tmp_path / "package").exists()


def test_identical_clinical_profiles_across_splits_are_rejected(tmp_path):
    data = tmp_path / "data"
    manifest = prepared_data(data)
    row = json.loads((data / "test.jsonl").read_text())
    row["patient"]["age"] = 20
    row["patient"]["differential"] = [["condition_a", 0.8]]
    (data / "test.jsonl").write_text(json.dumps(row) + "\n")
    manifest["selected_sha256"]["test.jsonl"] = hashlib.sha256(
        (data / "test.jsonl").read_bytes()
    ).hexdigest()
    (data / "source_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate patient profile"):
        package_tasks(data, tmp_path / "package")


def test_csv_selection_preserves_source_position_and_encoded_values(tmp_path):
    source = io.StringIO()
    writer = csv.DictWriter(
        source,
        fieldnames=[
            "AGE",
            "SEX",
            "PATHOLOGY",
            "EVIDENCES",
            "INITIAL_EVIDENCE",
            "DIFFERENTIAL_DIAGNOSIS",
        ],
    )
    writer.writeheader()
    for age in (20, 21):
        writer.writerow(
            {
                "AGE": age,
                "SEX": "F",
                "PATHOLOGY": "condition_a",
                "EVIDENCES": "['symptom', 'color_@_red']",
                "INITIAL_EVIDENCE": "symptom",
                "DIFFERENTIAL_DIAGNOSIS": "[['condition_a', 1.0]]",
            }
        )
    archive = tmp_path / "patients.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("release_train_patients", source.getvalue())
    metadata = {
        "symptom": {"data_type": "B"},
        "color": {"data_type": "C", "possible-values": ["red"]},
    }
    rows = select_rows(archive, "train", 1, 1, metadata, {"condition_a": {}})
    assert rows[0]["task_id"] == "train-row-1"
    assert rows[0]["patient"]["age"] == 21
    assert rows[0]["patient"]["evidences"] == ["symptom", "color_@_red"]
    assert len(rows[0]["source_row_sha256"]) == 64
