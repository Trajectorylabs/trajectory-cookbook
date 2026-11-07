import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from examples.ddxplus.ingest import build_benchmark, package_tasks
from examples.ddxplus.prepare_dataset import COMMON_SIX, SOURCE, select_rows


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
        (root / name).write_text(
            json.dumps({"condition_a": {}})
            if name == "release_conditions.json"
            else "{}"
        )
        names.append(name)
    (root / "actor_conditions.json").write_text(json.dumps(["condition_a"]))
    names.append("actor_conditions.json")
    manifest = {
        "schema_version": 2,
        "mode": "full49",
        "actor_conditions": ["condition_a"],
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


def test_balanced_selection_fills_each_native_condition_quota_after_offset(tmp_path):
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
    for age, label in enumerate(
        [
            "condition_b",
            "condition_a",
            "condition_a",
            "condition_a",
            "condition_b",
            "condition_b",
        ]
    ):
        writer.writerow(
            {
                "AGE": 20 + age,
                "SEX": "F",
                "PATHOLOGY": label,
                "EVIDENCES": "['symptom']",
                "INITIAL_EVIDENCE": "symptom",
                "DIFFERENTIAL_DIAGNOSIS": repr([[label, 1.0]]),
            }
        )
    archive = tmp_path / "patients.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("release_patients", source.getvalue())
    for split in ("train", "validate", "test"):
        rows = select_rows(
            archive,
            split,
            4,
            1,
            {"symptom": {"data_type": "B"}},
            {"condition_a": {}, "condition_b": {}},
            {"condition_a": 2, "condition_b": 2},
        )
        assert [row["source_row_index"] for row in rows] == [1, 2, 4, 5]
        assert all(row["split"] == split for row in rows)
        assert all(row["task_id"].startswith(split + "-row-") for row in rows)


def test_old_schema_requires_explicit_repreparation(tmp_path):
    data = tmp_path / "data"
    manifest = prepared_data(data)
    manifest.pop("schema_version")
    (data / "source_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requires re-preparing"):
        package_tasks(data, tmp_path / "package")


def test_incomplete_curriculum_coverage_is_rejected_before_packaging(tmp_path):
    data = tmp_path / "data"
    manifest = prepared_data(data)
    manifest.update(
        mode="common-six",
        actor_conditions=["condition_a", "condition_b"],
        per_condition={"train": 1, "validate": 1, "test": 1},
    )
    (data / "actor_conditions.json").write_text(
        json.dumps(manifest["actor_conditions"])
    )
    (data / "release_conditions.json").write_text(
        json.dumps({"condition_a": {}, "condition_b": {}})
    )
    for name in ("actor_conditions.json", "release_conditions.json"):
        manifest["selected_sha256"][name] = hashlib.sha256(
            (data / name).read_bytes()
        ).hexdigest()
    (data / "source_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="cover every actor condition"):
        package_tasks(data, tmp_path / "package")


def test_six_condition_curriculum_has_fixed_coverage_in_every_native_split(tmp_path):
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
    for number in range(3):
        for label in COMMON_SIX:
            writer.writerow(
                {
                    "AGE": 20 + number,
                    "SEX": "F",
                    "PATHOLOGY": label,
                    "EVIDENCES": "['symptom']",
                    "INITIAL_EVIDENCE": "symptom",
                    "DIFFERENTIAL_DIAGNOSIS": repr([[label, 1.0]]),
                }
            )
    archive = tmp_path / "patients.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("release_patients", source.getvalue())
    for split in ("train", "validate", "test"):
        rows = select_rows(
            archive,
            split,
            12,
            0,
            {"symptom": {"data_type": "B"}},
            {key: {} for key in COMMON_SIX},
            dict.fromkeys(COMMON_SIX, 2),
        )
        assert {
            label: sum(row["patient"]["pathology"] == label for row in rows)
            for label in COMMON_SIX
        } == dict.fromkeys(COMMON_SIX, 2)
        assert all(row["split"] == split for row in rows)
