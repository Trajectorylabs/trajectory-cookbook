import hashlib
import json
from pathlib import Path

import pytest
from trajectory.lib.benchmarks import _BenchmarkPackage

from examples.school_math.ingest import COMMIT, build_benchmark, package_tasks
from examples.school_math.prepare import prepare_source


def fixture_data(root: Path) -> Path:
    data = root / "data"
    data.mkdir()
    manifest = {
        "source_commit": COMMIT,
        "modules": ["algebra__linear_1d", "polynomials__evaluate"],
        "partitions": {},
    }
    for split, seed in (("train", 1), ("validation", 2), ("test", 3)):
        content = (
            json.dumps(
                {
                    "id": f"{split}_00000",
                    "split": split,
                    "generation_seed": seed,
                    "source_module": "algebra__linear_1d",
                    "source_regime": "train" if split == "train" else "interpolate",
                    "question": f"Solve {seed}*x = 1.",
                    "answer": f"1/{seed}",
                }
            )
            + "\n"
        )
        (data / f"{split}.jsonl").write_text(content)
        manifest["partitions"][split] = {
            "count": 1,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    (data / "LICENSE-DeepMind").write_text("license_fixture")
    manifest["license_sha256"] = hashlib.sha256(b"license_fixture").hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data


def test_package_preserves_splits_and_excludes_offline_validation(
    tmp_path: Path,
) -> None:
    rows = package_tasks(fixture_data(tmp_path), tmp_path / "package")
    spec = build_benchmark("school-math-test", rows)
    assert {task.split for task in spec.tasks} == {"train", "test"}
    assert {row["id"] for row in rows} == {"train_00000", "test_00000"}
    assert "validation_00000" not in (tmp_path / "package" / "tasks.jsonl").read_text()


def test_modified_data_is_rejected_before_packaging(tmp_path: Path) -> None:
    data = fixture_data(tmp_path)
    with (data / "train.jsonl").open("a") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        package_tasks(data, tmp_path / "package")


def test_sdk_upload_contains_every_dockerfile_copy_source(tmp_path: Path) -> None:
    package_dir = tmp_path / "package"
    data = fixture_data(tmp_path)
    rows = package_tasks(data, package_dir)
    package = _BenchmarkPackage(build_benchmark("school-math-test", rows), package_dir)
    uploaded = {file.path for file in package.files()}
    for line in (package_dir / "Dockerfile").read_text().splitlines():
        if line.startswith("COPY "):
            assert set(line.split()[1:-1]) <= uploaded
    assert "source_manifest.json" in uploaded
    assert (package_dir / "source_manifest.json").read_bytes() == (
        data / "manifest.json"
    ).read_bytes()


def test_cross_split_duplicate_is_rejected(tmp_path: Path) -> None:
    data = fixture_data(tmp_path)
    row = json.loads((data / "test.jsonl").read_text())
    row["question"] = "Solve 1*x = 1."
    payload = json.dumps(row) + "\n"
    (data / "test.jsonl").write_text(payload)
    manifest = json.loads((data / "manifest.json").read_text())
    manifest["partitions"]["test"]["sha256"] = hashlib.sha256(
        payload.encode()
    ).hexdigest()
    (data / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        package_tasks(data, tmp_path / "package")


def test_dirty_source_checkout_is_rejected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "subprocess.check_output",
        lambda command, **kwargs: COMMIT if command[1] == "rev-parse" else " M file.py",
    )
    with pytest.raises(ValueError, match="clean source"):
        prepare_source(tmp_path)
