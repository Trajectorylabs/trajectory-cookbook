"""Select checksum-verified rows from the official English DDXPlus v2 release."""

import argparse
import ast
import csv
import hashlib
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from tempfile import NamedTemporaryFile

FILES = {
    "release_evidences.json": (40278013, "0660962abe0bbf99e1fb77188c253597"),
    "release_conditions.json": (62561569, "a992eadcff3099dd72e1820fbd2115aa"),
    "release_train_patients.zip": (40278019, "f75a78420ccb061951daa4db2d62139b"),
    "release_validate_patients.zip": (40278022, "e4db88e9082f18410de6fd0cb7e74d18"),
    "release_test_patients.zip": (40278016, "6fd53d331cf19ed4439bf4f46b92ee94"),
}
COMMON_SIX = ("URTI", "GERD", "Cluster headache", "Panic attack", "Pneumonia", "Anemia")
SCHEMA_VERSION = 2
SOURCE = "https://figshare.com/articles/dataset/DDXPlus_Dataset_English_/22687585/2"


def atomic_write(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, path)


def fetch(cache: Path, name: str) -> Path:
    file_id, expected = FILES[name]
    target = cache / name
    if not target.exists():
        with NamedTemporaryFile(dir=cache, delete=False) as handle:
            temporary = handle.name
            with urllib.request.urlopen(
                f"https://ndownloader.figshare.com/files/{file_id}", timeout=120
            ) as source:
                shutil.copyfileobj(source, handle)
        os.replace(temporary, target)
    digest = hashlib.md5(target.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"Checksum mismatch for {target}: remove it and retry")
    return target


def select_rows(
    path: Path,
    split: str,
    count: int,
    offset: int,
    evidences: dict,
    conditions: dict,
    quotas: dict | None = None,
) -> list[dict]:
    rows = []
    selected_counts = dict.fromkeys(quotas or {}, 0)
    if quotas is not None and (
        sum(quotas.values()) != count or not set(quotas) <= set(conditions)
    ):
        raise ValueError(
            "Curriculum quotas must match the count and official condition catalog"
        )
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError("Expected one patient CSV in the official archive")
        with io.TextIOWrapper(archive.open(members[0]), encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            for index, raw in enumerate(reader):
                if index < offset:
                    continue
                if quotas is not None and (
                    raw["PATHOLOGY"] not in quotas
                    or selected_counts[raw["PATHOLOGY"]] >= quotas[raw["PATHOLOGY"]]
                ):
                    continue
                profile = ast.literal_eval(raw["EVIDENCES"])
                differential = ast.literal_eval(raw["DIFFERENTIAL_DIAGNOSIS"])
                if raw["PATHOLOGY"] not in conditions or any(
                    label not in conditions for label, _ in differential
                ):
                    raise ValueError("Unknown pathology in source")
                for token in profile:
                    evidence_id, separator, value = token.partition("_@_")
                    metadata = evidences[evidence_id]
                    if separator and value not in {
                        str(item) for item in metadata["possible-values"]
                    }:
                        raise ValueError(f"Unknown value: {token}")
                    if bool(separator) != (metadata["data_type"] != "B"):
                        raise ValueError(f"Wrong evidence type: {token}")
                initial = raw["INITIAL_EVIDENCE"]
                if initial not in profile:
                    raise ValueError("Initial evidence is not in the patient profile")
                rows.append(
                    {
                        "task_id": f"{split}-row-{index}",
                        "split": split,
                        "source_row_index": index,
                        "source_row_sha256": hashlib.sha256(
                            json.dumps(raw, sort_keys=True).encode()
                        ).hexdigest(),
                        "patient": {
                            "age": int(raw["AGE"]),
                            "sex": raw["SEX"],
                            "initial_evidence": initial,
                            "evidences": profile,
                            "pathology": raw["PATHOLOGY"],
                            "differential": differential,
                        },
                    }
                )
                if quotas is not None:
                    selected_counts[raw["PATHOLOGY"]] += 1
                if len(rows) == count:
                    break
    if len(rows) != count:
        raise ValueError(f"Not enough {split} rows for this selection")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/ddxplus-cache"))
    parser.add_argument("--train-count", type=int, default=24)
    parser.add_argument("--validation-count", type=int, default=12)
    parser.add_argument("--test-count", type=int, default=12)
    parser.add_argument(
        "--mode", choices=("common-six", "full49"), default="common-six"
    )
    parser.add_argument("--train-per-condition", type=int, default=4)
    parser.add_argument("--validation-per-condition", type=int, default=2)
    parser.add_argument("--test-per-condition", type=int, default=2)
    parser.add_argument("--train-offset", type=int, default=24)
    parser.add_argument("--validation-offset", type=int, default=8)
    parser.add_argument("--test-offset", type=int, default=12)
    args = parser.parse_args()
    if (
        min(args.train_count, args.validation_count, args.test_count) < 1
        or min(
            args.train_per_condition,
            args.validation_per_condition,
            args.test_per_condition,
        )
        < 1
        or min(args.train_offset, args.validation_offset, args.test_offset) < 0
    ):
        parser.error("Counts must be positive; offset must be nonnegative")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: fetch(args.cache_dir, name) for name in FILES}
    evidences = json.loads(paths["release_evidences.json"].read_text())
    conditions = json.loads(paths["release_conditions.json"].read_text())
    for name in ("release_evidences.json", "release_conditions.json"):
        atomic_write(args.output_dir / name, paths[name].read_text())
    actor_conditions = (
        list(COMMON_SIX) if args.mode == "common-six" else list(conditions)
    )
    atomic_write(
        args.output_dir / "actor_conditions.json", json.dumps(actor_conditions) + "\n"
    )
    offsets = {
        "train": args.train_offset,
        "validate": args.validation_offset,
        "test": args.test_offset,
    }
    per_condition = {
        "train": args.train_per_condition,
        "validate": args.validation_per_condition,
        "test": args.test_per_condition,
    }
    counts = {
        "train": args.train_count,
        "validate": args.validation_count,
        "test": args.test_count,
    }
    if args.mode == "common-six":
        counts = {
            split: len(actor_conditions) * quota
            for split, quota in per_condition.items()
        }
    for split, count in counts.items():
        rows = select_rows(
            paths[f"release_{split}_patients.zip"],
            split,
            count,
            offsets[split],
            evidences,
            conditions,
            dict.fromkeys(actor_conditions, per_condition[split])
            if args.mode == "common-six"
            else None,
        )
        atomic_write(
            args.output_dir / f"{split}.jsonl",
            "".join(json.dumps(row) + "\n" for row in rows),
        )
    manifest = {
        "source": SOURCE,
        "license": "CC BY 4.0",
        "schema_version": SCHEMA_VERSION,
        "mode": args.mode,
        "actor_conditions": actor_conditions,
        "selection": "first matching rows after native split offsets, with fixed per-condition quotas"
        if args.mode == "common-six"
        else "consecutive source rows; experimental full49 smoke subset",
        "offsets": offsets,
        "per_condition": per_condition if args.mode == "common-six" else None,
        "counts": counts,
        "selected_sha256": {
            name: hashlib.sha256((args.output_dir / name).read_bytes()).hexdigest()
            for name in (
                "actor_conditions.json",
                "train.jsonl",
                "validate.jsonl",
                "test.jsonl",
                "release_evidences.json",
                "release_conditions.json",
            )
        },
        "files": {
            name: {
                "figshare_file_id": FILES[name][0],
                "md5": FILES[name][1],
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in paths.items()
        },
    }
    atomic_write(
        args.output_dir / "source_manifest.json", json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "counts": counts}))


if __name__ == "__main__":
    main()
