"""Download checksum-pinned FinQA and select native train/public-test tasks."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from urllib.request import urlopen

from examples.finqa.runtime.finqa_harness import load_evaluator

ROOT = Path(__file__).parent
SOURCE_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
SOURCE_URL = f"https://raw.githubusercontent.com/czyssrs/FinQA/{SOURCE_COMMIT}"
SOURCES = {
    "dataset/train.json": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dataset/dev.json": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "dataset/test.json": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
    "code/evaluate/evaluate.py": "845cd131cab843eceff256cf6d392978cc470a7da4a80107beb56027fdca5c13",
    "LICENSE": "92c0e67bd762c15c07ba60de09558338315a8ebf99aac4f51f487179809ac087",
}


def checked_bytes(path: Path, expected: str) -> bytes:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Source checksum mismatch: {path.name}")
    return content


def fetch_sources(cache_dir: Path) -> None:
    for name, expected in SOURCES.items():
        path = cache_dir / name
        if not path.exists():
            with urlopen(f"{SOURCE_URL}/{name}", timeout=120) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError(f"Downloaded source checksum mismatch: {name}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        checked_bytes(path, expected)


def select_rows(rows: list[dict], count: int, split: str) -> list[dict]:
    if not 1 <= count <= len(rows):
        raise ValueError(f"{split} count must be in [1, {len(rows)}]")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate source IDs in {split}")
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"finqa-starter-v1:{split}:{row['id']}".encode()
        ).digest(),
    )
    return [
        {
            "task_id": hashlib.sha256(f"{split}:{row['id']}".encode()).hexdigest()[:20],
            "split": split,
            "source_id": row["id"],
            "source": row,
        }
        for row in ordered[:count]
    ]


def prepare(
    output_dir: Path, cache_dir: Path, train_count: int = 64, test_count: int = 16
) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {output_dir}")
    fetch_sources(cache_dir)
    rows = {
        split: json.loads(
            checked_bytes(
                cache_dir / f"dataset/{split}.json", SOURCES[f"dataset/{split}.json"]
            )
        )
        for split in ("train", "dev", "test")
    }
    selected = select_rows(rows["train"], train_count, "train") + select_rows(
        rows["test"], test_count, "test"
    )
    evaluator = load_evaluator(cache_dir / "code/evaluate/evaluate.py")
    for record in selected:
        row = record["source"]
        invalid, result = evaluator.eval_program(
            evaluator.program_tokenization(row["qa"]["program"]), row["table"]
        )
        if invalid or result != row["qa"]["exe_ans"]:
            raise ValueError(f"Source gold execution mismatch: {row['id']}")
    manifest = {
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCES,
        "selection": "sha256(finqa-starter-v1:source_split:source_id), smallest first",
        "source_counts": {split: len(items) for split, items in rows.items()},
        "selected": [
            {key: record[key] for key in ("task_id", "split", "source_id")}
            for record in selected
        ],
        "offline_dev": "dev.json is preserved locally and is excluded from the managed package",
        "metric": "official eval_program valid result == qa.exe_ans (five-decimal upstream rounding)",
        "private_tasks_sha256": hashlib.sha256(
            "".join(json.dumps(record) + "\n" for record in selected).encode()
        ).hexdigest(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    (output_dir / "private_tasks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in selected)
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copyfile(cache_dir / "dataset/dev.json", output_dir / "dev.json")
    shutil.copyfile(
        cache_dir / "code/evaluate/evaluate.py", output_dir / "upstream_evaluate.py"
    )
    shutil.copyfile(cache_dir / "LICENSE", output_dir / "SOURCE_LICENSE")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".source")
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=16)
    args = parser.parse_args()
    manifest = prepare(
        args.output_dir, args.cache_dir, args.train_count, args.test_count
    )
    print(
        f"data_dir={args.output_dir} tasks={len(manifest['selected'])} commit={SOURCE_COMMIT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
