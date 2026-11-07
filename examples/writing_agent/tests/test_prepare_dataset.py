import csv
import json
import zipfile
from pathlib import Path

import pytest

from examples.writing_agent.prepare_dataset import prepare_dataset


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["id", "created_at", "text", "is_reply"]
        )
        writer.writeheader()
        writer.writerows(rows)


def post_rows() -> list[dict]:
    return [
        {
            "id": f"post_{index:02d}",
            "created_at": f"2026-01-{index + 1:02d}T12:00:00Z",
            "text": f"Original idea {index}",
            "is_reply": "false",
        }
        for index in range(25)
    ]


def test_csv_excludes_replies_reposts_and_duplicates_before_chronological_split(
    tmp_path: Path,
) -> None:
    source = tmp_path / "posts.csv"
    rows = post_rows()
    rows[0]["text"] = "Cats &amp; dogs"
    rows[1]["created_at"] = "2026-01-01T23:00:00-12:00"
    extra = [
        {
            **rows[0],
            "id": "duplicate_text",
            "text": " CATS & dogs ",
            "created_at": "2026-01-26T12:00:00Z",
        },
        {**rows[2], "id": "reply", "is_reply": "true"},
        {**rows[2], "id": "repost", "text": "RT @example: Original idea"},
        {**rows[2], "id": "empty", "text": "  "},
    ]
    write_csv(source, list(reversed(rows + extra)))
    output = tmp_path / "prepared"
    report = prepare_dataset(source, output)
    assert report["excluded"] == {"reply_or_retweet": 2, "empty": 1, "duplicate": 1}
    assert report["partitions"] == {
        "voice_examples": 5,
        "train_seeds": 10,
        "validation_seeds": 5,
        "test_seeds": 5,
    }
    voice = [
        json.loads(line)
        for line in (output / "voice_examples.jsonl").read_text().splitlines()
    ]
    test = [
        json.loads(line)
        for line in (output / "test_seeds.jsonl").read_text().splitlines()
    ]
    assert [post["id"] for post in voice] == [
        f"post_{index}" for index in range(20, 25)
    ]
    assert [post["id"] for post in test] == [f"post_{index}" for index in range(15, 20)]
    train = [
        json.loads(line)
        for line in (output / "train_seeds.jsonl").read_text().splitlines()
    ]
    assert train[0]["text"] == "Cats & dogs"
    assert not {post["id"] for post in voice} & {post["id"] for post in test + train}
    assert "Original idea" not in (output / "preparation_report.json").read_text()


def test_multipart_archive_reads_only_post_data_without_executing_javascript(
    tmp_path: Path,
) -> None:
    source = tmp_path / "archive.zip"
    posts = [
        {
            "tweet": {
                "id_str": row["id"],
                "created_at": row["created_at"],
                "full_text": row["text"],
            }
        }
        for row in post_rows()
    ]
    posts[0]["tweet"]["created_at"] = "Thu Jan 01 12:00:00 +0000 2026"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "archive/data/tweets.js",
            "window.YTD.tweets.part0 = " + json.dumps(posts[:10]),
        )
        archive.writestr(
            "archive/data/tweets-part1.js",
            "window.YTD.tweets.part1 = " + json.dumps(posts[10:]) + ";",
        )
        archive.writestr(
            "archive/data/direct-messages.js", "not valid JSON; must never be parsed"
        )
        archive.writestr("../../outside.txt", "must not be extracted")
    output = tmp_path / "prepared"
    assert prepare_dataset(source, output)["retained_posts"] == 25
    assert sorted(path.name for path in output.iterdir()) == [
        "preparation_report.json",
        "test_seeds.jsonl",
        "train_seeds.jsonl",
        "validation_seeds.jsonl",
        "voice_examples.jsonl",
    ]


def test_small_dataset_leaves_no_misleading_output(tmp_path: Path) -> None:
    source = tmp_path / "posts.csv"
    write_csv(source, post_rows()[:24])
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="at least 25"):
        prepare_dataset(source, output)
    assert not output.exists()


def test_existing_dataset_is_never_overwritten(tmp_path: Path) -> None:
    with pytest.raises(FileExistsError):
        prepare_dataset(tmp_path / "posts.csv", tmp_path)


def test_archive_rejects_executable_suffix(tmp_path: Path) -> None:
    source = tmp_path / "archive.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr(
            "data/tweets.js", "window.YTD.tweets.part0 = []; alert('unexpected');"
        )
    with pytest.raises(ValueError, match="Unsupported archive wrapper"):
        prepare_dataset(source, tmp_path / "prepared")
