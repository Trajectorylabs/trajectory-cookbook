import argparse
import csv
import html
import json
import os
import re
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile


def write_private_text(path: Path, text: str) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temporary_path = handle.name
    os.replace(temporary_path, path)


def parse_timestamp(value: str) -> datetime:
    if re.match(r"^\d{4}-\d{2}-\d{2}", value):
        timestamp = datetime.fromisoformat(value)
    else:
        timestamp = datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    if timestamp.tzinfo is None:
        raise ValueError("Post timestamps must include a timezone")
    return timestamp.astimezone(UTC)


def read_archive(source: Path) -> list[dict]:
    posts = []
    with zipfile.ZipFile(source) as archive:
        names = sorted(
            name
            for name in archive.namelist()
            if re.search(r"(?:^|/)data/tweets?(?:-part\d+)?\.js$", name)
        )
        if not names:
            raise ValueError(
                "No data/tweets.js, tweet.js, or tweets-partN.js found in ZIP"
            )
        for name in names:
            content = archive.read(name).decode("utf-8-sig")
            match = re.fullmatch(
                r"\s*window\.YTD\.tweets?\.part\d+\s*=\s*(\[.*\])\s*;?\s*",
                content,
                re.DOTALL,
            )
            if match is None:
                raise ValueError(
                    f"Unsupported archive wrapper in {name}; use CSV instead"
                )
            for entry in json.loads(match[1]):
                tweet = entry["tweet"]
                posts.append(
                    {
                        "id": tweet["id_str"],
                        "created_at": tweet["created_at"],
                        "text": tweet["full_text"],
                        "is_reply": bool(tweet.get("in_reply_to_status_id_str")),
                        "is_retweet": tweet["full_text"].startswith("RT @"),
                    }
                )
    return posts


def read_csv(source: Path) -> list[dict]:
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "created_at", "text"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV requires id, created_at, text columns")
        posts = []
        for row in reader:
            post = {key: row[key] for key in ("id", "created_at", "text")}
            for flag in ("is_reply", "is_retweet"):
                value = row.get(flag, "").strip().lower()
                if value not in {"", "0", "false", "1", "true"}:
                    raise ValueError(f"CSV {flag} must be true, false, 1, 0, or blank")
                post[flag] = value in {"1", "true"}
            posts.append(post)
    return posts


def prepare_dataset(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if source.suffix.lower() == ".zip":
        posts = read_archive(source)
    elif source.suffix.lower() == ".csv":
        posts = read_csv(source)
    else:
        raise ValueError("Input must be an X archive ZIP or a CSV")

    ordered = sorted(
        posts, key=lambda post: (parse_timestamp(post["created_at"]), post["id"])
    )
    seen_ids = set()
    seen_texts = set()
    retained = []
    excluded = {"reply_or_retweet": 0, "empty": 0, "duplicate": 0}
    for post in ordered:
        text = html.unescape(post["text"]).strip()
        if post["is_reply"] or post["is_retweet"] or text.startswith("RT @"):
            excluded["reply_or_retweet"] += 1
            continue
        normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
        if not normalized:
            excluded["empty"] += 1
            continue
        if post["id"] in seen_ids or normalized in seen_texts:
            excluded["duplicate"] += 1
            continue
        seen_ids.add(post["id"])
        seen_texts.add(normalized)
        retained.append(
            {
                "id": post["id"],
                "created_at": parse_timestamp(post["created_at"]).isoformat(),
                "text": text,
            }
        )

    if len(retained) < 25:
        raise ValueError(
            "Need at least 25 usable posts; 100+ is preferable for a pilot"
        )
    count = len(retained)
    voice_count, train_end, validation_end = count // 5, count * 2 // 5, count * 3 // 5
    partitions = {
        "voice_examples": retained[-voice_count:],
        "train_seeds": retained[:train_end],
        "validation_seeds": retained[train_end:validation_end],
        "test_seeds": retained[validation_end:-voice_count],
    }
    output.mkdir(parents=True, mode=0o700)
    for name, rows in partitions.items():
        path = output / f"{name}.jsonl"
        write_private_text(
            path, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )
    report = {
        "input_posts": len(posts),
        "retained_posts": count,
        "excluded": excluded,
        "partitions": {name: len(rows) for name, rows in partitions.items()},
        "status": "seeds_only_not_a_trainable_benchmark",
        "voice_selection": "newest_20_percent_review_for_current_style",
    }
    report_path = output / "preparation_report.json"
    write_private_text(report_path, json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local writing-agent dataset seeds"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
