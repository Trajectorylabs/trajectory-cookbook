# /// script
# requires-python = ">=3.11"
# dependencies = ["chess==1.11.2"]
# ///
"""Select balanced best-move tasks from four curated Stockfish-labeled datasets."""

import argparse
import hashlib
import heapq
import json
from urllib.parse import urlparse
from collections import Counter
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("openings", "tactics", "middlegame", "endgame")
BANDS = ("below-2000", "2000-2199", "2200-2399", "2400+")


def rating_band(rating):
    if rating < 2000:
        return BANDS[0]
    if rating < 2200:
        return BANDS[1]
    if rating < 2400:
        return BANDS[2]
    return BANDS[3]


def validate_position(row):
    board = chess.Board(row["fen"])
    if not board.is_valid() or board.is_game_over():
        raise ValueError("Task must start at a valid, nonterminal position")
    if row["category"] not in CATEGORIES or row["split"] not in {"train", "test"}:
        raise ValueError("Unknown category or split")
    if len(row["solution"]) != 1:
        raise ValueError("A best-move task needs exactly one reference move")
    move = board.parse_uci(row["solution"][0])
    if row["engine"]["name"] != "Stockfish 19":
        raise ValueError("Expected frozen Stockfish 19 labels")
    passes = row["filter_passes"]
    if [check["limit_nodes"] for check in passes] != [30_000, 180_000]:
        raise ValueError("Expected both recorded search budgets")
    for check in passes:
        if len(check["lines"]) < 2 or check["gap_cp"] <= 0:
            raise ValueError("Expected a recorded advantage over an alternative")
        if check["lines"][0]["move"] != move.uci():
            raise ValueError("Best move changed between recorded searches")
        for line in check["lines"]:
            replay = board.copy()
            if not line["pv"] or line["pv"][0] != line["move"]:
                raise ValueError("PV does not start with its labeled move")
            for uci in line["pv"]:
                replay.push_uci(uci)
    recheck = row.get("eval_recheck")
    if row["split"] == "test":
        if not recheck or recheck["limit_nodes"] != 2_000_000:
            raise ValueError("Held-out position is missing its deeper recheck")
        if recheck["depth"] < 16 or recheck["gap_cp"] <= 0:
            raise ValueError("Held-out recheck did not confirm a distinct best move")
        if recheck["pv"][0] != move.uci():
            raise ValueError("Held-out recheck disagrees with the reference move")
        replay = board.copy()
        for uci in recheck["pv"]:
            replay.push_uci(uci)
    source = row["source"]
    if urlparse(source["game"]).hostname not in {
        "lichess.org",
        "chess.com",
        "www.chess.com",
    }:
        raise ValueError("Expected a traceable Lichess or Chess.com source game")
    player = "white_elo" if board.turn else "black_elo"
    if row["player_rating"] != int(source[player]):
        raise ValueError("Rating must belong to the side to move")
    if row["rating_band"] != rating_band(row["player_rating"]):
        raise ValueError("Incorrect player-rating band")
    return row


def normalize(row):
    board = chess.Board(row["fen"])
    rating = int(row["source"]["white_elo" if board.turn else "black_elo"])
    record = {
        "position_id": row["position_id"],
        "split": row["split"],
        "category": row["category"],
        "fen": row["fen"],
        "solution": row["solution_uci"],
        "player_rating": rating,
        "rating_band": rating_band(rating),
        "source": row["source"],
        "engine": row["engine"],
        "label_version": row["label_version"],
        "filter_passes": row["filter_passes"],
    }
    for key in ("eval_recheck", "tablebase_check"):
        if key in row:
            record[key] = row[key]
    return validate_position(record)


def selection_key(row):
    return hashlib.sha256(
        ("chess-reasoning-v1:" + row["position_id"]).encode()
    ).hexdigest()


def select(source_dir, train_per_category=1250, test_per_category=25):
    records, sources = [], {}
    for category in CATEGORIES:
        path = source_dir / f"{category}-v3.jsonl"
        pools = {"train": [], "test": []}
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for raw in source:
                digest.update(raw)
                row = json.loads(raw)
                if row["category"] != category:
                    raise ValueError(f"Wrong category in {path.name}")
                pools[row["split"]].append(row)
        sources[category] = {
            "file": path.name,
            "sha256": digest.hexdigest(),
            "counts": {split: len(rows) for split, rows in pools.items()},
        }
        for split, count in (
            ("train", train_per_category),
            ("test", test_per_category),
        ):
            if len(pools[split]) < count:
                raise ValueError(f"Not enough {split} positions for {category}")
            records.extend(
                normalize(row)
                for row in heapq.nsmallest(count, pools[split], key=selection_key)
            )
    audit(records, train_per_category, test_per_category)
    return records, sources


def audit(records, train_per_category=1250, test_per_category=25):
    counts, ids, boards = Counter(), set(), set()
    games = {"train": set(), "test": set()}
    for row in records:
        validate_position(row)
        # Exclude repeated boards even when their clocks or castling rights differ.
        board_key = " ".join(row["fen"].split()[:2])
        if row["position_id"] in ids or board_key in boards:
            raise ValueError("Duplicate position")
        ids.add(row["position_id"])
        boards.add(board_key)
        game = row["source"]["game"].split("#")[0].rstrip("/")
        games[row["split"]].add(game)
        counts[row["split"], row["category"]] += 1
    if games["train"] & games["test"]:
        raise ValueError("Source game overlaps training and evaluation")
    for split, count in (("train", train_per_category), ("test", test_per_category)):
        for category in CATEGORIES:
            if counts[split, category] != count:
                raise ValueError(f"Incorrect {split} count for {category}")
    return {str(key): value for key, value in sorted(counts.items())}


def evaluation_subset(records):
    selected = sorted(
        (row for row in records if row["split"] == "test"),
        key=lambda row: row["position_id"],
    )
    if Counter(row["category"] for row in selected) != {
        category: 25 for category in CATEGORIES
    }:
        raise ValueError("Expected 25 held-out positions in each of four categories")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Directory containing the four curated *-v3.jsonl source files",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "positions.jsonl"
    )
    args = parser.parse_args()
    records, sources = select(args.source_dir)
    payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload)
    manifest = {
        "task": "best-move-reasoning",
        "source": "Curated Lichess and Chess.com game positions with frozen Stockfish 19 annotations",
        "selection": "Lowest SHA256(chess-reasoning-v1: + position_id), within original category and split",
        "source_files": sources,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "count": len(records),
        "counts": audit(records),
        "rating": "Player Elo of the side to move; observed distribution, no rating quotas",
        "grading": "1.0 only for the stored best UCI move; all other submissions score 0.0",
        "reference_searches": [30_000, 180_000],
        "test_recheck_nodes": 2_000_000,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
