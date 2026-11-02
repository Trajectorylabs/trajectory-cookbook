# /// script
# requires-python = ">=3.11"
# dependencies = ["chess==1.11.2", "zstandard>=0.25.0"]
# ///
"""Select and validate balanced Lichess mate-in-1 through mate-in-5 puzzles."""

import argparse
import csv
import hashlib
import io
import json
import urllib.request
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import chess
import zstandard

SOURCE = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
BANDS = ("0-999", "1000-1499", "1500-1999", "2000-3000")


def rating_band(rating):
    if not 0 <= rating <= 3000:
        return None
    if rating < 1000:
        return BANDS[0]
    if rating < 1500:
        return BANDS[1]
    if rating < 2000:
        return BANDS[2]
    return BANDS[3]


def game_id(url):
    path = url.split("lichess.org/", 1)[1].removeprefix("game/")
    return path.split("/", 1)[0].split("#", 1)[0][:8]


def validate_puzzle(row):
    """Return a normalized puzzle only when its whole source line is valid."""
    band = rating_band(int(row["Rating"]))
    moves = row["Moves"].split()
    mate = len(moves) // 2
    if band is None or len(moves) != 2 * mate or not 1 <= mate <= 5:
        return None
    themes = row["Themes"].split()
    # Lichess mateIn5 includes mates longer than five. Check the exact length above.
    if "mate" not in themes or f"mateIn{mate}" not in themes:
        return None
    board = chess.Board(row["FEN"])
    if not board.is_valid():
        return None
    try:
        board.push_uci(moves[0])  # Source FEN precedes the opponent's setup move.
        fen = board.fen()
        for index, uci in enumerate(moves[1:]):
            move = board.parse_uci(uci)
            if index % 2 == 0 and not board.gives_check(move):
                return None
            board.push(move)
        if not board.is_checkmate():
            return None
    except ValueError:
        return None
    return {
        "puzzle_id": row["PuzzleId"],
        "fen": fen,
        "source_fen": row["FEN"],
        "source_moves": moves,
        "solution": moves[1:],
        "mate_in": mate,
        "rating": int(row["Rating"]),
        "rating_band": band,
        "themes": themes,
        "game_url": row["GameUrl"],
    }


@contextmanager
def source_rows(source):
    remote = source.startswith("https://")
    with (
        urllib.request.urlopen(source, timeout=120) if remote else open(source, "rb")
    ) as raw:
        if source.endswith(".zst"):
            with zstandard.ZstdDecompressor().stream_reader(raw) as decoded:
                with io.TextIOWrapper(decoded) as text:
                    yield csv.DictReader(text)
        else:
            with io.TextIOWrapper(raw) as text:
                yield csv.DictReader(text)


def allocate(available, per_category):
    """Fill equal category and rating margins, keeping cells even where possible."""
    from collections import deque

    # Nodes 1–5 are mate lengths, and 6–9 are rating bands. Route each category's
    # quota through these bands; reverse edges let a scarce cell borrow capacity.
    source, sink = 0, 10
    capacity = {}
    flow = Counter()
    for mate in range(1, 6):
        capacity[source, mate] = per_category
        for band_index, band in enumerate(BANDS):
            capacity[mate, 6 + band_index] = available.get((mate, band), 0)
    for band_index in range(4):
        capacity[6 + band_index, sink] = 5 * per_category // 4
    # Start close to equal in every cell; augmenting paths handle scarce cells.
    for mate in range(1, 6):
        for band_index in range(4):
            band_node = 6 + band_index
            n = min(per_category // 4, capacity[mate, band_node])
            for edge in [(source, mate), (mate, band_node), (band_node, sink)]:
                flow[edge] += n
                flow[edge[::-1]] -= n
    total = sum(flow[source, mate] for mate in range(1, 6))
    while total < 5 * per_category:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            node = queue.popleft()
            # Rotate traversal so redistributed cells remain as even as possible.
            for other in [(i + total) % 11 for i in range(11)]:
                if (
                    other not in parents
                    and capacity.get((node, other), 0) > flow[node, other]
                ):
                    parents[other] = node
                    queue.append(other)
        if sink not in parents:
            raise ValueError(
                f"Not enough distinct source puzzles to balance {per_category} per mate category"
            )
        node = sink
        while parents[node] is not None:
            previous = parents[node]
            flow[previous, node] += 1
            flow[node, previous] -= 1
            node = previous
        total += 1
    return {
        (mate, band): flow[mate, 6 + index]
        for mate in range(1, 6)
        for index, band in enumerate(BANDS)
    }


def select(rows, train_per_category=1000, test_per_category=100):
    pools, positions, games, ids = {}, set(), set(), set()
    quotas = {"train": train_per_category, "test": test_per_category}
    for scanned, row in enumerate(rows, 1):
        if scanned % 500_000 == 0:
            print(
                json.dumps(
                    {"scanned": scanned, "candidates": sum(map(len, pools.values()))}
                ),
                flush=True,
            )
        band = rating_band(int(row["Rating"]))
        mate = len(row["Moves"].split()) // 2
        if (
            band is None
            or not 1 <= mate <= 5
            or f"mateIn{mate}" not in row["Themes"].split()
        ):
            continue
        game = game_id(row["GameUrl"])
        split = (
            "test"
            if int(hashlib.sha256(game.encode()).hexdigest(), 16) % 11 == 0
            else "train"
        )
        key = (split, mate, band)
        pool = pools.setdefault(key, [])
        if len(pool) >= quotas[split] or game in games or row["PuzzleId"] in ids:
            continue
        record = validate_puzzle(row)
        if record is None:
            continue
        position = " ".join(record["fen"].split()[:4])
        if position in positions:
            continue
        record["split"] = split
        pool.append(record)
        positions.add(position)
        games.add(game)
        ids.add(record["puzzle_id"])
    records = []
    for split, quota in quotas.items():
        available = {
            (mate, band): len(pools.get((split, mate, band), []))
            for mate in range(1, 6)
            for band in BANDS
        }
        allocation = allocate(available, quota)
        selected = {
            (mate, band): pools.get((split, mate, band), [])[:n]
            for (mate, band), n in allocation.items()
        }
        # Interleave cells, so small evaluations cover all available categories.
        for index in range(max(allocation.values())):
            for cell in selected.values():
                if index < len(cell):
                    records.append(cell[index])
    return records


def audit(records, train_per_category=1000, test_per_category=100):
    counts, positions, games, ids = Counter(), set(), set(), set()
    for record in records:
        row = {
            "PuzzleId": record["puzzle_id"],
            "FEN": record["source_fen"],
            "Moves": " ".join(record["source_moves"]),
            "Rating": record["rating"],
            "Themes": " ".join(record["themes"]),
            "GameUrl": record["game_url"],
        }
        normalized = validate_puzzle(row)
        if normalized is None or any(record[k] != v for k, v in normalized.items()):
            raise ValueError(f"Invalid puzzle: {record['puzzle_id']}")
        position = " ".join(record["fen"].split()[:4])
        game = game_id(record["game_url"])
        if position in positions or game in games or record["puzzle_id"] in ids:
            raise ValueError("Duplicate puzzle, source game, or position")
        positions.add(position)
        games.add(game)
        ids.add(record["puzzle_id"])
        counts[(record["split"], record["mate_in"], record["rating_band"])] += 1
    expected_splits = {"train": train_per_category, "test": test_per_category}
    if any(s not in expected_splits for s, _, _ in counts):
        raise ValueError("Invalid split")
    for split, n in expected_splits.items():
        for mate in range(1, 6):
            if sum(counts[split, mate, band] for band in BANDS) != n:
                raise ValueError(f"Incorrect category count: {split}, mate in {mate}")
        for band in BANDS:
            if sum(counts[split, mate, band] for mate in range(1, 6)) != 5 * n // 4:
                raise ValueError(f"Incorrect rating-band count: {split}, {band}")
    return {str(k): v for k, v in sorted(counts.items())}


def evaluation_subset(records):
    """Select the same 100 held-out puzzles for every model and harness."""
    pools = {}
    for mate in range(1, 6):
        for band in BANDS:
            matching = [
                puzzle
                for puzzle in records
                if puzzle["split"] == "test"
                and puzzle["mate_in"] == mate
                and puzzle["rating_band"] == band
            ]
            pools[mate, band] = sorted(matching, key=lambda puzzle: puzzle["puzzle_id"])
    counts = allocate({key: len(rows) for key, rows in pools.items()}, per_category=20)
    selected = [row for key, rows in pools.items() for row in rows[: counts[key]]]
    return sorted(
        selected, key=lambda r: (r["mate_in"], r["rating_band"], r["puzzle_id"])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "puzzles.jsonl"
    )
    args = parser.parse_args()
    with source_rows(args.source) as rows:
        records = select(rows)
    counts = audit(records)
    payload = "".join(json.dumps(r, sort_keys=True) + "\n" for r in records)
    args.output.write_text(payload)
    manifest = {
        "source": args.source,
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "count": len(records),
        "counts": counts,
        "forcing": "every solver move gives check",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
