# /// script
# requires-python = ">=3.11"
# dependencies = ["chess==1.11.2", "playwright>=1.58", "matplotlib>=3.10", "zstandard>=0.25.0"]
# ///
"""Render puzzle previews, a mate-in-two walkthrough, and dataset counts."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import chess
import chess.svg
from playwright.sync_api import sync_playwright

from dataset import BANDS, audit, evaluation_subset

ROOT = Path(__file__).resolve().parent.parent


def render_distribution(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    audit(records)
    splits = {
        "Training": [row for row in records if row["split"] == "train"],
        "Held-out pool": [row for row in records if row["split"] == "test"],
        "Evaluation subset": evaluation_subset(records),
    }
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.7), layout="constrained")
    serialized = {}
    for axis, (name, puzzles) in zip(axes, splits.items()):
        counts = Counter((row["mate_in"], row["rating_band"]) for row in puzzles)
        matrix = [[counts[mate, band] for band in BANDS] for mate in range(1, 6)]
        row_totals = [sum(row) for row in matrix]
        column_totals = [sum(row[column] for row in matrix) for column in range(4)]
        shares = [[count / total for count in row] for row, total in zip(matrix, row_totals)]
        image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=0.5, aspect="auto")
        for row, values in enumerate(matrix):
            for column, count in enumerate(values):
                axis.text(
                    column, row, f"{count:,}", ha="center", va="center", fontsize=13,
                    color="white" if shares[row][column] >= 0.3 else "#172b4d",
                )
        axis.set_title(f"{name}\n{len(puzzles):,} puzzles", fontsize=14, pad=14)
        axis.set_yticks(range(5), [f"Mate {mate} ({total:,})" for mate, total in enumerate(row_totals, 1)])
        axis.set_xticks(range(4), [f"{band}\n({total:,})" for band, total in zip(BANDS, column_totals)])
        axis.tick_params(axis="both", length=0, labelsize=9)
        axis.set_xlabel("Puzzle rating band (total)", labelpad=10)
        for spine in axis.spines.values():
            spine.set_visible(False)
        serialized[name] = {
            "total": len(puzzles), "rating_bands": list(BANDS),
            "mate_lengths": list(range(1, 6)), "counts": matrix,
            "mate_totals": row_totals, "rating_totals": column_totals,
        }
    figure.suptitle(
        "Lichess mating tactics: dataset distribution\n"
        "Cell labels are puzzle counts. Totals appear in parentheses. Both agents use the same evaluation subset.",
        fontsize=15,
    )
    colorbar = figure.colorbar(image, ax=axes, location="bottom", shrink=0.5, pad=0.08, aspect=45)
    colorbar.set_label("Share of puzzles within each mate category")
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
    figure.savefig(output / "dataset-distribution.png", dpi=180)
    plt.close(figure)
    (output / "dataset-distribution.json").write_text(json.dumps(serialized, indent=2) + "\n")


def render_walkthrough(puzzle, output, page):
    board = chess.Board(puzzle["fen"])
    orientation = board.turn
    panels = []

    def add_panel(title, description, last_move=None):
        svg = chess.svg.board(
            board, orientation=orientation, size=330, lastmove=last_move,
            check=board.king(board.turn) if board.is_check() else None,
        )
        panels.append(
            f"<article><h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(description)}</p>{svg}</article>"
        )

    add_panel("Start · White to move", "You have 2 moves to make with submit_move.")
    first_move, opponent_move, final_move = puzzle["solution"]
    first_san = board.san(chess.Move.from_uci(first_move))
    board.push_uci(first_move)
    opponent_san = board.san(chess.Move.from_uci(opponent_move))
    board.push_uci(opponent_move)
    add_panel(
        f"Move 1 · {first_san} {opponent_san}",
        "Harness plays Black's reply. You have 1 move left.",
        chess.Move.from_uci(opponent_move),
    )
    final_san = board.san(chess.Move.from_uci(final_move))
    board.push_uci(final_move)
    assert board.is_checkmate(), "The walkthrough must end in checkmate"
    add_panel(
        f"Move 2 · {final_san} · Reward 1.0",
        "Full reference continuation matched. Checkmate.",
        chess.Move.from_uci(final_move),
    )
    markup = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Mate in two: the full continuation</title>
<style>
body{margin:0;background:#fff;color:#222;font:15px Arial,sans-serif}
main{width:1140px;padding:24px;box-sizing:border-box}
h1{font-size:24px;margin:0 0 8px}.intro{margin:0 0 20px;color:#555}
.boards{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}
article{padding:12px;border:1px solid #ddd;border-radius:6px}
h2{font-size:18px;margin:0 0 8px}article p{font-size:13px;min-height:32px;margin:0 0 10px}
article:last-child h2{color:#176534}svg{display:block;width:100%;height:auto}
</style><main><h1>Mate in two: submit your moves, the harness plays Black</h1>
"""
    markup += (
        f'<p class="intro">Lichess {html.escape(puzzle["puzzle_id"])} · '
        f'Rating {puzzle["rating"]} · Reference replay</p>'
        '<div class="boards">' + "".join(panels) + "</div></main></html>"
    )
    path = output / "mate-in-2.html"
    path.write_text(markup)
    page.set_viewport_size({"width": 1140, "height": 600})
    page.goto(path.resolve().as_uri())
    page.locator("main").screenshot(path=str(output / "mate-in-2.png"))


def render(source, output, browser_path=None):
    records = [json.loads(line) for line in source.read_text().splitlines()]
    selected = []
    for mate in range(1, 6):
        for band in (BANDS[(mate - 1) % 4], BANDS[(mate + 1) % 4]):
            selected.append(
                min(
                    (
                        r
                        for r in records
                        if r["split"] == "test"
                        and r["mate_in"] == mate
                        and r["rating_band"] == band
                    ),
                    key=lambda r: r["puzzle_id"],
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    render_distribution(records, output)
    (output / "puzzles.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected)
    )
    cards = []
    for row in selected:
        board = chess.Board(row["fen"])
        svg = chess.svg.board(board, orientation=board.turn, size=330)
        label = (
            f"Mate in {row['mate_in']} · Rating {row['rating']} · {row['puzzle_id']}"
        )
        side = "White" if board.turn else "Black"
        cards.append(
            f'<article id="puzzle-{row["puzzle_id"]}"><h2>{html.escape(label)}</h2>'
            f"<p>{side} to move</p>{svg}</article>"
        )
    markup = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Ten Lichess mating puzzles</title>
<style>
body{margin:0;background:#fff;color:#222;font:16px Arial,sans-serif}
main{width:760px;padding:20px;box-sizing:border-box}
h1{font-size:24px;margin:0 0 18px}h2{font-size:16px;margin:0 0 6px}
p{font-size:14px;margin:0 0 8px}.puzzles{display:grid;grid-template-columns:1fr 1fr;gap:20px}
article{padding:10px;border:1px solid #ddd;box-sizing:border-box}
article svg{display:block;width:100%;height:auto}
</style><main><h1>Lichess: ten exact mating continuations</h1><div class="puzzles">"""
    markup += "".join(cards) + "</div></main></html>"
    page_path = output / "puzzles.html"
    page_path.write_text(markup)
    with sync_playwright() as playwright:
        kwargs = {"executable_path": str(browser_path)} if browser_path else {}
        browser = playwright.chromium.launch(headless=True, **kwargs)
        page = browser.new_page(
            viewport={"width": 780, "height": 1000}, device_scale_factor=2
        )
        page.goto(page_path.resolve().as_uri())
        page.locator("main").screenshot(path=str(output / "puzzles.png"))
        for row in selected:
            page.locator(f"#puzzle-{row['puzzle_id']}").screenshot(
                path=str(output / f"{row['puzzle_id']}.png")
            )
        render_walkthrough(
            next(row for row in records if row["puzzle_id"] == "00KYU"), output, page
        )
        browser.close()
    print(
        f"Serialized {len(selected)} puzzles and generated {len(selected) + 3} images in {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "puzzles.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "visuals")
    parser.add_argument("--browser-path", type=Path)
    args = parser.parse_args()
    render(args.source, args.output, args.browser_path)


if __name__ == "__main__":
    main()
