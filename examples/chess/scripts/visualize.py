# /// script
# requires-python = ">=3.11"
# dependencies = ["chess==1.11.2", "playwright>=1.58", "matplotlib>=3.10"]
# ///
"""Render annotated best-move boards and the observed dataset distribution."""

import argparse
import html
import json
from collections import Counter
from pathlib import Path

import chess
import chess.svg
from playwright.sync_api import sync_playwright

from dataset import BANDS, CATEGORIES, audit

ROOT = Path(__file__).resolve().parent.parent


def render_distribution(records, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    audit(records)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.7), layout="constrained")
    serialized = {}
    for axis, (name, split) in zip(
        axes, (("Training", "train"), ("Evaluation", "test"))
    ):
        rows = [row for row in records if row["split"] == split]
        counts = Counter((row["category"], row["rating_band"]) for row in rows)
        matrix = [[counts[category, band] for band in BANDS] for category in CATEGORIES]
        totals = [sum(row) for row in matrix]
        band_totals = [sum(row[column] for row in matrix) for column in range(4)]
        shares = [
            [count / total for count in row] for row, total in zip(matrix, totals)
        ]
        image = axis.imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row, values in enumerate(matrix):
            for column, count in enumerate(values):
                axis.text(
                    column,
                    row,
                    f"{count:,}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    color="white" if shares[row][column] > 0.5 else "#172b4d",
                )
        axis.set_title(f"{name}: {len(rows):,} positions", fontsize=14, pad=14)
        axis.set_yticks(
            range(4),
            [
                f"{category.title()} ({total:,})"
                for category, total in zip(CATEGORIES, totals)
            ],
        )
        axis.set_xticks(
            range(4),
            [f"{band}\n({total:,})" for band, total in zip(BANDS, band_totals)],
        )
        axis.tick_params(length=0, labelsize=9)
        axis.set_xlabel("Player Elo of side to move (total)", labelpad=10)
        serialized[name] = {
            "total": len(rows),
            "categories": list(CATEGORIES),
            "rating_bands": list(BANDS),
            "counts": matrix,
            "category_totals": totals,
            "rating_totals": band_totals,
        }
    figure.suptitle(
        "Chess reasoning: category and player-rating distribution\n"
        "Cells show counts. Categories are balanced; player ratings follow the source data.",
        fontsize=15,
    )
    colorbar = figure.colorbar(
        image, ax=axes, location="bottom", shrink=0.5, pad=0.08, aspect=45
    )
    colorbar.set_label("Share within each category")
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
    figure.savefig(output / "dataset-distribution.png", dpi=180)
    plt.close(figure)
    (output / "dataset-distribution.json").write_text(
        json.dumps(serialized, indent=2) + "\n"
    )


def card(row, title=None, after=False):
    board = chess.Board(row["fen"])
    orientation = board.turn
    move = chess.Move.from_uci(row["solution"][0])
    if after:
        board.push(move)
    svg = chess.svg.board(
        board, orientation=orientation, size=330, lastmove=move if after else None
    )
    label = title or f"{row['category'].title()} · Player Elo {row['player_rating']}"
    return f'<article id="position-{row["position_id"]}"><h2>{html.escape(label)}</h2>{svg}</article>'


def capture(page, output, name, title, cards):
    markup = """<!doctype html><html lang="en"><meta charset="utf-8">
<style>
body{margin:0;background:#fff;color:#222;font:16px Arial,sans-serif}
main{width:840px;padding:24px;box-sizing:border-box}h1{font-size:23px;margin:0 0 20px}
h2{font-size:16px;margin:0 0 12px}.boards{display:grid;grid-template-columns:1fr 1fr;gap:20px}
article{padding:12px;border:1px solid #ddd;border-radius:6px}svg{display:block;width:100%;height:auto}
</style><main>"""
    markup += (
        f'<h1>{html.escape(title)}</h1><div class="boards">'
        + "".join(cards)
        + "</div></main></html>"
    )
    path = output / f"{name}.html"
    path.write_text(markup)
    page.goto(path.resolve().as_uri())
    page.locator("main").screenshot(path=str(output / f"{name}.png"))


def render(source, output, browser_path=None):
    records = [json.loads(line) for line in source.read_text().splitlines()]
    selected = []
    for category in CATEGORIES:
        selected.extend(
            sorted(
                (
                    row
                    for row in records
                    if row["split"] == "test" and row["category"] == category
                ),
                key=lambda row: row["position_id"],
            )[:2]
        )
    output.mkdir(parents=True, exist_ok=True)
    render_distribution(records, output)
    (output / "positions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected)
    )
    with sync_playwright() as playwright:
        kwargs = {"executable_path": str(browser_path)} if browser_path else {}
        browser = playwright.chromium.launch(headless=True, **kwargs)
        page = browser.new_page(
            viewport={"width": 840, "height": 1000}, device_scale_factor=2
        )
        capture(
            page,
            output,
            "positions",
            "Eight chess reasoning positions: two per category",
            [card(row) for row in selected],
        )
        for row in selected:
            page.locator(f"#position-{row['position_id']}").screenshot(
                path=str(output / f"{row['position_id']}.png")
            )
        example = next(
            row
            for row in records
            if row["split"] == "test" and row["category"] == "tactics"
        )
        board = chess.Board(example["fen"])
        move_name = board.san(chess.Move.from_uci(example["solution"][0]))
        capture(
            page,
            output,
            "best-move",
            "Reference replay: choose the stored best move",
            [
                card(example, "Start · You have 1 move to make"),
                card(example, f"Submit {move_name} · Reward 1.0", after=True),
            ],
        )
        browser.close()
    print(
        f"Rendered the distribution, best-move example, and {len(selected)} board previews in {output}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data" / "positions.jsonl"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "visuals")
    parser.add_argument("--browser-path", type=Path)
    args = parser.parse_args()
    render(args.source, args.output, args.browser_path)


if __name__ == "__main__":
    main()
