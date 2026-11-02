# /// script
# requires-python = ">=3.11"
# dependencies = ["chess==1.11.2", "playwright>=1.58"]
# ///
"""Serialize ten real puzzles and capture one contact sheet plus individual PNGs."""

import argparse
import html
import json
from pathlib import Path

import chess
import chess.svg
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).parent
BANDS = ("0-999", "1000-1499", "1500-1999", "2000-3000")


def render(source, output, browser_path=None):
    records = [json.loads(line) for line in source.read_text().splitlines()]
    selected = []
    for mate in range(1, 6):
        for band in (BANDS[(mate - 1) % 4], BANDS[(mate + 1) % 4]):
            selected.append(min((r for r in records if r["split"] == "test"
                                 and r["mate_in"] == mate and r["rating_band"] == band),
                                key=lambda r: r["puzzle_id"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "puzzles.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in selected))
    cards = []
    for row in selected:
        board = chess.Board(row["fen"])
        svg = chess.svg.board(board, orientation=board.turn, size=330)
        label = f"Mate in {row['mate_in']} · Rating {row['rating']} · {row['puzzle_id']}"
        side = "White" if board.turn else "Black"
        cards.append(f'<article id="puzzle-{row["puzzle_id"]}"><h2>{html.escape(label)}</h2>'
                     f'<p>{side} to move</p>{svg}</article>')
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
        page = browser.new_page(viewport={"width": 780, "height": 1000}, device_scale_factor=2)
        page.goto(page_path.resolve().as_uri())
        page.locator("main").screenshot(path=str(output / "puzzles.png"))
        for row in selected:
            page.locator(f"#puzzle-{row['puzzle_id']}").screenshot(path=str(output / f"{row['puzzle_id']}.png"))
        browser.close()
    print(f"Serialized {len(selected)} puzzles and captured {len(selected) + 1} images in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "puzzles.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "visuals")
    parser.add_argument("--browser-path", type=Path)
    args = parser.parse_args()
    render(args.source, args.output, args.browser_path)


if __name__ == "__main__":
    main()
