# /// script
# requires-python = ">=3.11"
# dependencies = ["trajectory-sdk==0.9.37", "matplotlib>=3.10"]
# ///
"""Evaluate ten models with both chess harnesses: twenty jobs, 400 rollouts each."""

import argparse
import json
from pathlib import Path

from trajectory import Client

from train import (
    EVAL_OPTIONS,
    ROOT,
    evaluate,
    evaluation_tasks,
    model_options,
    result_name,
)

OPEN_MODELS = (
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3.6-35B-A3B",
    "Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.5-397B-A17B",
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3.5-Super-120B-A12B-BF16",
)
CLOSED_MODELS = (
    "openai/gpt-6.1-sol",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "google/gemini-3.1-pro-preview",
)
MODELS = OPEN_MODELS + CLOSED_MODELS


def run_matrix(client, benchmarks, results_dir):
    # Check all twenty jobs before starting the first one.
    tasks = {
        variant: evaluation_tasks(client, benchmark_id)
        for variant, benchmark_id in benchmarks.items()
    }
    for variant, rows in tasks.items():
        if any(row["variant"] != variant for row in rows.values()):
            raise ValueError(f"Wrong harness supplied for {variant}")
    if {p["puzzle_id"] for p in tasks["reasoning"].values()} != {
        p["puzzle_id"] for p in tasks["python"].values()
    }:
        raise ValueError("Both harnesses must use the same 100 evaluation puzzles")
    for model in MODELS:
        for benchmark_id in benchmarks.values():
            model_options(client, benchmark_id, model, **EVAL_OPTIONS)
    for model in MODELS:
        for variant, benchmark_id in benchmarks.items():
            path = results_dir / f"{result_name(model, variant)}-baseline.json"
            evaluate(client, benchmark_id, model, path)


def plot_results(results_dir, output_dir):
    """Plot recorded data only. No client or model calls are made here."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    states = [
        json.loads(p.read_text()) for p in sorted(results_dir.glob("*-baseline.json"))
    ]
    states = [s for s in states if "report" in s]
    reports = {(s["model"], s["variant"]): s["report"] for s in states}
    if set(reports) != {(m, v) for m in MODELS for v in ("reasoning", "python")}:
        raise ValueError(
            "Plotting the baseline matrix requires all twenty completed evaluations"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [
        m.split("/")[-1].replace("NVIDIA-Nemotron-", "Nemotron-").removesuffix("-BF16")
        for m in MODELS
    ]
    colors = {"reasoning": "#31688e", "python": "#35b779"}

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)

    def bars(ax, metric, title, percentage=False):
        y = np.arange(len(MODELS))
        for offset, variant in ((-0.18, "reasoning"), (0.18, "python")):
            values = [metric(reports[m, variant]) for m in MODELS]
            ax.barh(
                y + offset, values, height=0.34, label=variant, color=colors[variant]
            )
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Score (%)" if percentage else "Correct moves per attempt")
        ax.legend()

    fig, axes = plt.subplots(1, 4, figsize=(24, 7), sharey=True)
    for k, ax in enumerate(axes, 1):
        bars(
            ax,
            lambda r, k=k: 100 * r["pass_at_k"][str(k)],
            f"Exact continuation: pass@{k}",
            True,
        )
        ax.set_xlim(0, 100)
    finish(fig, "eval-models.png")
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    bars(
        axes[0],
        lambda r: r["mean_correct_half_moves"],
        "Correct half-move prefix (includes automatic replies)",
    )
    bars(
        axes[1],
        lambda r: r["mean_correct_solver_moves"],
        "Correct solver moves (model submissions only)",
    )
    finish(fig, "eval-prefix.png")

    for field, filename in (
        ("rating_band", "eval-rating.png"),
        ("mate_in", "eval-mate-length.png"),
    ):
        categories = sorted(next(iter(reports.values()))["breakdowns"][field])
        fig, axes = plt.subplots(2, 3, figsize=(19, 11))
        for row, variant in enumerate(("reasoning", "python")):
            for col, (title, extract, vmax) in enumerate(
                (
                    ("pass@1 (%)", lambda r: 100 * r["pass_at_k"]["1"], 100),
                    ("pass@4 (%)", lambda r: 100 * r["pass_at_k"]["4"], 100),
                    (
                        "Mean correct half-move prefix",
                        lambda r: r["mean_correct_half_moves"],
                        9,
                    ),
                )
            ):
                values = [
                    [
                        extract(reports[m, variant]["breakdowns"][field][c])
                        for c in categories
                    ]
                    for m in MODELS
                ]
                ax = axes[row, col]
                im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
                ax.set_yticks(range(len(MODELS)), labels)
                ax.set_xticks(range(len(categories)), categories)
                ax.set_title(f"{variant}: {title}")
                ax.set_xlabel(
                    "Puzzle rating band" if field == "rating_band" else "Mate in N"
                )
                fig.colorbar(im, ax=ax)
        finish(fig, filename)

    trained = [
        json.loads(p.read_text()) for p in sorted(results_dir.glob("*-training.json"))
    ]
    trained = [s for s in trained if "after" in s]
    if trained:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        y = np.arange(len(trained))
        names = [
            s["model"].split("/")[-1] + " / " + s["before"]["variant"] for s in trained
        ]
        for ax, (title, extract) in zip(
            axes,
            (
                ("pass@1 (%)", lambda r: 100 * r["pass_at_k"]["1"]),
                ("pass@4 (%)", lambda r: 100 * r["pass_at_k"]["4"]),
                (
                    "Mean correct half-move prefix",
                    lambda r: r["mean_correct_half_moves"],
                ),
            ),
        ):
            for offset, phase, color in (
                (-0.18, "before", "#31688e"),
                (0.18, "after", "#35b779"),
            ):
                ax.barh(
                    y + offset,
                    [extract(s[phase]["report"]) for s in trained],
                    0.34,
                    label=phase,
                    color=color,
                )
            ax.set_yticks(y, names)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.legend()
        finish(fig, "before-after.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reasoning-bench-id")
    parser.add_argument("--python-bench-id")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--visuals-dir", type=Path, default=ROOT / "visuals")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        if not args.reasoning_bench_id or not args.python_bench_id:
            parser.error("Both benchmark IDs are required")
        client = Client()
        print(client.organizations.retrieve_current(), flush=True)
        run_matrix(
            client,
            {"reasoning": args.reasoning_bench_id, "python": args.python_bench_id},
            args.results_dir,
        )
    plot_results(args.results_dir, args.visuals_dir)


if __name__ == "__main__":
    main()
