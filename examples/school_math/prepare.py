"""Fetch the pinned generator and create reproducible numeric school-math tasks."""

import argparse
import os
import subprocess
from pathlib import Path

COMMIT = "427f45075f84b8b9774950196ad63867ca20ffb3"
SOURCE = "https://github.com/google-deepmind/mathematics_dataset.git"
DEPENDENCIES = (
    "numpy==1.26.4",
    "sympy==1.5.1",
    "absl-py==2.1.0",
    "six==1.17.0",
    "mpmath==1.3.0",
    "setuptools==70.3.0",
)


def prepare_source(path: Path) -> None:
    if not path.exists():
        subprocess.run(["git", "clone", SOURCE, str(path)], check=True)
        subprocess.run(["git", "checkout", "--detach", COMMIT], cwd=path, check=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=path, text=True
    )
    if revision != COMMIT or dirty:
        raise ValueError(f"Expected clean source checkout at {COMMIT}: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--train-count", type=int, default=64)
    parser.add_argument("--validation-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261106)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    source = (
        args.source_dir or output.parent / f"school-math-source-{COMMIT[:12]}"
    ).resolve()
    prepare_source(source)
    command = ["uv", "run", "--no-project", "--python", "3.12"]
    for dependency in DEPENDENCIES:
        command.extend(["--with", dependency])
    command.extend(
        [
            "python",
            "-m",
            "examples.school_math.generate_dataset",
            "--source-dir",
            str(source),
            "--output-dir",
            str(output),
            "--train-count",
            str(args.train_count),
            "--validation-count",
            str(args.validation_count),
            "--test-count",
            str(args.test_count),
            "--seed",
            str(args.seed),
        ]
    )
    subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYTHONHASHSEED": "0"},
        check=True,
    )


if __name__ == "__main__":
    main()
