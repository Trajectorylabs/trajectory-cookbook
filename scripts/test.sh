#!/usr/bin/env bash
# Apply the same pinned upstream + local overlay used by the Harvey runtime image.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SOURCE_COMMIT="1dd81403b2fbb60596f7aea3fcecafad7bf73143"
SOURCE_DIR="$ROOT/.test-sources/harvey"
if [[ ! -d "$SOURCE_DIR/.git" ]]; then
  git init -q "$SOURCE_DIR"
  git -C "$SOURCE_DIR" remote add origin https://github.com/harveyai/harvey-labs.git
  git -C "$SOURCE_DIR" sparse-checkout set --no-cone /lab_core/ /pyproject.toml /README.md /LICENSE
fi
if [[ "$(git -C "$SOURCE_DIR" rev-parse HEAD 2>/dev/null)" != "$SOURCE_COMMIT" ]]; then
  git -C "$SOURCE_DIR" -c protocol.version=2 fetch --depth=1 --filter=blob:none origin "$SOURCE_COMMIT"
  git -C "$SOURCE_DIR" checkout -q --detach "$SOURCE_COMMIT"
fi
uv sync --locked --link-mode copy --reinstall-package lab-core
uv run --no-sync python - <<'PY'
import importlib.util
import shutil
from pathlib import Path

spec = importlib.util.find_spec("lab_core")
installed = Path(next(iter(spec.submodule_search_locations)))
shutil.copytree("examples/harvey_labs/runtime/lab_core", installed, dirs_exist_ok=True)
PY
exec uv run --no-sync pytest -q examples/harvey_labs/tests "$@"
