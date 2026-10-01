#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Reuse the complete adapter from TRA-1444's consolidation PR.
revision=9fb55cf671385b2f21de0e62e095d108f3478d46
mkdir -p .work
git clone https://github.com/Trajectorylabs/bigcodebench.git .work/source
git -C .work/source checkout --detach "$revision"
git -C .work/source apply ../../mainline-sdk.patch
cd .work/source
uv run --project sdk --frozen --python 3.12 python -B -m pytest -q sdk/tests
uv run --project sdk --frozen --only-group prepare --python 3.12 python -B -m sdk.prepare \
  --name bigcodebench-full-original-tra1444 --output ../prepared \
  --max-output-tokens-per-step 32768
