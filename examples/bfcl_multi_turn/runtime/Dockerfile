FROM python:3.12-slim

# CPU-only torch first, so bfcl-eval's sentence-transformers dependency does not pull CUDA wheels.
# soundfile: qwen-agent (a bfcl-eval dependency) imports it at load time without declaring it.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
  && python -m pip install --no-cache-dir "bfcl-eval==2026.3.23" "trajectory-sdk>=0.10.1" soundfile

# Fail the build if the harness cannot import (catches undeclared upstream dependencies).
COPY protocol.py bfcl_harness.py /tmp/import-check/
RUN cd /tmp/import-check && BFCL_PROJECT_ROOT=/tmp/import-check python -c "import bfcl_harness" \
  && rm -rf /tmp/import-check

# BFCL writes file locks and results under its project root.
ENV BFCL_PROJECT_ROOT=/opt/bfcl-root \
  PYTHONUNBUFFERED=1
RUN mkdir -p /opt/bfcl-root

WORKDIR /opt/bfcl
COPY protocol.py bfcl_harness.py LICENSE-BFCL ./

# Fail the build if the packaged multi-turn data differs from the 4 x 200 entries ingested.
RUN python -c "\
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry; \
from protocol import CATEGORIES, SCENARIOS; \
sizes = {c: (len(load_dataset_entry(c)), len(load_ground_truth_entry(c))) for c in CATEGORIES}; \
assert all(s == (SCENARIOS, SCENARIOS) for s in sizes.values()), sizes"

CMD ["sleep", "infinity"]
