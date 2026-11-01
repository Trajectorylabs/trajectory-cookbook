FROM python:3.12-slim

ARG BFB_COMMIT=bafc769bc58032f659e4117c3d839258ce0fda40
WORKDIR /opt/big_finance
RUN python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], '/tmp/upstream.tar.gz')" \
    "https://github.com/Rogo-Technologies/big-finance-benchmark/archive/${BFB_COMMIT}.tar.gz" \
  && tar -xzf /tmp/upstream.tar.gz --strip-components=1 \
    "big-finance-benchmark-${BFB_COMMIT}/LICENSE" \
    "big-finance-benchmark-${BFB_COMMIT}/README.md" \
    "big-finance-benchmark-${BFB_COMMIT}/pyproject.toml" \
    "big-finance-benchmark-${BFB_COMMIT}/big_finance_harness" \
    "big-finance-benchmark-${BFB_COMMIT}/scripts/run_eval_set.py" \
    "big-finance-benchmark-${BFB_COMMIT}/data/big_finance_subset.jsonl" \
  && rm /tmp/upstream.tar.gz
COPY overlay/ ./
RUN python -m pip install --no-cache-dir . trajectory-sdk==0.9.32
CMD ["sleep", "infinity"]
