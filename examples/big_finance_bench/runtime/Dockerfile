FROM python:3.12-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/big_finance_bench
COPY source .
COPY python_exec.patch run_task.py ./
# python_exec runs model code as nobody with an empty environment; only root reads the gold.
RUN patch -p1 < python_exec.patch && chmod 600 data/big_finance_subset.jsonl
RUN pip install --no-cache-dir . "trajectory-sdk>=0.10.1"

CMD ["sleep", "infinity"]
