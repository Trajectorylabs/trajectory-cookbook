FROM python:3.12-slim

WORKDIR /opt/gsm8k
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.10.1"
COPY gsm8k_harness.py /opt/gsm8k/gsm8k_harness.py
COPY tasks.jsonl /opt/gsm8k/tasks.jsonl
COPY LICENSE-GSM8K /opt/gsm8k/LICENSE-GSM8K

CMD ["sleep", "infinity"]
