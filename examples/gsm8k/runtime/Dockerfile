FROM python:3.12-slim

WORKDIR /opt/gsm8k
RUN python -m pip install --no-cache-dir trajectory-sdk==0.6.25
COPY gsm8k_harness.py /opt/gsm8k/gsm8k_harness.py

CMD ["sleep", "infinity"]
