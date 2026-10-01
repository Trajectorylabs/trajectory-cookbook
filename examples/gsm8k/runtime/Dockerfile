FROM python:3.12-slim

WORKDIR /opt/gsm8k
RUN python -m pip install --no-cache-dir --upgrade 'trajectory-sdk>=0.8.10'
COPY gsm8k_harness.py /opt/gsm8k/gsm8k_harness.py
COPY tasks /opt/gsm8k/tasks

CMD ["sleep", "infinity"]
