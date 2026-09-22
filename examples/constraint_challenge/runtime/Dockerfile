FROM python:3.12-slim

WORKDIR /opt/constraint_challenge
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.6.20"
COPY constraint_harness.py /opt/constraint_challenge/constraint_harness.py

CMD ["sleep", "infinity"]
