FROM python:3.12-slim

WORKDIR /opt/writing_agent
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.10.1" "openai>=3.46,<4"
COPY writing_harness.py tasks.jsonl voice_examples.jsonl /opt/writing_agent/

CMD ["sleep", "infinity"]
