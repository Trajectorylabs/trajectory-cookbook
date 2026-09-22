FROM python:3.12-slim

WORKDIR /opt/trajectory_word
RUN python -m pip install --no-cache-dir trajectory-sdk
COPY trajectory_word_harness.py /opt/trajectory_word/trajectory_word_harness.py

CMD ["sleep", "infinity"]
