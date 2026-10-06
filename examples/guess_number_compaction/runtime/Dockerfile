FROM python:3.12-slim

WORKDIR /opt/guess_number_compaction
RUN python -m pip install --no-cache-dir --upgrade trajectory-sdk
COPY guess_number_compaction.py /opt/guess_number_compaction/guess_number_compaction.py
COPY tasks /opt/guess_number_compaction/tasks

CMD ["sleep", "infinity"]
