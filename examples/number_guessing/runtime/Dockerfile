FROM python:3.12-slim

WORKDIR /opt/number_guessing
RUN python -m pip install --no-cache-dir --upgrade trajectory-sdk
COPY number_guessing_harness.py /opt/number_guessing/number_guessing_harness.py
COPY tasks /opt/number_guessing/tasks

CMD ["sleep", "infinity"]
