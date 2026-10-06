FROM python:3.12-slim

WORKDIR /opt/guess_number
RUN python -m pip install --no-cache-dir --upgrade trajectory-sdk
COPY guess_number.py /opt/guess_number/guess_number.py
COPY tasks /opt/guess_number/tasks

CMD ["sleep", "infinity"]
