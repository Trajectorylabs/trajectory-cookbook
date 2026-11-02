FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir harbor==0.25.0 trajectory-sdk==0.9.37

COPY tasks /opt/terminal_bench_4/tasks
COPY LICENSE /opt/terminal_bench_4/LICENSE
COPY run_task.py /opt/terminal_bench_4/run_task.py

CMD ["sleep", "infinity"]
