FROM python:3.12-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io docker-cli docker-compose docker-buildx \
    && rm -rf /var/lib/apt/lists/*
RUN docker compose version && docker buildx version
RUN pip install --no-cache-dir harbor==0.25.0 "trajectory-sdk>=0.10.13"

COPY tasks.tar.gz /opt/extractbench/tasks.tar.gz
RUN tar -xzf /opt/extractbench/tasks.tar.gz -C /opt/extractbench \
    && rm /opt/extractbench/tasks.tar.gz
COPY LICENSE /opt/extractbench/LICENSE
COPY run_task.py /opt/extractbench/run_task.py

CMD ["sleep", "infinity"]
