FROM python:3.12-slim

ENV LAB_ROOT=/opt/harvey
WORKDIR /opt/harvey
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl docker-cli docker.io pandoc podman uidmap \
    && ln -s /usr/sbin/dockerd /usr/bin/dockerd \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL https://codeload.github.com/harveyai/harvey-labs/tar.gz/1dd81403b2fbb60596f7aea3fcecafad7bf73143 \
    | tar -xz --strip-components=1
COPY lab_core/ lab_core/
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.9.32" .
CMD ["sleep", "infinity"]
