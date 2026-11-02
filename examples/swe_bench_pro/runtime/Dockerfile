FROM python:3.12-slim

RUN apt-get update \
  && apt-get install --yes --no-install-recommends ca-certificates docker-cli docker.io \
  && rm -rf /var/lib/apt/lists/* \
  && docker --version \
  && dockerd --version
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.9.32" "jinja2>=3.1" "pyyaml>=6"

# ingest.py stages the SWE-Bench Pro V2 prompts and verifiers, without reference solutions.
COPY swe_bench_pro_v2.tar.gz /tmp/swe_bench_pro_v2.tar.gz
RUN mkdir -p /opt/swe_bench_pro/upstream \
  && tar -xzf /tmp/swe_bench_pro_v2.tar.gz -C /opt/swe_bench_pro/upstream \
  && rm /tmp/swe_bench_pro_v2.tar.gz

COPY swe_bench_pro_harness.py /opt/swe_bench_pro/swe_bench_pro_harness.py
CMD ["sleep", "infinity"]
