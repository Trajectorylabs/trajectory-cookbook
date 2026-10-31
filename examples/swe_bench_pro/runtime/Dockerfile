FROM python:3.12-slim

RUN apt-get update \
  && apt-get install --yes --no-install-recommends ca-certificates docker-cli docker.io git \
  && rm -rf /var/lib/apt/lists/* \
  && docker --version \
  && dockerd --version
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.9.32" "jinja2>=3.1" "pyyaml>=6"

# Task instructions and verifiers from SWE-Bench Pro V2 (v2.0.0); the reference solutions are removed.
RUN git clone --filter=blob:none --no-checkout https://github.com/scaleapi/SWE-bench_Pro-os /opt/swe_bench_pro/upstream \
  && git -C /opt/swe_bench_pro/upstream sparse-checkout set v2 \
  && git -C /opt/swe_bench_pro/upstream checkout 66f92766bba642462d4bbe5479e83f91f9211862 \
  && cd /opt/swe_bench_pro/upstream/v2 \
  && sha256sum --check --quiet SHA256SUMS \
  && rm -rf tasks/*/solution

COPY swe_bench_pro_harness.py /opt/swe_bench_pro/swe_bench_pro_harness.py
CMD ["sleep", "infinity"]
