FROM python:3.12-slim

WORKDIR /opt/breakout
RUN python -m pip install --no-cache-dir "numpy==2.4.3" "trajectory-sdk>=0.10.1" \
    && python -m pip install --no-cache-dir --no-deps "MinAtar==1.0.15"
# MinAtar's optional plotting dependencies are unnecessary for CPU simulation.
COPY scorer.py breakout_harness.py prompt.txt tasks.jsonl source.json /opt/breakout/
COPY LICENSE-MinAtar ATTRIBUTION.txt /opt/breakout/
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
