FROM python:3.12-slim

RUN apt-get update \
  && apt-get install --yes --no-install-recommends ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

# FinQA data and its execution-accuracy evaluator, pinned and digest-checked.
RUN git clone --filter=blob:none https://github.com/czyssrs/FinQA /opt/finqa-src \
  && git -C /opt/finqa-src checkout 0f16e2867befa6840783e58be38c9efb9229d742 \
  && cd /opt/finqa-src \
  && echo "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6  dataset/train.json" | sha256sum -c \
  && echo "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc  dataset/test.json" | sha256sum -c \
  && echo "845cd131cab843eceff256cf6d392978cc470a7da4a80107beb56027fdca5c13  code/evaluate/evaluate.py" | sha256sum -c

# evaluate.py imports numpy, sympy and tqdm at module level.
RUN python -m pip install --no-cache-dir "trajectory-sdk>=0.10.1" numpy sympy tqdm

WORKDIR /opt/finqa
COPY finqa_harness.py LICENSE-FINQA ./
ENV PYTHONUNBUFFERED=1
CMD ["sleep", "infinity"]
