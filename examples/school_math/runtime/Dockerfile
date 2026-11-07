FROM python:3.12-slim
WORKDIR /opt/school_math
RUN pip install --no-cache-dir "trajectory-sdk==0.10.13"
COPY math_harness.py tasks.jsonl source_manifest.json LICENSE-DeepMind NOTICE ./
CMD ["sleep", "infinity"]
