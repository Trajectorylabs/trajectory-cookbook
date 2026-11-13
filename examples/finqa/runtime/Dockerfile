FROM python:3.12-slim
WORKDIR /opt/finqa
RUN python -m pip install --no-cache-dir trajectory-sdk==0.10.13 numpy==2.2.6 sympy==1.14.0 tqdm==4.67.1
COPY finqa_harness.py upstream_evaluate.py private_tasks.jsonl source_manifest.json SOURCE_LICENSE NOTICE /opt/finqa/
CMD ["sleep", "infinity"]
