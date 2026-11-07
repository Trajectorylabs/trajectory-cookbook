FROM python:3.12-slim
WORKDIR /opt/ddxplus
RUN python -m pip install --no-cache-dir trajectory-sdk==0.10.13
COPY ddxplus_harness.py tasks.jsonl release_evidences.json release_conditions.json actor_conditions.json source_manifest.json NOTICE /opt/ddxplus/
CMD ["sleep", "infinity"]
