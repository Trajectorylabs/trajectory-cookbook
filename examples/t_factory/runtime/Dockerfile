FROM python:3.12-slim

WORKDIR /opt/t_factory
RUN python -m pip install --no-cache-dir trajectory-sdk==0.6.20
COPY t_factory_harness.py /opt/t_factory/t_factory_harness.py

CMD ["sleep", "infinity"]
