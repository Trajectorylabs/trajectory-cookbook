FROM python:3.12-slim

WORKDIR /opt/big_finance
COPY upstream/ ./
COPY overlay/ ./
RUN python -m pip install --no-cache-dir . trajectory-sdk==0.9.32
CMD ["sleep", "infinity"]
