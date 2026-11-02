FROM python:3.12-slim

WORKDIR /opt/chess
RUN python -m pip install --no-cache-dir "trajectory-sdk==0.9.37" "chess==1.11.2"
COPY chess_harness.py python_tool.py /opt/chess/

CMD ["sleep", "infinity"]
