FROM python:3.12-slim

WORKDIR /opt/weight_program_bench
RUN python -m pip install --no-cache-dir "numpy==2.3.4" "trajectory-sdk==0.10.1" \
    && python -m pip install --no-cache-dir --no-deps "MinAtar==1.0.15"
COPY . /opt/weight_program_bench/
ENV PYTHONPATH=/opt/weight_program_bench OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1

CMD ["sleep", "infinity"]
