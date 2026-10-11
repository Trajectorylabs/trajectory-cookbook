FROM ghcr.io/astral-sh/uv:0.8.15 AS uv
FROM python:3.12-slim

COPY --from=uv /uv /uvx /bin/
RUN apt-get update \
  && apt-get install --yes --no-install-recommends ca-certificates git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/tau3
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

RUN git clone --filter=blob:none --no-checkout https://github.com/sierra-research/tau2-bench /opt/tau2-source \
  && git -C /opt/tau2-source sparse-checkout init --cone \
  && git -C /opt/tau2-source sparse-checkout set \
    data/tau2/domains/banking_knowledge data/tau2/user_simulator \
  && git -C /opt/tau2-source checkout fc0055dc4e0a316c3f83133267fbd6faaa770992 \
  && test "$(git -C /opt/tau2-source rev-parse HEAD)" = "fc0055dc4e0a316c3f83133267fbd6faaa770992"

ENV PATH="/opt/tau3/.venv/bin:$PATH" \
  PYTHONUNBUFFERED=1 \
  TAU2_DATA_DIR=/opt/tau2-source/data
COPY tau3_banking /opt/tau3/tau3_banking
CMD ["sleep", "infinity"]
