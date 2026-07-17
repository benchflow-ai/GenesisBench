FROM debian:bookworm-slim AS restrict-builder

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        gcc libc6-dev linux-libc-dev \
    && rm -rf /var/lib/apt/lists/*

COPY security/restrict_exec.c /tmp/restrict_exec.c
RUN gcc -O2 -Wall -Wextra -Werror \
    /tmp/restrict_exec.c -o /restrict-exec

FROM python:3.12-slim

COPY --from=restrict-builder /restrict-exec /usr/local/bin/restrict-exec

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        bash ca-certificates util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "envpool==1.1.1" \
    "numpy>=1.26,<3" \
    "pyyaml>=6,<7"

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app /opt/genesisbench

COPY src/genesisbench /opt/genesisbench/genesisbench
COPY tasks/simulation_heuristics_breakout_ram_v1/evaluate.py /app/evaluate.py
COPY tasks/simulation_heuristics_breakout_ram_v1/starter_policy /app/starter_policy
COPY tasks/simulation_heuristics_breakout_ram_v1/task_context /app/task_context

RUN cp -R /app/starter_policy /app/final_policy \
    && mkdir -p /app/work \
    && chown -R root:root /app /opt/genesisbench \
    && chmod -R a-w /opt/genesisbench /app/evaluate.py \
        /app/starter_policy /app/task_context \
    && chown -R agent:agent /app/final_policy /app/work

ENV PYTHONPATH="/opt/genesisbench" \
    GENESISBENCH_POLICY_ISOLATION="required"

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
