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
    && apt-get install -y -qq --no-install-recommends util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app /app/work /opt/genesisbench

COPY src/genesisbench /opt/genesisbench/genesisbench
COPY tasks/{{task_name}}/evaluate.py /app/evaluate.py
COPY tasks/{{task_name}}/starter_artifact /app/starter_artifact
COPY tasks/{{task_name}}/task_context /app/task_context

RUN cp -R /app/starter_artifact /app/final_artifact \
    && chown -R root:root /app /opt/genesisbench \
    && chmod -R a-w /opt/genesisbench /app/evaluate.py \
        /app/starter_artifact /app/task_context \
    && chown -R agent:agent /app/final_artifact /app/work

ENV PYTHONPATH="/opt/genesisbench" \
    GENESISBENCH_POLICY_ISOLATION="required"

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
