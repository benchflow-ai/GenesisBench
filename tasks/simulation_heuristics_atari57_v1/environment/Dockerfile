FROM python:3.11-slim

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        bash \
        ca-certificates \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "envpool==1.1.1" \
    "gymnasium[mujoco]>=1.2.2,<2" \
    "numpy>=1.26,<3" \
    "pyyaml>=6,<7"

RUN useradd --create-home --uid 1000 agent \
    && mkdir -p /app /opt/genesisbench

COPY src/genesisbench /opt/genesisbench/genesisbench
COPY tasks/simulation_heuristics_atari57_v1/evaluate.py /app/evaluate.py
COPY tasks/simulation_heuristics_atari57_v1/starter_artifact /app/starter_artifact
COPY tasks/simulation_heuristics_atari57_v1/task_context /app/task_context

RUN cp -R /app/starter_artifact /app/final_artifact \
    && chown -R root:root /opt/genesisbench \
    && chmod -R a-w /opt/genesisbench \
    && chown -R agent:agent /app

ENV PYTHONPATH="/opt/genesisbench"

WORKDIR /app

RUN mkdir -p /logs/verifier /logs/agent /logs/artifacts
