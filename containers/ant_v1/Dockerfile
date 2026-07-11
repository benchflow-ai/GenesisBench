FROM ghcr.io/openhands/agent-server:latest-python

USER root

RUN /usr/local/venv/bin/python -m pip install --no-cache-dir \
    "openhands-sdk==1.34.0" \
    "openhands-tools==1.34.0" \
    "gymnasium[mujoco]>=1.2.2,<2" \
    "numpy>=1.26,<3"

RUN npm install --global --force @agentclientprotocol/claude-agent-acp@0.58.1

RUN rm -f /usr/local/bin/python /usr/local/bin/pip \
    && printf '#!/bin/sh\nexec /usr/local/venv/bin/python "$@"\n' \
      > /usr/local/bin/python \
    && printf '#!/bin/sh\nexec /usr/local/venv/bin/pip "$@"\n' \
      > /usr/local/bin/pip \
    && chmod +x /usr/local/bin/python /usr/local/bin/pip

COPY scripts/run_openhands_agent.py /opt/genesisbench/run_openhands_agent.py

ENV PATH="/usr/local/venv/bin:${PATH}"

WORKDIR /workspace

ENTRYPOINT ["/usr/local/venv/bin/python", "/opt/genesisbench/run_openhands_agent.py"]
