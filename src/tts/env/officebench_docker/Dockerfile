# OfficeBench app/evaluation image for the dw read-grounded TTS harness.
#
# This is a self-contained, modernised port of OfficeBench's docker/Dockerfile
# (zlwang-cs/OfficeBench).  Differences from upstream, all to make the build reproducible today and
# to fit how the dw harness uses it:
#   * pinned base `ubuntu:22.04` (upstream `ubuntu:latest` now resolves to a PEP-668 "externally
#     managed" Python that rejects `pip install`; jammy is not externally-managed, so a plain
#     `pip install` works);
#   * no `COPY openai_key.txt` and no `git init`/initial-commit — the dw harness scores with the
#     official config-based `utils/evaluate.py` checkers (run by officebench_driver.py), not
#     intercode's git-diff reward, so the repo bootstrap is unnecessary.
#
# The task `apps/`, `utils/` and per-task `testbed/` are copied in at RUNTIME (per rollout), exactly
# as the official runner does — nothing task-specific is baked into the image.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies the OfficeBench apps shell out to (libreoffice: excel/word -> pdf;
# tesseract: ocr; imagemagick/poppler: pdf <-> image; git/tree/coreutils: shell app).
RUN apt-get update && \
    apt-get install -y psmisc bsdmainutils && \
    apt-get install -y bash sudo python3 python3-pip && \
    apt-get install -y imagemagick poppler-utils git tree && \
    apt-get install -y net-tools coreutils && \
    apt-get install -y curl jq vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y libreoffice && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /requirements.txt && \
    rm /requirements.txt

WORKDIR /
