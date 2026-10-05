FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/officebench-venv/bin:${PATH}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN mkdir -p /usr/share/man/man1 && apt-get update && \
    apt-get install -y --no-install-recommends \
      bzip2 \
      bsdmainutils \
      ca-certificates \
      curl \
      gcc \
      imagemagick \
      libtesseract-dev \
      libreoffice \
      libreoffice-java-common \
      default-jre-headless \
      poppler-utils \
      proot \
      tesseract-ocr \
      tree \
    && rm -rf /var/lib/apt/lists/*

COPY environment/officebench-local/tts_no_network.c /tmp/tts_no_network.c
RUN gcc -O2 -Wall -Wextra -o /usr/local/bin/tts-no-network /tmp/tts_no_network.c && \
    rm -f /tmp/tts_no_network.c

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

RUN micromamba create -y -p /opt/officebench-venv -c conda-forge python=3.12 pip && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir \
      fire \
      pandas \
      python-docx \
      openpyxl \
      tabulate \
      Pillow \
      PyMuPDF \
      PyPDF2 \
      pdf2docx \
      pytesseract \
      icalendar \
      pytz \
      "openai>=1.30.0" \
      "azure-identity>=1.15.0" \
      "pyyaml>=6.0" \
      "tqdm>=4.66.0" && \
    micromamba clean -afy

RUN python -c "import azure.identity, fitz, icalendar, openai, openpyxl, pandas, pytesseract, yaml" && \
    /usr/local/bin/tts-no-network python -c "import socket; socket.socket(socket.AF_UNIX)" && \
    libreoffice --version && \
    tesseract --version >/dev/null
