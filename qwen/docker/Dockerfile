# syntax=docker/dockerfile:1
#
# Transcritor API — Qwen: Qwen3-ASR + Qwen3-ForcedAligner sobre o audio.cpp, na
# GPU. Duas variantes, o mesmo servidor e a mesma API:
#
#   vulkan  AMD e Intel (Linux, com /dev/dri repassado). Sem placa, roda na CPU.
#   cuda    NVIDIA (Linux ou Windows com Docker Desktop, com --gpus all).
#
#   docker build -t transcritor-api:qwen-vulkan .
#   docker build --build-arg VARIANT=cuda -t transcritor-api:qwen-cuda .
#
# Motor e modelos entram na imagem no build (~3,7 GB, SHA-256 conferido, espelho
# no Google Drive se o GitHub ou o HuggingFace falharem). O container roda
# offline e sem token; nenhum download precisa de login.

ARG VARIANT=vulkan
ARG UBUNTU=ubuntu:24.04
ARG CUDA_BASE=nvidia/cuda:12.8.1-runtime-ubuntu24.04

# ------------------------------------------------------------------ fetch ----
FROM ${UBUNTU} AS fetch
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*
COPY scripts/fetch.sh /usr/local/bin/fetch.sh
RUN sed -i 's/\r$//' /usr/local/bin/fetch.sh && chmod +x /usr/local/bin/fetch.sh

FROM fetch AS models
ARG HF=https://huggingface.co/audio-cpp/audio.cpp-gguf/resolve/main
RUN mkdir /models \
 && fetch.sh /models/qwen3-asr-1.7b-q8_0.gguf \
      da4fc2ac7f24dee784d1684eb1f35836cdbf559519452ae11777670734c0a4f8 \
      "$HF/Qwen3-ASR-1.7B-GGUF/qwen3-asr-1.7b-q8_0.gguf" \
      1BF6NgHuCLrrSkn8xlSbaPe6WfECoYhTb \
 && fetch.sh /models/qwen3-forced-aligner-0.6b-q8_0.gguf \
      75209490b11cec2b0db749ca5f4ff92266f58efd30f7fd04d9eb2a3ac9cc929f \
      "$HF/Qwen3-ForcedAligner-0.6B-GGUF/qwen3-forced-aligner-0.6b-q8_0.gguf" \
      1jC-4mX6-TLZHUh8dL7rWLyGrJivL28kL

# "portable": uma biblioteca de CPU por geração de processador, escolhida ao
# rodar. A versão comum é compilada para o processador de quem a gerou e pode
# não abrir em outro. O pacote vem sem permissão de execução para "outros",
# e o container roda sem ser root: daí o chmod.
FROM fetch AS engine-vulkan
ARG REL=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.1
RUN fetch.sh /tmp/a.tgz \
      63f778ef4c863ece0bca85b97c78e9629bd561724b1aa5330a9d47806608bbfa \
      "$REL/audio-v0.8.1-bin-ubuntu-x64-vulkan-portable.tar.gz" \
 && mkdir /opt/audiocpp && tar xzf /tmp/a.tgz -C /opt/audiocpp --no-same-owner \
 && rm -rf /tmp/a.tgz /opt/audiocpp/tools \
 && chmod -R a+rX /opt/audiocpp && chmod a+x /opt/audiocpp/audiocpp_*

FROM fetch AS engine-cuda
ARG REL=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.1
RUN fetch.sh /tmp/a.tgz \
      f969811783f206b6d1f6566c020211ab9df7b6bb96c6eded6ad7a58deb725025 \
      "$REL/audio-v0.8.1-bin-ubuntu-x64-cuda12.8-colab.tar.gz" \
 && mkdir /opt/audiocpp && tar xzf /tmp/a.tgz -C /opt/audiocpp --no-same-owner \
 && rm -rf /tmp/a.tgz /opt/audiocpp/tools \
 && chmod -R a+rX /opt/audiocpp && chmod a+x /opt/audiocpp/audiocpp_*

# Só na imagem cuda: o servidor CUDA depende da libcuda.so.1, que vem do driver
# da máquina — sem placa NVIDIA ele nem abre, nem no modo CPU. Esta versão só-CPU
# é a queda dele (na vulkan a própria versão portable já traz a CPU).
FROM fetch AS engine-cpu
ARG REL=https://github.com/0xShug0/audio.cpp/releases/download/v0.8.1
RUN fetch.sh /tmp/a.tgz \
      90e8d538338cc209875a18c940529302805563e54738489da1d684c6e0de12d0 \
      "$REL/audio-v0.8.1-bin-ubuntu-x64-cpu-portable.tar.gz" \
 && mkdir /opt/audiocpp-cpu && tar xzf /tmp/a.tgz -C /opt/audiocpp-cpu --no-same-owner \
 && rm -rf /tmp/a.tgz /opt/audiocpp-cpu/tools /opt/audiocpp-cpu/model_specs \
 && chmod -R a+rX /opt/audiocpp-cpu && chmod a+x /opt/audiocpp-cpu/audiocpp_*

# -------------------------------------------------------------- python ----
# Mesmo Ubuntu 24.04 (Python 3.12) das duas bases finais: o venv vai pronto.
FROM ${UBUNTU} AS venv
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r /tmp/requirements.txt

# --------------------------------------------------------- base por placa ----
FROM ${UBUNTU} AS runtime-vulkan
# libvulkan1 = o carregador; mesa-vulkan-drivers = RADV (AMD) e ANV (Intel).
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 libgomp1 curl ca-certificates libvulkan1 mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*
COPY --from=engine-vulkan /opt/audiocpp /opt/audiocpp
ENV QWEN_VARIANT=vulkan \
    LD_LIBRARY_PATH=/opt/audiocpp

FROM ${CUDA_BASE} AS runtime-cuda
# CUDA, cuBLAS, cuFFT e NCCL vêm na imagem runtime; o driver vem da máquina
# (NVIDIA Container Toolkit).
RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 libgomp1 curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=engine-cuda /opt/audiocpp /opt/audiocpp
COPY --from=engine-cpu /opt/audiocpp-cpu /opt/audiocpp-cpu
ENV QWEN_VARIANT=cuda

# ----------------------------------------------------------------- final ----
FROM runtime-${VARIANT}

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JOBS_DIR=/data/jobs \
    JOB_TTL_SECONDS=21600 \
    MAX_UPLOAD_MB=1024 \
    QWEN_BACKEND=auto \
    QWEN_MAX_MODELS=2

# Usuário antes dos COPY: modelos somente-leitura para ele, sem uma camada de
# chown -R que duplicaria 3,6 GB. Os grupos video/render dão acesso à /dev/dri.
RUN useradd --create-home --uid 10001 transcritor \
 && (getent group render >/dev/null || groupadd render) \
 && usermod -aG video,render transcritor \
 && mkdir -p /data/jobs && chown -R transcritor:transcritor /data

COPY --from=venv /opt/venv /opt/venv
COPY --from=models /models /models
WORKDIR /srv
COPY app /srv/app
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

USER transcritor
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
