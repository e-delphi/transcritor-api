# syntax=docker/dockerfile:1
#
# Imagem autocontida: modelos embutidos, nenhum download em runtime, sem token.
#
#   docker build -t transcritor-api:latest .
#   docker run -p 8000:8000 transcritor-api:latest
#
# Modelos disponíveis: tiny, base, small, medium, large-v2, large-v3.
# O padrão é large-v3, o mais preciso. Para imagem menor e muito mais rápida:
#   docker build --build-arg WHISPER_MODEL=small -t transcritor-api:small .

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim AS builder

ARG WHISPER_MODEL=large-v3
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv

# Modelos primeiro: assim mexer nas dependências não refaz o download.
# Ambos os repositórios são públicos e não-gated — nenhum token é necessário.
ENV WHISPER_MODEL_DIR=/models/whisper \
    ECAPA_MODEL_DIR=/models/ecapa
COPY docker/download_models.py /tmp/download_models.py
RUN pip install --no-cache-dir huggingface_hub==0.26.5 \
 && WHISPER_MODEL=${WHISPER_MODEL} python /tmp/download_models.py

COPY requirements.txt /tmp/requirements.txt

# torch CPU-only: evita arrastar ~2 GB de bibliotecas CUDA que não seriam usadas.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 torchaudio==2.5.1 \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Kernels com hardening (WSL2/Docker Desktop) recusam .so marcadas com stack
# executável — o que quebra o carregamento da libctranslate2. Ver o script.
COPY docker/fix_execstack.py /tmp/fix_execstack.py
RUN python /tmp/fix_execstack.py /opt/venv

# Falha o build (em vez do runtime) se alguma extensão nativa não carregar.
RUN python -c "import ctranslate2, faster_whisper, torch, speechbrain, sklearn, fastapi; print('imports ok')"

# --------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG WHISPER_MODEL=large-v3

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_MODEL_DIR=/models/whisper \
    ECAPA_MODEL_DIR=/models/ecapa \
    WHISPER_MODEL=${WHISPER_MODEL} \
    DEVICE=cpu \
    COMPUTE_TYPE=int8 \
    CPU_THREADS=8 \
    MAX_UPLOAD_MB=512 \
    OMP_NUM_THREADS=8 \
    # Trava qualquer acesso de rede das libs de modelo: tudo já está na imagem.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    SB_SAVEDIR=/tmp/speechbrain

# Usuário criado antes dos COPY: os modelos ficam somente-leitura para ele, o que
# evita uma camada de `chown -R` que duplicaria centenas de MB na imagem.
RUN useradd --create-home --uid 10001 transcritor

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models

WORKDIR /srv
COPY app /srv/app

USER transcritor

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
