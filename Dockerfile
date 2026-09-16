# syntax=docker/dockerfile:1
#
# Imagem autocontida: modelos embutidos, nenhum download em runtime.
#
# O modelo de diarização do pyannote exige aceite de termos no HuggingFace, então
# o BUILD precisa de um token. Ele entra por secret e não fica em nenhuma camada
# da imagem; o container roda sem token e sem rede.
#
#   docker build --secret id=hf_token,src=$HOME/hf_token.txt -t transcritor-api .
#   docker run -p 8000:8000 -v transcritor-dados:/data transcritor-api
#
# Modelos disponíveis: tiny, base, small, medium, large-v2, large-v3.
# O padrão é large-v3, o mais preciso. Para imagem menor e mais rápida:
#   docker build --secret id=hf_token,src=... --build-arg WHISPER_MODEL=small ...

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------- builder ----
FROM python:${PYTHON_VERSION}-slim AS builder

ARG WHISPER_MODEL=large-v3
ARG ALIGN_LANGUAGES=pt,en

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    WHISPER_MODEL_DIR=/models/whisper \
    HF_HOME=/models/hf \
    TORCH_HOME=/models/torch

# ffmpeg no builder: o torchcodec se liga às bibliotecas do ffmpeg na instalação.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

COPY requirements.txt /tmp/requirements.txt

# torch CPU-only: evita arrastar ~2 GB de bibliotecas CUDA que não seriam usadas.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Kernels com hardening (WSL2/Docker Desktop) recusam .so marcadas com stack
# executável, o que quebra o carregamento da libctranslate2. Ver o script.
COPY docker/fix_execstack.py /tmp/fix_execstack.py
RUN python /tmp/fix_execstack.py /opt/venv

# Falha o build (em vez do runtime) se alguma extensão nativa não carregar.
RUN python -c "import whisperx, torch, torchaudio, ctranslate2, pyannote.audio; print('imports ok')"

# Modelos gravados na imagem. O token só existe durante este RUN.
COPY docker/download_models.py /tmp/download_models.py
RUN --mount=type=secret,id=hf_token \
    WHISPER_MODEL=${WHISPER_MODEL} ALIGN_LANGUAGES=${ALIGN_LANGUAGES} \
    NLTK_DATA=/models/nltk python /tmp/download_models.py

# --------------------------------------------------------------- runtime ----
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG WHISPER_MODEL=large-v3
ARG ALIGN_LANGUAGES=pt,en

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    WHISPER_MODEL_DIR=/models/whisper \
    WHISPER_MODEL=${WHISPER_MODEL} \
    ALIGN_LANGUAGES=${ALIGN_LANGUAGES} \
    HF_HOME=/models/hf \
    # O cache do hub é o próprio /models/hf, e não /models/hf/hub: os modelos foram
    # baixados no build com cache_dir apontando para cá. Sem isto as bibliotecas
    # procuram um nível abaixo, não acham nada e tentam ir à rede.
    HF_HUB_CACHE=/models/hf \
    TORCH_HOME=/models/torch \
    NLTK_DATA=/models/nltk \
    DEVICE=cpu \
    COMPUTE_TYPE=int8 \
    CPU_THREADS=8 \
    OMP_NUM_THREADS=8 \
    BATCH_SIZE=8 \
    MAX_UPLOAD_MB=1024 \
    JOBS_DIR=/data/jobs \
    JOB_TTL_SECONDS=21600 \
    # Trava qualquer acesso de rede das libs de modelo: tudo já está na imagem.
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Usuário criado antes dos COPY: os modelos ficam somente-leitura para ele, o que
# evita uma camada de `chown -R` que duplicaria vários GB na imagem.
RUN useradd --create-home --uid 10001 transcritor \
 && mkdir -p /data/jobs \
 && chown -R transcritor:transcritor /data

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models

WORKDIR /srv
COPY app /srv/app

USER transcritor

# Jobs e áudios ficam aqui; monte um volume para sobreviverem ao container.
VOLUME ["/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "300"]
