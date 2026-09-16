#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Baixa os modelos para dentro da imagem no momento do build.

O modelo de diarização do pyannote exige aceite de termos no HuggingFace, então o
build recebe um token por `--mount=type=secret`. O token é usado só aqui: ele não
entra em nenhuma camada da imagem e o container roda sem token nenhum.

Os demais modelos são públicos.
"""

import os
import shutil
import sys

from huggingface_hub import snapshot_download

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DIR = os.getenv("WHISPER_MODEL_DIR", "/models/whisper")
HF_CACHE = os.getenv("HF_HOME", "/models/hf")
TORCH_HOME = os.getenv("TORCH_HOME", "/models/torch")
ALIGN_LANGUAGES = os.getenv("ALIGN_LANGUAGES", "pt,en").replace(" ", "").split(",")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
TOKEN_FILE = os.getenv("HF_TOKEN_FILE", "/run/secrets/hf_token")
NLTK_DIR = os.getenv("NLTK_DATA", "/models/nltk")

WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

# Alinhadores servidos pelo HuggingFace, por idioma (espelha o mapa do WhisperX).
ALIGN_HF = {"pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese"}
# Alinhadores servidos pelo torchaudio, que não passam pelo HuggingFace.
ALIGN_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

# Pesos alternativos que o repositório guarda mas o WhisperX nunca usa: juntos
# somam cerca de 2,4 GB que não precisam entrar na imagem.
ALIGN_IGNORE = ["*.msgpack", "*.h5", "language_model/*", "log_*", "*.tflite"]


def read_token() -> str:
    if not os.path.exists(TOKEN_FILE):
        print(f"ERRO: secret com o token não encontrado em {TOKEN_FILE}.")
        print("Construa com: docker build --secret id=hf_token,src=<caminho do arquivo> ...")
        return ""
    with open(TOKEN_FILE, encoding="utf-8") as handle:
        return handle.read().strip()


def fetch(repo: str, *, local_dir=None, cache_dir=None, allow=None, ignore=None, token=None) -> None:
    print(f"==> baixando {repo}", flush=True)
    snapshot_download(
        repo_id=repo,
        local_dir=local_dir,
        cache_dir=cache_dir,
        allow_patterns=allow,
        ignore_patterns=ignore,
        token=token or None,
        max_workers=4,
    )


def main() -> int:
    if WHISPER_MODEL not in WHISPER_REPOS:
        print(f"WHISPER_MODEL inválido: {WHISPER_MODEL}. Opções: {', '.join(WHISPER_REPOS)}")
        return 1

    fetch(WHISPER_REPOS[WHISPER_MODEL], local_dir=WHISPER_DIR)
    shutil.rmtree(os.path.join(WHISPER_DIR, ".cache"), ignore_errors=True)

    token = read_token()
    if not token:
        return 1
    fetch(DIARIZATION_MODEL, cache_dir=HF_CACHE, token=token)

    for lang in ALIGN_LANGUAGES:
        if lang in ALIGN_HF:
            fetch(ALIGN_HF[lang], cache_dir=HF_CACHE, ignore=ALIGN_IGNORE)
        elif lang in ALIGN_TORCH:
            # O torchaudio guarda os pesos no TORCH_HOME; baixar é só instanciar.
            import torchaudio

            print(f"==> baixando alinhador torchaudio de '{lang}'", flush=True)
            torchaudio.pipelines.__dict__[ALIGN_TORCH[lang]].get_model()
        else:
            print(f"AVISO: sem alinhador conhecido para '{lang}'; será transcrito sem alinhamento")

    # O align() do WhisperX divide frases com o punkt do NLTK, que por padrão é
    # baixado na primeira execução. Sem isto aqui o alinhamento falha em qualquer
    # container sem rede — e falha em silêncio, caindo nos timestamps do Whisper.
    import nltk

    print(f"==> baixando punkt_tab (NLTK) -> {NLTK_DIR}", flush=True)
    if not nltk.download("punkt_tab", download_dir=NLTK_DIR, quiet=True):
        print("ERRO: falha ao baixar punkt_tab")
        return 1

    for path in (WHISPER_DIR, HF_CACHE, NLTK_DIR):
        if not os.path.isdir(path) or not os.listdir(path):
            print(f"ERRO: {path} vazio")
            return 1

    total = sum(
        os.path.getsize(os.path.join(root, f))
        for base in (WHISPER_DIR, HF_CACHE, TORCH_HOME, NLTK_DIR)
        if os.path.isdir(base)
        for root, _, files in os.walk(base)
        for f in files
    )
    print(f"==> modelos gravados: {total / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
