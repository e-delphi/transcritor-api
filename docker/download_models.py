#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Baixa os modelos para dentro da imagem no momento do build.

Ambos os repositórios são públicos e não-gated: nenhum token HuggingFace é
necessário, nem no build nem em runtime.
"""

import os
import shutil
import sys

from huggingface_hub import snapshot_download

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DIR = os.getenv("WHISPER_MODEL_DIR", "/models/whisper")
ECAPA_DIR = os.getenv("ECAPA_MODEL_DIR", "/models/ecapa")

WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}


def fetch(repo: str, target: str, allow_patterns=None) -> None:
    print(f"==> baixando {repo} -> {target}", flush=True)
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=repo,
        local_dir=target,
        allow_patterns=allow_patterns,
        max_workers=4,
    )
    # O cache .git-like do hub dobraria o tamanho da camada.
    shutil.rmtree(os.path.join(target, ".cache"), ignore_errors=True)


def main() -> int:
    if WHISPER_MODEL not in WHISPER_REPOS:
        print(f"WHISPER_MODEL inválido: {WHISPER_MODEL}. Opções: {', '.join(WHISPER_REPOS)}")
        return 1

    fetch(WHISPER_REPOS[WHISPER_MODEL], WHISPER_DIR)
    fetch(
        "speechbrain/spkrec-ecapa-voxceleb",
        ECAPA_DIR,
        allow_patterns=["*.yaml", "*.ckpt", "*.txt", "*.json"],
    )

    # O hyperparams.yaml aponta para o repo remoto; reescreve para o diretório local
    # de modo que o SpeechBrain nunca tente acessar a rede.
    hparams = os.path.join(ECAPA_DIR, "hyperparams.yaml")
    if os.path.exists(hparams):
        with open(hparams, encoding="utf-8") as handle:
            content = handle.read()
        content = content.replace("speechbrain/spkrec-ecapa-voxceleb", ECAPA_DIR)
        with open(hparams, "w", encoding="utf-8") as handle:
            handle.write(content)

    for path in (WHISPER_DIR, ECAPA_DIR):
        files = sorted(os.listdir(path))
        print(f"==> {path}: {files}", flush=True)
        if not files:
            print(f"ERRO: {path} vazio")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
