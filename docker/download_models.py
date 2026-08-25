#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Baixa os modelos para dentro da imagem no momento do build.

Todos os repositórios são públicos e não-gated: nenhum token HuggingFace é
necessário, nem no build nem em runtime.
"""

import os
import shutil
import sys

from huggingface_hub import snapshot_download

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_DIR = os.getenv("WHISPER_MODEL_DIR", "/models/whisper")
EMBEDDER_DIR = os.getenv("EMBEDDER_MODEL_DIR", "/models/embedder")

WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}

EMBEDDER_REPO = "Wespeaker/wespeaker-voxceleb-resnet293-LM"
EMBEDDER_FILE = "voxceleb_resnet293_LM.onnx"


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

    # WeSpeaker ResNet293-LM em ONNX: roda no onnxruntime que a imagem já usa,
    # sem depender do SpeechBrain nem do PyTorch para a inferência.
    fetch(EMBEDDER_REPO, EMBEDDER_DIR, allow_patterns=[EMBEDDER_FILE])
    src = os.path.join(EMBEDDER_DIR, EMBEDDER_FILE)
    dst = os.path.join(EMBEDDER_DIR, "model.onnx")
    if os.path.exists(src):
        os.replace(src, dst)

    for path in (WHISPER_DIR, EMBEDDER_DIR):
        files = sorted(os.listdir(path))
        print(f"==> {path}: {files}", flush=True)
        if not files:
            print(f"ERRO: {path} vazio")
            return 1

    if not os.path.exists(dst):
        print(f"ERRO: modelo do embedder ausente em {dst}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
