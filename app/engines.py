# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Escolha do motor de transcrição pela variável ENGINE.

  whisperx  Whisper + wav2vec2 + pyannote (a imagem Docker). Separa falantes.
  audiocpp  Qwen3-ASR + Qwen3-ForcedAligner num servidor audio.cpp. Mais rápido
            e com o instante de cada palavra mais preciso; sem separação de
            falantes. É o motor da instalação nativa no Windows.

Os módulos são importados só quando escolhidos: a instalação nativa não tem o
WhisperX nem o PyTorch.
"""

import os

ENGINE = os.getenv("ENGINE", "whisperx").strip().lower()

CAPABILITIES = {
    "whisperx": {"diarization": True, "translate": True, "default_diarization": True},
    "audiocpp": {"diarization": False, "translate": False, "default_diarization": False},
}

if ENGINE not in CAPABILITIES:
    raise RuntimeError(f"ENGINE={ENGINE!r} desconhecido; use um de: {', '.join(CAPABILITIES)}")


def capabilities() -> dict:
    return CAPABILITIES[ENGINE]


def get_engine():
    if ENGINE == "audiocpp":
        from .engine_audiocpp import get_engine as factory
    else:
        from .engine import get_engine as factory
    return factory()
