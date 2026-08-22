# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Decodificação de áudio/vídeo para PCM mono 16 kHz via ffmpeg."""

import subprocess

import numpy as np

SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    pass


def decode_to_mono16k(path: str) -> np.ndarray:
    """Converte qualquer container suportado pelo ffmpeg em float32 mono 16 kHz.

    Aceita áudio e vídeo (a trilha de áudio é extraída automaticamente).
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        path,
        "-vn",
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise AudioDecodeError(stderr.strip().splitlines()[-1] if stderr.strip() else "ffmpeg falhou")

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    if pcm.size == 0:
        raise AudioDecodeError("nenhuma trilha de áudio decodificável no arquivo enviado")

    return (pcm.astype(np.float32) / 32768.0).copy()
