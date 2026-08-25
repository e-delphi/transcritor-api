# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Embeddings de voz com WeSpeaker ResNet293 (ONNX).

Substitui o ECAPA-TDNN do SpeechBrain, que é de 2020 e produzia embeddings
ruidosos o bastante para fragmentar falantes em áudio real. O ResNet293-LM é o
maior modelo da família WeSpeaker e roda no onnxruntime que a imagem já usa para
o VAD, o que dispensa a dependência do SpeechBrain.

O modelo espera features Fbank no padrão Kaldi, não waveform bruto, e um sinal na
escala de int16 — os mesmos parâmetros usados pelo WeSpeaker original.
"""

import logging
import os
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import torch
import torchaudio.compliance.kaldi as kaldi

logger = logging.getLogger(__name__)

NUM_MEL_BINS = 80
FRAME_LENGTH_MS = 25
FRAME_SHIFT_MS = 10
SAMPLE_RATE = 16000
EMBEDDING_DIM = 256


class SpeakerEmbedder:
    """Carregado uma vez, a partir do modelo gravado na imagem."""

    def __init__(self, model_path: str, num_threads: int = 0):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads > 0:
            options.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Embedder WeSpeaker carregado de %s", model_path)

    @staticmethod
    def _fbank(window: np.ndarray) -> np.ndarray:
        """Fbank Kaldi com normalização de média por janela (CMN)."""
        # O WeSpeaker opera na escala de int16; nosso áudio vem em [-1, 1].
        wav = torch.from_numpy(window).unsqueeze(0) * (1 << 15)
        feat = kaldi.fbank(
            wav,
            num_mel_bins=NUM_MEL_BINS,
            frame_length=FRAME_LENGTH_MS,
            frame_shift=FRAME_SHIFT_MS,
            sample_frequency=SAMPLE_RATE,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        return feat.numpy().astype(np.float32)

    def embed(self, windows: List[np.ndarray], batch_size: int = 16) -> np.ndarray:
        """Um embedding L2-normalizado por janela, na ordem de entrada."""
        if not windows:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        feats = [self._fbank(w) for w in windows]

        # O pooling estatístico do modelo é sensível a frames de padding, então
        # janelas de durações diferentes são processadas em lotes separados.
        by_length: Dict[int, List[int]] = {}
        for idx, feat in enumerate(feats):
            by_length.setdefault(feat.shape[0], []).append(idx)

        out = np.zeros((len(windows), EMBEDDING_DIM), dtype=np.float32)
        for indices in by_length.values():
            for start in range(0, len(indices), batch_size):
                chunk = indices[start : start + batch_size]
                batch = np.stack([feats[i] for i in chunk], axis=0)
                embs = self.session.run(None, {self.input_name: batch})[0]
                out[chunk] = embs

        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-8)


def load_embedder() -> SpeakerEmbedder:
    path = os.getenv("EMBEDDER_MODEL_PATH", "/models/embedder/model.onnx")
    threads = int(os.getenv("CPU_THREADS", "0"))
    return SpeakerEmbedder(path, num_threads=threads)
