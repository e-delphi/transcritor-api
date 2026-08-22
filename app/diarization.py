# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Diarização de falantes sem token: embeddings ECAPA + clustering aglomerativo.

Substitui o pyannote.audio (que exige aceite de termos e token HuggingFace) por
`speechbrain/spkrec-ecapa-voxceleb`, que é Apache-2.0 e não é gated — portanto
pode ser baixado no build da imagem e usado 100% offline.
"""

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from .audio_io import SAMPLE_RATE

logger = logging.getLogger(__name__)

WINDOW_SECONDS = 1.5
HOP_SECONDS = 0.75
MIN_WINDOW_SECONDS = 0.5
# Distância cosseno máxima para duas janelas serem consideradas do mesmo falante.
CLUSTER_THRESHOLD = float(os.getenv("DIARIZATION_THRESHOLD", "0.65"))
# Clusters com menos janelas que isto são tratados como ruído e absorvidos.
MIN_SPEAKER_WINDOWS = 3


class SpeakerEmbedder:
    """Wrapper carregado uma única vez, a partir do modelo embutido na imagem."""

    def __init__(self, model_dir: str, device: str = "cpu"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device
        # savedir fora de /models: a árvore de modelos é montada somente-leitura.
        savedir = os.getenv("SB_SAVEDIR", "/tmp/speechbrain")
        os.makedirs(savedir, exist_ok=True)
        self.model = EncoderClassifier.from_hparams(
            source=model_dir,
            savedir=savedir,
            run_opts={"device": device},
        )
        self.model.eval()

    @torch.inference_mode()
    def embed(self, windows: np.ndarray, lengths: np.ndarray, batch_size: int = 16) -> np.ndarray:
        """windows: (n, samples) com zero-padding; lengths: comprimento relativo (0-1)."""
        out = []
        for i in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[i : i + batch_size]).to(self.device)
            wav_lens = torch.from_numpy(lengths[i : i + batch_size]).to(self.device)
            emb = self.model.encode_batch(batch, wav_lens=wav_lens)
            out.append(emb.squeeze(1).cpu().numpy())
        embeddings = np.concatenate(out, axis=0)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-8)


def _build_windows(
    audio: np.ndarray, regions: Sequence[Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """Fatia as regiões de fala em janelas sobrepostas de tamanho fixo."""
    win_len = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop_len = int(HOP_SECONDS * SAMPLE_RATE)
    min_len = int(MIN_WINDOW_SECONDS * SAMPLE_RATE)

    chunks: List[np.ndarray] = []
    spans: List[Tuple[float, float]] = []

    for start_s, end_s in regions:
        start = max(0, int(start_s * SAMPLE_RATE))
        end = min(len(audio), int(end_s * SAMPLE_RATE))
        if end - start < min_len:
            continue

        pos = start
        while pos < end:
            stop = min(pos + win_len, end)
            if stop - pos < min_len:
                # cauda curta: estende para trás para aproveitar o áudio restante
                pos = max(start, stop - win_len)
                if stop - pos < min_len:
                    break
            chunks.append(audio[pos:stop])
            spans.append((pos / SAMPLE_RATE, stop / SAMPLE_RATE))
            if stop >= end:
                break
            pos += hop_len

    if not chunks:
        return np.empty((0, win_len), np.float32), np.empty((0,), np.float32), []

    padded = np.zeros((len(chunks), win_len), dtype=np.float32)
    lengths = np.zeros(len(chunks), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        padded[i, : len(chunk)] = chunk
        lengths[i] = len(chunk) / win_len

    return padded, lengths, spans


def _cluster(
    embeddings: np.ndarray,
    num_speakers: Optional[int],
    min_speakers: int,
    max_speakers: int,
) -> np.ndarray:
    n = len(embeddings)
    if n == 1:
        return np.zeros(1, dtype=int)

    def ward(k: int) -> np.ndarray:
        # Ward sobre embeddings L2-normalizados: a distancia euclidiana ao quadrado
        # equivale a 2*(1-cosseno), mas o linkage e bem menos sensivel a janelas
        # atipicas -- com "average" um unico trecho ruidoso pode virar um cluster e
        # forcar os falantes reais a se fundirem.
        return AgglomerativeClustering(
            n_clusters=max(1, min(k, n)), linkage="ward"
        ).fit_predict(embeddings)

    if num_speakers is not None:
        return ward(num_speakers)

    # Sem contagem informada, corta o dendrograma por distancia em vez de por k:
    # nao impoe um numero de falantes, e um trecho atipico vira um cluster proprio
    # (absorvido no passo seguinte) em vez de distorcer a particao inteira.
    labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=CLUSTER_THRESHOLD,
        metric="cosine",
        linkage="average",
    ).fit_predict(embeddings)

    found = len(set(labels))
    if found > max_speakers:
        return ward(max_speakers)
    if found < min_speakers:
        return ward(min_speakers)
    return labels


def _absorb_tiny_clusters(labels: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Reatribui clusters pequenos demais ao falante mais parecido.

    Sem isto, uma palavra isolada ou um trecho ruidoso aparece no resultado como
    um falante extra que nunca existiu.
    """
    unique, counts = np.unique(labels, return_counts=True)
    keep = [int(u) for u, c in zip(unique, counts) if c >= MIN_SPEAKER_WINDOWS]
    if not keep or len(keep) == len(unique):
        return labels

    centroids = {}
    for label in keep:
        centroid = embeddings[labels == label].mean(axis=0)
        centroids[label] = centroid / max(np.linalg.norm(centroid), 1e-8)

    out = labels.copy()
    for i, label in enumerate(labels):
        if int(label) not in keep:
            out[i] = max(keep, key=lambda c: float(embeddings[i] @ centroids[c]))
    return out


def _rename_by_first_appearance(
    labels: np.ndarray, spans: Sequence[Tuple[float, float]]
) -> List[str]:
    order: List[int] = []
    for label in labels[np.argsort([s for s, _ in spans], kind="stable")]:
        if label not in order:
            order.append(int(label))
    mapping = {label: f"SPEAKER_{i:02d}" for i, label in enumerate(order)}
    return [mapping[int(label)] for label in labels]


def diarize(
    audio: np.ndarray,
    regions: Sequence[Tuple[float, float]],
    embedder: SpeakerEmbedder,
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 8,
) -> List[dict]:
    """Retorna [{start, end, speaker}] cobrindo as regiões de fala informadas."""
    windows, lengths, spans = _build_windows(audio, regions)
    if len(windows) == 0:
        return []

    if len(windows) == 1:
        return [{"start": spans[0][0], "end": spans[0][1], "speaker": "SPEAKER_00"}]

    embeddings = embedder.embed(windows, lengths)
    labels = _cluster(embeddings, num_speakers, min_speakers, max_speakers)
    labels = _absorb_tiny_clusters(labels, embeddings)
    names = _rename_by_first_appearance(labels, spans)

    frames = [
        {"start": span[0], "end": span[1], "speaker": name}
        for span, name in zip(spans, names)
    ]
    frames.sort(key=lambda f: f["start"])

    # Funde janelas vizinhas do mesmo falante em turnos contínuos.
    merged: List[dict] = []
    for frame in frames:
        if merged and merged[-1]["speaker"] == frame["speaker"] and frame["start"] <= merged[-1]["end"] + 0.1:
            merged[-1]["end"] = max(merged[-1]["end"], frame["end"])
        else:
            merged.append(dict(frame))
    return merged


def speaker_at(segments: Sequence[dict], start: float, end: float) -> Optional[str]:
    """Falante com maior sobreposição temporal no intervalo [start, end]."""
    best_speaker, best_overlap = None, 0.0
    for seg in segments:
        overlap = min(end, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best_speaker, best_overlap = seg["speaker"], overlap

    if best_speaker is not None:
        return best_speaker

    # Sem sobreposição: cai para o segmento temporalmente mais próximo.
    center = (start + end) / 2
    nearest, nearest_dist = None, float("inf")
    for seg in segments:
        dist = 0.0 if seg["start"] <= center <= seg["end"] else min(
            abs(center - seg["start"]), abs(center - seg["end"])
        )
        if dist < nearest_dist:
            nearest, nearest_dist = seg["speaker"], dist
    return nearest
