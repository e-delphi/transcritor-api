# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Diarização de falantes sem token HuggingFace.

Fatia as regiões de fala em janelas sobrepostas, extrai um embedding de voz de
cada uma (WeSpeaker ResNet293) e agrupa as janelas por falante (spectral
clustering com eigengap). Nenhum dos modelos é gated.
"""

import logging
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .audio_io import SAMPLE_RATE
from .clustering import cluster_speakers
from .embedding import SpeakerEmbedder

logger = logging.getLogger(__name__)

# Janelas mais longas que os 1,5 s iniciais: embeddings curtos são instáveis e
# eram uma das causas de falantes fragmentados.
WINDOW_SECONDS = float(os.getenv("DIARIZATION_WINDOW", "2.0"))
HOP_SECONDS = float(os.getenv("DIARIZATION_HOP", "0.75"))
MIN_WINDOW_SECONDS = 0.7
# Segmentos até esta duração viram uma única unidade de análise, em vez de serem
# fatiados. Uma janela deslizante de tamanho fixo atravessa a troca de falante em
# diálogos de turnos curtos e produz embeddings misturados, que apagam a estrutura
# de grupos; a fronteira do segmento respeita a pausa natural da fala.
MAX_SEGMENT_AS_UNIT = float(os.getenv("DIARIZATION_MAX_UNIT", "3.5"))
# Falantes com menos que esta fração da fala total são absorvidos pelo mais
# parecido. É proporcional de propósito: um limite fixo em número de janelas
# funcionava em áudio curto, mas em gravações longas um cluster espúrio
# acumulava janelas suficientes para sobreviver.
MIN_SPEAKER_RATIO = float(os.getenv("DIARIZATION_MIN_RATIO", "0.03"))


def _build_windows(
    audio: np.ndarray, regions: Sequence[Tuple[float, float]]
) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    """Fatia as regiões de fala em janelas sobrepostas."""
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

        # Segmento curto: uma unidade só, respeitando a fronteira natural da fala.
        if (end - start) <= MAX_SEGMENT_AS_UNIT * SAMPLE_RATE:
            chunks.append(audio[start:end])
            spans.append((start / SAMPLE_RATE, end / SAMPLE_RATE))
            continue

        # Segmento longo: pode conter mais de um falante, então é fatiado.
        pos = start
        while pos < end:
            stop = min(pos + win_len, end)
            if stop - pos < min_len:
                # Cauda curta: estende para trás para aproveitar o áudio restante.
                pos = max(start, stop - win_len)
                if stop - pos < min_len:
                    break
            chunks.append(audio[pos:stop])
            spans.append((pos / SAMPLE_RATE, stop / SAMPLE_RATE))
            if stop >= end:
                break
            pos += hop_len

    return chunks, spans


def _absorb_minor_speakers(
    labels: np.ndarray, embeddings: np.ndarray, spans: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """Reatribui falantes com participação irrisória ao mais parecido."""
    durations = {}
    for label, (start, end) in zip(labels, spans):
        durations[int(label)] = durations.get(int(label), 0.0) + (end - start)

    total = sum(durations.values())
    if total <= 0 or len(durations) < 2:
        return labels

    keep = [l for l, d in durations.items() if d / total >= MIN_SPEAKER_RATIO]
    if not keep or len(keep) == len(durations):
        return labels

    centroids = {}
    for label in keep:
        centroid = embeddings[labels == label].mean(axis=0)
        centroids[label] = centroid / max(np.linalg.norm(centroid), 1e-8)

    dropped = len(durations) - len(keep)
    logger.info("Diarização: %d falante(s) marginal(is) absorvido(s)", dropped)

    out = labels.copy()
    for i, label in enumerate(labels):
        if int(label) not in keep:
            out[i] = max(keep, key=lambda c: float(embeddings[i] @ centroids[c]))
    return out


def _rename_by_first_appearance(
    labels: np.ndarray, spans: Sequence[Tuple[float, float]]
) -> List[str]:
    order: List[int] = []
    for idx in np.argsort([s for s, _ in spans], kind="stable"):
        label = int(labels[idx])
        if label not in order:
            order.append(label)
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
    windows, spans = _build_windows(audio, regions)
    if not windows:
        return []
    if len(windows) == 1:
        return [{"start": spans[0][0], "end": spans[0][1], "speaker": "SPEAKER_00"}]

    embeddings = embedder.embed(windows)
    labels = cluster_speakers(embeddings, num_speakers, min_speakers, max_speakers)
    if num_speakers is None:
        labels = _absorb_minor_speakers(labels, embeddings, spans)
    names = _rename_by_first_appearance(labels, spans)

    frames = sorted(
        (
            {"start": span[0], "end": span[1], "speaker": name}
            for span, name in zip(spans, names)
        ),
        key=lambda f: f["start"],
    )

    # Funde janelas vizinhas do mesmo falante em turnos contínuos.
    merged: List[dict] = []
    for frame in frames:
        if (
            merged
            and merged[-1]["speaker"] == frame["speaker"]
            and frame["start"] <= merged[-1]["end"] + 0.1
        ):
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
        dist = (
            0.0
            if seg["start"] <= center <= seg["end"]
            else min(abs(center - seg["start"]), abs(center - seg["end"]))
        )
        if dist < nearest_dist:
            nearest, nearest_dist = seg["speaker"], dist
    return nearest
