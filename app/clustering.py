# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Agrupamento de falantes por spectral clustering com refinamento de afinidade.

Substitui o corte do dendrograma por uma distância fixa, que fragmentava falantes:
um limiar global não se adapta a ruído, canal ou duração, e cada janela ruidosa
podia virar um "falante" novo. Aqui a matriz de similaridade passa por um
refinamento antes do agrupamento e o número de falantes sai do maior salto entre
autovalores (eigengap), em vez de um limiar arbitrário.

O refinamento segue a linha de "Speaker Diarization with LSTM" (Wang et al., 2018):
suavização temporal, poda das similaridades fracas de cada linha e simetrização.
"""

import logging
import os
from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans

logger = logging.getLogger(__name__)

# Fração das menores similaridades zeradas em cada linha. Valores altos podam mais
# ruído, mas fundem falantes parecidos.
PRUNE_PERCENTILE = float(os.getenv("DIARIZATION_PRUNE", "0.60"))
# Janela da suavização temporal, em número de unidades. Desligada por padrão: as
# unidades são segmentos de fala, e em diálogo os vizinhos temporais costumam ser
# justamente falantes diferentes, de modo que suavizar mistura quem se quer separar.
# Só ajuda quando as unidades são janelas deslizantes sobre fala contínua.
SMOOTHING_WINDOW = int(os.getenv("DIARIZATION_SMOOTHING", "1"))
# Salto mínimo entre autovalores consecutivos para aceitar mais de um falante.
# Abaixo disso o grafo não tem estrutura de grupos e o áudio é tratado como monólogo.
MIN_EIGENGAP_RATIO = float(os.getenv("DIARIZATION_EIGENGAP", "1.45"))
# Aceita o MENOR k cuja razão chegue a esta fração da melhor. Sem isso, um k maior
# com vantagem marginal vence o argmax e o resultado ganha falantes inexistentes.
RATIO_TOLERANCE = float(os.getenv("DIARIZATION_RATIO_TOLERANCE", "0.90"))


def _temporal_smoothing(affinity: np.ndarray, window: int) -> np.ndarray:
    """Média móvel sobre as linhas: falantes têm continuidade no tempo.

    Reduz o efeito de uma janela isolada e ruidosa, que sozinha poderia formar um
    cluster próprio.
    """
    if window < 2:
        return affinity
    kernel = np.ones(window) / window
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=affinity
    )
    return (smoothed + smoothed.T) / 2


def _prune_and_symmetrize(affinity: np.ndarray, percentile: float) -> np.ndarray:
    """Zera as similaridades fracas de cada linha e volta a simetrizar.

    Mantém apenas as ligações mais confiáveis de cada janela, o que separa melhor
    os blocos correspondentes a cada falante.
    """
    pruned = affinity.copy()
    if percentile > 0:
        thresholds = np.quantile(pruned, percentile, axis=1, keepdims=True)
        pruned = np.where(pruned < thresholds, 0.0, pruned)
    pruned = np.maximum(pruned, pruned.T)
    np.fill_diagonal(pruned, 1.0)
    return pruned


def _laplacian_eigenvalues(affinity: np.ndarray) -> np.ndarray:
    """Autovalores do laplaciano normalizado, em ordem crescente."""
    degree = affinity.sum(axis=1)
    degree = np.maximum(degree, 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(len(affinity)) - (affinity * d_inv_sqrt).T * d_inv_sqrt
    eigenvalues = np.linalg.eigvalsh((laplacian + laplacian.T) / 2)
    return np.clip(eigenvalues, 0.0, None)


def _estimate_num_speakers(eigenvalues: np.ndarray, lo: int, hi: int) -> int:
    """Número de falantes pelo maior eigengap dentro da faixa permitida.

    O laplaciano de um grafo com k componentes bem separadas tem k autovalores
    próximos de zero; o salto após esses k indica onde a estrutura termina.
    """
    hi = min(hi, len(eigenvalues) - 1)
    if hi < lo:
        return max(1, lo)

    # O k certo é onde a razão entre autovalores consecutivos salta. A razão só vale
    # de eigenvalues[1] em diante: eigenvalues[0] é sempre ~0 e dividir por ele
    # explodiria, elegendo k=1 em qualquer áudio.
    ratios = {
        k: float(eigenvalues[k]) / max(float(eigenvalues[k - 1]), 1e-6)
        for k in range(max(2, lo), hi + 1)
    }
    if not ratios:
        return max(1, lo)

    best_ratio = max(ratios.values())

    # Sem salto relevante não há estrutura de grupos: um único falante.
    if lo <= 1 and best_ratio < MIN_EIGENGAP_RATIO:
        return 1

    # Empate técnico resolve para menos falantes, não para mais.
    for k in sorted(ratios):
        if ratios[k] >= best_ratio * RATIO_TOLERANCE:
            return k
    return max(2, lo)


def cluster_speakers(
    embeddings: np.ndarray,
    num_speakers: Optional[int] = None,
    min_speakers: int = 1,
    max_speakers: int = 8,
) -> np.ndarray:
    """Rótulo de falante para cada janela. Espera embeddings L2-normalizados."""
    n = len(embeddings)
    if n == 1:
        return np.zeros(1, dtype=int)
    if num_speakers is not None and num_speakers <= 1:
        return np.zeros(n, dtype=int)

    # Similaridades negativas viram ausência de ligação. Reescalar com (cos+1)/2
    # deixaria todos os pesos não-negativos, mas comprimiria o contraste — 0,20 e
    # 0,80 virariam 0,60 e 0,90 — jogando fora a separação que o embedder produziu.
    affinity = np.maximum(embeddings @ embeddings.T, 0.0)
    np.fill_diagonal(affinity, 1.0)
    affinity = _temporal_smoothing(affinity, SMOOTHING_WINDOW)
    affinity = _prune_and_symmetrize(affinity, PRUNE_PERCENTILE)

    eigenvalues = _laplacian_eigenvalues(affinity)
    hi = min(max_speakers, n)
    lo = max(1, min(min_speakers, hi))

    k = num_speakers if num_speakers is not None else _estimate_num_speakers(eigenvalues, lo, hi)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)

    logger.info("Diarização: %d janelas -> %d falante(s)", n, k)
    return _spectral_labels(affinity, k, embeddings)


def _spectral_labels(affinity: np.ndarray, k: int, embeddings: np.ndarray) -> np.ndarray:
    """Projeta nos k autovetores principais e agrupa nesse espaço."""
    degree = np.maximum(affinity.sum(axis=1), 1e-10)
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    laplacian = np.eye(len(affinity)) - (affinity * d_inv_sqrt).T * d_inv_sqrt
    _, vectors = np.linalg.eigh((laplacian + laplacian.T) / 2)

    features = vectors[:, :k]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-10)

    try:
        return KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(features)
    except Exception:  # noqa: BLE001 - fallback determinístico
        logger.warning("KMeans falhou; usando agrupamento aglomerativo")
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)
