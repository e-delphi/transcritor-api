# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Motor de transcrição sobre um servidor audio.cpp: Qwen3-ASR + Qwen3-ForcedAligner.

Pensado para legendas: prioriza o instante exato de cada palavra. Nos testes, o
Qwen3-ForcedAligner errou o início das palavras em ~20 ms (mediana), contra ~65 ms
do wav2vec2, e roda na GPU por Vulkan, inclusive em placas AMD.

Fluxo:
1. o áudio é cortado em blocos de até BLOCO_MAX_SEGUNDOS nos trechos mais
   silenciosos; blocos só de silêncio são descartados;
2. o Qwen3-ASR transcreve cada bloco;
3. em português, números por extenso viram algarismos (ver numbers_pt);
4. o Qwen3-ForcedAligner marca o instante de cada palavra do texto.

As duas passadas (transcrever tudo, depois alinhar tudo) evitam que o servidor
troque de modelo a cada bloco. Não há separação de falantes neste motor.
"""

import difflib
import io
import json
import logging
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import wave
from typing import Callable, List, Optional

import numpy as np

from .numbers_pt import to_digits

logger = logging.getLogger(__name__)

# Identificação e recursos do motor; a API valida as opções com isto.
NAME = "audiocpp"
CAPABILITIES = {"diarization": False, "translate": False, "default_diarization": False}

SERVER = os.getenv("AUDIOCPP_URL", "http://127.0.0.1:8081").rstrip("/")
ASR_MODEL = os.getenv("AUDIOCPP_ASR_MODEL", "qwen3-asr")
ALIGN_MODEL = os.getenv("AUDIOCPP_ALIGN_MODEL", "qwen3-align")
REQUEST_TIMEOUT = float(os.getenv("AUDIOCPP_TIMEOUT", "900"))
RATE = 16000

# O alinhador recusa áudio acima de ~120 s; 60 s deixa folga e acerta o corte.
MAX_CHUNK_S = float(os.getenv("BLOCO_MAX_SEGUNDOS", "60"))
SEARCH_S = 15.0          # onde procurar o ponto de corte, antes do limite
FRAME_S = 0.03
SILENCE_PCT = 20         # percentil de energia tratado como silêncio
MIN_VOICED = 0.05        # abaixo disso o bloco é considerado silêncio
# Regiões de fala. Só elas vão para o modelo: em ruído ou silêncio longo o Qwen3
# inventa frases (medido: 90 s de ruído com estalos viraram "É um dos mais
# importantes..."), e um bloco sem fala nenhuma faz o audio.cpp responder 500.
MIN_RUN_S = 0.12         # trecho acima do limiar mais curto que isto é estalo
MIN_SPEECH_S = 0.5       # região com menos fala que isto é descartada
SPLIT_GAP_S = 5.0        # silêncio maior que isto separa blocos
PAD_S = 0.3              # folga antes e depois da fala

# Laço do modelo: em música ou ruído contínuo o Qwen3 às vezes repete a mesma
# palavra ou frase até o limite (medido numa gravação de 2 h: blocos de 50 s com
# 258 palavras e só 3 distintas). Uma frase de até LOOP_MAX_UNIT palavras
# repetida LOOP_MIN_REPS vezes seguidas fica com LOOP_KEEP repetições.
LOOP_MAX_UNIT = 8
LOOP_MIN_REPS = 4
LOOP_KEEP = 2

# Idiomas que o Qwen3-ForcedAligner aceita, pelo nome que ele espera.
ALIGN_LANGUAGES = {
    "pt": "Portuguese", "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "ru": "Russian", "ko": "Korean", "ja": "Japanese", "zh": "Chinese",
    "yue": "Cantonese",
}

# O Qwen3-ASR repete o idioma pedido ("pt"), mas quando detecta sozinho devolve o
# nome em inglês ("Portuguese"). Sem normalizar, a detecção automática perdia o
# alinhador e a conversão de números, que procuram pelo código.
LANGUAGE_CODES = {name.lower(): code for code, name in ALIGN_LANGUAGES.items()}


def language_code(language: Optional[str]) -> Optional[str]:
    lang = (language or "").strip().lower()
    return LANGUAGE_CODES.get(lang, lang) or None


class NoSpeech(Exception):
    """O audio.cpp não achou texto no bloco (responde 500 em vez de texto vazio)."""

TRANSCRIBE_SHARE = 0.5   # peso da transcrição no progresso; o alinhamento leva o resto

ProgressFn = Optional[Callable[[str, float], None]]


# ------------------------------------------------------------------- áudio --
def decode(path: str) -> np.ndarray:
    """Qualquer formato de áudio ou vídeo -> PCM 16 bits, 16 kHz, mono."""
    try:
        import av  # PyAV traz o FFmpeg embutido; é o caminho da instalação nativa
    except ImportError:
        av = None

    if av is not None:
        parts = []
        with av.open(path) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise ValueError("o arquivo não tem trilha de áudio")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    parts.append(out.to_ndarray().reshape(-1))
            for out in resampler.resample(None):
                parts.append(out.to_ndarray().reshape(-1))
        if not parts:
            raise ValueError("o áudio está vazio")
        return np.concatenate(parts).astype(np.int16)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("para decodificar o áudio é preciso o pacote PyAV ou o ffmpeg")
    out = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path, "-ac", "1", "-ar", str(RATE),
         "-f", "s16le", "-"],
        capture_output=True, check=True,
    )
    return np.frombuffer(out.stdout, dtype=np.int16)


def _split_long(rms: np.ndarray, start: int, end: int) -> List[tuple]:
    """Parte [start, end) em blocos de até MAX_CHUNK_S, no vale de energia perto do limite."""
    max_f, search_f = int(MAX_CHUNK_S / FRAME_S), int(SEARCH_S / FRAME_S)
    out = []
    while start < end:
        if end - start <= max_f:
            stop = end
        else:
            lo, hi = start + max_f - search_f, start + max_f
            window = np.convolve(rms[lo:hi], np.ones(10) / 10, mode="same")
            stop = lo + int(np.argmin(window))
        out.append((start, stop))
        start = stop
    return out


def energy_chunks(samples: np.ndarray) -> List[dict]:
    """Blocos de até MAX_CHUNK_S cobrindo só as regiões com fala.

    1. quadro "com voz" = energia acima do dobro do percentil 20 (o chão de ruído);
    2. sequências mais curtas que MIN_RUN_S são estalos e não contam;
    3. regiões separadas por menos de SPLIT_GAP_S ficam no mesmo bloco;
    4. região com menos de MIN_SPEECH_S de fala é descartada;
    5. região longa é partida no ponto mais silencioso perto de MAX_CHUNK_S.
    """
    frame = int(FRAME_S * RATE)
    n = len(samples) // frame
    if n == 0:
        return []
    rms = np.sqrt(np.mean(samples[: n * frame].astype(np.float64).reshape(n, frame) ** 2, axis=1))
    silence = np.percentile(rms, SILENCE_PCT)
    loud = rms > silence * 2
    min_run = int(round(MIN_RUN_S / FRAME_S))
    gap = int(round(SPLIT_GAP_S / FRAME_S))
    pad = int(round(PAD_S / FRAME_S))
    min_speech = int(round(MIN_SPEECH_S / FRAME_S))

    runs, i = [], 0
    while i < n:
        if loud[i]:
            j = i
            while j < n and loud[j]:
                j += 1
            if j - i >= min_run:
                runs.append([i, j, j - i])
            i = j
        else:
            i += 1

    regions = []
    for s0, e0, v in runs:
        if regions and s0 - regions[-1][1] < gap:
            regions[-1][1] = e0
            regions[-1][2] += v
        else:
            regions.append([s0, e0, v])

    chunks = []
    for s0, e0, v in regions:
        if v < min_speech:
            continue
        for a, b in _split_long(rms, max(0, s0 - pad), min(n, e0 + pad)):
            voiced = float(np.mean(loud[a:b]))
            if voiced >= MIN_VOICED:
                chunks.append({"start": a * FRAME_S, "end": b * FRAME_S, "voiced": voiced})
    return chunks


def wav_bytes(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.astype(np.int16).tobytes())
    return buf.getvalue()


# -------------------------------------------------------------------- http --
def _post(route: str, fields: dict, audio: bytes) -> dict:
    boundary = "----transcritor" + os.urandom(8).hex()
    body = bytearray()
    for key, value in fields.items():
        if value is None:
            continue
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode("utf-8")
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="bloco.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode()
    body += audio
    body += f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{SERVER}{route}", data=bytes(body), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        if "did not contain transcript text" in detail:
            raise NoSpeech(detail) from exc
        raise RuntimeError(f"audio.cpp respondeu {exc.code} em {route}: {detail}") from exc


def _get(route: str) -> dict:
    with urllib.request.urlopen(f"{SERVER}{route}", timeout=15) as resp:
        return json.loads(resp.read())


# ------------------------------------------------------------ palavras/texto --
def _key(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalnum())


def attach_times(text: str, aligned: List[dict]) -> List[dict]:
    """Devolve as palavras do texto original, com a pontuação, e o tempo do alinhador.

    O alinhador devolve as palavras sem pontuação e às vezes divide um token
    ("A.C." -> "A", "C"). Os dois lados são casados pela forma sem pontuação.
    """
    tokens = text.split()
    a_keys = [_key(t) for t in tokens]
    b_keys = [_key(w.get("word") or "") for w in aligned]
    times: List[Optional[tuple]] = [None] * len(tokens)
    matcher = difflib.SequenceMatcher(a=a_keys, b=b_keys, autojunk=False)
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal" or (tag == "replace" and a1 - a0 == b1 - b0):
            for k in range(a1 - a0):
                w = aligned[b0 + k]
                times[a0 + k] = (w.get("start"), w.get("end"))
        elif tag == "replace" and a1 - a0 == 1 and "".join(b_keys[b0:b1]) == a_keys[a0]:
            times[a0] = (aligned[b0].get("start"), aligned[b1 - 1].get("end"))
    out = []
    for token, t in zip(tokens, times):
        item = {"word": token}
        if t and t[0] is not None:
            item["start"] = round(float(t[0]), 3)
            item["end"] = round(float(t[1] if t[1] is not None else t[0]), 3)
        out.append(item)
    return out


def collapse_loops(text: str) -> str:
    """Corta repetições em laço ("vai vai vai vai vai" -> "vai vai").

    Compara sem pontuação e sem caixa; a pontuação final da última repetição
    é preservada ("sim, sim, sim, sim." -> "sim, sim.").
    """
    words = text.split()
    keys = [_key(w) for w in words]
    out, i = [], 0
    while i < len(words):
        best = None
        for n in range(1, LOOP_MAX_UNIT + 1):
            if i + n > len(words):
                break
            unit = keys[i:i + n]
            if not any(unit):
                continue
            reps, j = 1, i + n
            while keys[j:j + n] == unit:
                reps += 1
                j += n
            if reps >= LOOP_MIN_REPS and (best is None or reps * n > best[0] * best[1]):
                best = (n, reps)
        if best is None:
            out.append(words[i])
            i += 1
            continue
        n, reps = best
        kept = words[i:i + n * LOOP_KEEP]
        last = words[i + n * reps - 1]
        tail = last[len(last.rstrip(".,;:!?…»\"')]")):]
        core = kept[-1].rstrip(".,;:!?…»\"')]")
        kept[-1] = core + tail
        out.extend(kept)
        i += n * reps
    return " ".join(out)


def spread_times(text: str, start: float, end: float) -> List[dict]:
    """Sem alinhador para o idioma: distribui o bloco pelas palavras, pelo tamanho."""
    tokens = text.split()
    total = sum(len(t) + 1 for t in tokens) or 1
    out, cursor = [], start
    for t in tokens:
        span = (end - start) * (len(t) + 1) / total
        out.append({"word": t, "start": round(cursor, 3), "end": round(cursor + span, 3)})
        cursor += span
    return out


# ------------------------------------------------------------------- motor --
class Engine:
    def __init__(self) -> None:
        try:
            health = _get("/health")
            models = {m.get("id") for m in _get("/v1/models").get("data", [])}
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"servidor audio.cpp indisponível em {SERVER}: {exc}") from exc
        missing = {ASR_MODEL, ALIGN_MODEL} - models
        if missing:
            raise RuntimeError(f"o servidor audio.cpp não tem os modelos: {', '.join(sorted(missing))}")
        self._lock = threading.Lock()
        logger.info("audio.cpp em %s (%s), modelos %s e %s", SERVER, health.get("backend"), ASR_MODEL, ALIGN_MODEL)

    def run(self, path: str, language: Optional[str] = None, task: str = "transcribe",
            diarization: bool = False, alignment: bool = True, on_progress: ProgressFn = None,
            **_ignored) -> dict:
        def report(stage: str, fraction: float) -> None:
            if on_progress:
                on_progress(stage, max(0.0, min(1.0, fraction)))

        report("decoding", 0.0)
        samples = decode(path)
        duration = len(samples) / RATE
        chunks = energy_chunks(samples)
        # "Portuguese" vira "pt": o alinhador e a conversão de números procuram o código.
        lang = language_code(language)

        with self._lock:
            # 1a passada: transcrição de todos os blocos.
            for i, c in enumerate(chunks):
                report("transcribing", TRANSCRIBE_SHARE * i / max(len(chunks), 1))
                audio = wav_bytes(samples[int(c["start"] * RATE):int(c["end"] * RATE)])
                try:
                    res = _post("/v1/audio/transcriptions/details", {"model": ASR_MODEL, "language": lang}, audio)
                except NoSpeech:
                    # Bloco sem fala reconhecível: fica vazio, o arquivo segue.
                    res = {"text": ""}
                c["language"] = language_code(res.get("language") or lang)
                text = collapse_loops(" ".join((res.get("text") or "").split()))
                c["text"] = to_digits(text) if (c["language"] or "").startswith("pt") else text
                c["audio"] = audio

            # 2a passada: instante de cada palavra.
            for i, c in enumerate(chunks):
                report("aligning", TRANSCRIBE_SHARE + (1 - TRANSCRIBE_SHARE) * i / max(len(chunks), 1))
                lang_c = c["language"] or ""
                name = ALIGN_LANGUAGES.get(lang_c) or ALIGN_LANGUAGES.get(lang_c[:2])
                c["aligned"] = False
                if not c["text"]:
                    c["words"] = []
                elif alignment and name:
                    res = _post("/v1/audio/alignments",
                                {"model": ALIGN_MODEL, "language": name, "text": c["text"]}, c["audio"])
                    words = attach_times(c["text"], res.get("words", []))
                    for w in words:
                        if "start" in w:
                            w["start"] = round(w["start"] + c["start"], 3)
                            w["end"] = round(w["end"] + c["start"], 3)
                    c["words"] = words
                    c["aligned"] = True
                else:
                    c["words"] = spread_times(c["text"], c["start"], c["end"])
                c.pop("audio", None)

        report("aligning", 1.0)
        return self._assemble(chunks, duration)

    @staticmethod
    def _assemble(chunks: List[dict], duration: float) -> dict:
        segments = [
            {"id": i, "start": round(c["start"], 3), "end": round(c["end"], 3), "speaker": None,
             "text": c["text"], "aligned": c["aligned"], "words": c["words"]}
            for i, c in enumerate(ch for ch in chunks if ch["text"])
        ]
        langs = [c["language"] for c in chunks if c.get("language")]
        return {
            "language": max(set(langs), key=langs.count) if langs else None,
            "duration": round(duration, 3),
            "alignment": any(s["aligned"] for s in segments),
            "diarization": False,
            "num_speakers": 0,
            "speakers": [],
            "text": " ".join(s["text"] for s in segments).strip(),
            "turns": [],
            "by_speaker": {},
            "segments": segments,
            "engine": "audiocpp:qwen3",
        }


_instance: Optional[Engine] = None
_init_lock = threading.Lock()


def get_engine() -> Engine:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = Engine()
    return _instance
