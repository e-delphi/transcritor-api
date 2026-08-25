# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Pipeline de transcrição + diarização, totalmente offline."""

import logging
import os
import threading
from typing import Callable, List, Optional

import numpy as np
from faster_whisper import WhisperModel

from .audio_io import SAMPLE_RATE, decode_to_mono16k
from .diarization import diarize, speaker_at
from .embedding import load_embedder

logger = logging.getLogger(__name__)

WHISPER_DIR = os.getenv("WHISPER_MODEL_DIR", "/models/whisper")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "0"))

# Pesos de cada fase no progresso total. A transcrição domina o tempo; decodificar
# é quase instantâneo e a diarização fica com o restante.
DECODE_SHARE = 0.03
TRANSCRIBE_SHARE = 0.82


class Transcriber:
    def __init__(self) -> None:
        logger.info("Carregando Whisper de %s (device=%s, compute=%s)", WHISPER_DIR, DEVICE, COMPUTE_TYPE)
        self.whisper = WhisperModel(
            WHISPER_DIR,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            cpu_threads=CPU_THREADS,
            local_files_only=True,
        )
        logger.info("Carregando embedder de falantes")
        self.embedder = load_embedder()
        # CTranslate2 não é thread-safe para chamadas concorrentes no mesmo objeto
        # de modelo; serializa as requisições.
        self._lock = threading.Lock()
        logger.info("Modelos prontos")

    def run(
        self,
        path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        diarization: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 8,
        beam_size: int = 5,
        initial_prompt: Optional[str] = None,
        on_progress: Optional[Callable[[str, float], None]] = None,
    ) -> dict:
        def report(stage: str, fraction: float) -> None:
            if on_progress:
                on_progress(stage, max(0.0, min(1.0, fraction)))

        report("decoding", 0.0)
        audio = decode_to_mono16k(path)
        duration = len(audio) / SAMPLE_RATE
        report("decoding", DECODE_SHARE)

        with self._lock:
            segments_iter, info = self.whisper.transcribe(
                audio,
                language=language,
                task=task,
                beam_size=beam_size,
                vad_filter=True,
                word_timestamps=True,
                initial_prompt=initial_prompt,
            )
            # O stage muda antes do primeiro segmento sair: o Whisper processa em
            # janelas de 30 s, então há uma espera até o primeiro resultado aparecer.
            report("transcribing", DECODE_SHARE)

            # O gerador só transcreve sob demanda, então consumi-lo aos poucos dá
            # o progresso real: quanto do áudio já foi coberto por segmentos.
            segments = []
            for segment in segments_iter:
                segments.append(segment)
                if duration > 0:
                    done = min(segment.end / duration, 1.0)
                    report("transcribing", DECODE_SHARE + TRANSCRIBE_SHARE * done)

            report("diarizing", DECODE_SHARE + TRANSCRIBE_SHARE)
            speaker_turns: List[dict] = []
            if diarization and segments:
                regions = [(s.start, s.end) for s in segments]
                speaker_turns = diarize(
                    audio,
                    regions,
                    self.embedder,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )

        report("diarizing", 1.0)
        return self._assemble(segments, speaker_turns, info, duration, diarization)

    def _assemble(self, segments, speaker_turns, info, duration, diarization) -> dict:
        out_segments: List[dict] = []

        for idx, seg in enumerate(segments):
            words = []
            for word in seg.words or []:
                words.append(
                    {
                        "start": round(word.start, 3),
                        "end": round(word.end, 3),
                        "word": word.word,
                        "probability": round(word.probability, 4),
                        "speaker": speaker_at(speaker_turns, word.start, word.end)
                        if speaker_turns
                        else None,
                    }
                )

            speaker = None
            if speaker_turns:
                # Voto majoritário das palavras; sem palavras, usa o intervalo do segmento.
                votes: dict = {}
                for word in words:
                    if word["speaker"]:
                        votes[word["speaker"]] = votes.get(word["speaker"], 0.0) + (
                            word["end"] - word["start"]
                        )
                speaker = (
                    max(votes, key=votes.get)
                    if votes
                    else speaker_at(speaker_turns, seg.start, seg.end)
                )

            out_segments.append(
                {
                    "id": idx,
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "speaker": speaker,
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "words": words,
                }
            )

        turns = self._build_turns(out_segments) if speaker_turns else []
        speakers = sorted({t["speaker"] for t in turns if t["speaker"]})

        by_speaker = {}
        for name in speakers:
            owned = [t for t in turns if t["speaker"] == name]
            by_speaker[name] = {
                "total_time": round(sum(t["end"] - t["start"] for t in owned), 3),
                "turns": len(owned),
                "text": " ".join(t["text"] for t in owned).strip(),
            }

        return {
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(duration, 3),
            "diarization": bool(speaker_turns),
            "num_speakers": len(speakers),
            "speakers": speakers,
            "text": " ".join(s["text"] for s in out_segments).strip(),
            "turns": turns,
            "by_speaker": by_speaker,
            "segments": out_segments,
        }

    @staticmethod
    def _build_turns(segments: List[dict]) -> List[dict]:
        """Reagrupa as palavras em turnos contínuos por falante.

        Quebra segmentos do Whisper que contenham troca de falante no meio.
        """
        turns: List[dict] = []

        def push(speaker, start, end, text):
            text = text.strip()
            if not text:
                return
            if turns and turns[-1]["speaker"] == speaker:
                turns[-1]["end"] = end
                turns[-1]["text"] = f"{turns[-1]['text']} {text}".strip()
            else:
                turns.append({"speaker": speaker, "start": start, "end": end, "text": text})

        for seg in segments:
            if not seg["words"]:
                push(seg["speaker"], seg["start"], seg["end"], seg["text"])
                continue

            current = seg["words"][0]["speaker"] or seg["speaker"]
            buffer, start, end = [], seg["words"][0]["start"], seg["words"][0]["end"]
            for word in seg["words"]:
                speaker = word["speaker"] or current
                if speaker != current:
                    push(current, start, end, "".join(buffer))
                    current, buffer, start = speaker, [], word["start"]
                buffer.append(word["word"])
                end = word["end"]
            push(current, start, end, "".join(buffer))

        for turn in turns:
            turn["start"] = round(turn["start"], 3)
            turn["end"] = round(turn["end"], 3)
        return turns


_instance: Optional[Transcriber] = None
_init_lock = threading.Lock()


def get_transcriber() -> Transcriber:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = Transcriber()
    return _instance
