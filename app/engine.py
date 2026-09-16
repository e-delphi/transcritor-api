# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Motor de transcrição baseado em WhisperX.

Três etapas encadeadas, todas com modelos gravados na imagem:

1. transcrição com faster-whisper (`large-v3`), em lotes guiados por VAD;
2. alinhamento forçado com wav2vec2, que dá timestamps precisos por palavra;
3. diarização com pyannote, que atribui o falante de cada palavra.

O VAD é o do pyannote, cujos pesos vêm dentro do próprio pacote WhisperX — não há
download nem repositório restrito envolvido. O modo Silero seria pior aqui: ele
baixa o modelo do GitHub via `torch.hub` na primeira execução, o que exige rede e
permissão de escrita no cache do torch.

Em runtime nada é baixado e nenhum token é usado — os pesos do pyannote entram na
imagem durante o build, via secret, e `HF_HUB_OFFLINE` fecha qualquer acesso.
"""

import logging
import os
import threading
from typing import Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

WHISPER_DIR = os.getenv("WHISPER_MODEL_DIR", "/models/whisper")
DIARIZATION_MODEL = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
DEVICE = os.getenv("DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "int8")
CPU_THREADS = int(os.getenv("CPU_THREADS", "8"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

# Idiomas com alinhador embutido na imagem. Os demais transcrevem normalmente,
# apenas sem o refinamento de timestamp por palavra.
ALIGN_LANGUAGES = set(os.getenv("ALIGN_LANGUAGES", "pt,en").replace(" ", "").split(","))

# Pesos de cada etapa no progresso total. A transcrição domina o tempo.
TRANSCRIBE_SHARE = 0.60
ALIGN_SHARE = 0.20
DIARIZE_SHARE = 0.20

ProgressFn = Optional[Callable[[str, float], None]]


class Engine:
    """Modelos carregados uma vez e reutilizados entre requisições."""

    def __init__(self) -> None:
        import whisperx

        self._whisperx = whisperx
        logger.info("Carregando Whisper de %s (%s/%s)", WHISPER_DIR, DEVICE, COMPUTE_TYPE)
        self.asr = whisperx.load_model(
            WHISPER_DIR,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            threads=CPU_THREADS,
            vad_method="pyannote",
            local_files_only=True,
        )

        logger.info("Carregando diarização (%s)", DIARIZATION_MODEL)
        from whisperx.diarize import DiarizationPipeline

        self.diarizer = DiarizationPipeline(model_name=DIARIZATION_MODEL, device=DEVICE)

        # Alinhadores são carregados sob demanda e mantidos em cache: cada um ocupa
        # mais de 1 GB e só o idioma realmente usado precisa estar em memória.
        self._align_cache: Dict[str, tuple] = {}
        self._lock = threading.Lock()
        logger.info("Modelos prontos (alinhamento disponível: %s)", ", ".join(sorted(ALIGN_LANGUAGES)))

    def _align_model(self, language: str):
        if language not in self._align_cache:
            logger.info("Carregando alinhador de '%s'", language)
            self._align_cache[language] = self._whisperx.load_align_model(
                language_code=language, device=DEVICE, model_cache_only=True
            )
        return self._align_cache[language]

    def run(
        self,
        path: str,
        language: Optional[str] = None,
        task: str = "transcribe",
        diarization: bool = True,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        alignment: bool = True,
        on_progress: ProgressFn = None,
    ) -> dict:
        def report(stage: str, fraction: float) -> None:
            if on_progress:
                on_progress(stage, max(0.0, min(1.0, fraction)))

        report("decoding", 0.0)
        audio = self._whisperx.load_audio(path)
        duration = len(audio) / 16000.0

        # Os modelos não são seguros para chamadas concorrentes no mesmo objeto.
        with self._lock:
            report("transcribing", 0.0)
            result = self.asr.transcribe(
                audio,
                batch_size=BATCH_SIZE,
                language=language,
                task=task,
                progress_callback=lambda p: report("transcribing", TRANSCRIBE_SHARE * p / 100.0),
            )
            detected = result.get("language") or language or "??"
            base = TRANSCRIBE_SHARE

            aligned = False
            if alignment and detected in ALIGN_LANGUAGES and result.get("segments"):
                try:
                    model, metadata = self._align_model(detected)
                    result = self._whisperx.align(
                        result["segments"],
                        model,
                        metadata,
                        audio,
                        DEVICE,
                        return_char_alignments=False,
                        progress_callback=lambda p: report("aligning", base + ALIGN_SHARE * p / 100.0),
                    )
                    aligned = True
                except Exception:  # noqa: BLE001 - alinhar é um refinamento, não um requisito
                    logger.exception("Alinhamento falhou; seguindo com os timestamps do Whisper")
            base += ALIGN_SHARE

            speakers_found = False
            if diarization and result.get("segments"):
                report("diarizing", base)
                diarize_df = self.diarizer(
                    audio,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    progress_callback=lambda p: report("diarizing", base + DIARIZE_SHARE * p / 100.0),
                )
                result = self._whisperx.assign_word_speakers(diarize_df, result)
                speakers_found = True

        report("diarizing", 1.0)
        return self._assemble(result, detected, duration, aligned, speakers_found)

    @staticmethod
    def _assemble(result: dict, language: str, duration: float, aligned: bool, diarized: bool) -> dict:
        segments: List[dict] = []
        for idx, seg in enumerate(result.get("segments", [])):
            words = []
            for word in seg.get("words", []) or []:
                # Palavras sem timestamp aparecem quando o alinhador não reconhece
                # o token (números, símbolos); mantê-las preserva o texto.
                item = {"word": word.get("word", "")}
                if word.get("start") is not None:
                    item["start"] = round(float(word["start"]), 3)
                    item["end"] = round(float(word.get("end", word["start"])), 3)
                if word.get("score") is not None:
                    item["score"] = round(float(word["score"]), 4)
                if word.get("speaker"):
                    item["speaker"] = word["speaker"]
                words.append(item)

            segments.append(
                {
                    "id": idx,
                    "start": round(float(seg.get("start", 0.0)), 3),
                    "end": round(float(seg.get("end", 0.0)), 3),
                    "speaker": seg.get("speaker"),
                    "text": (seg.get("text") or "").strip(),
                    "words": words,
                }
            )

        turns = _build_turns(segments) if diarized else []
        speakers = sorted({t["speaker"] for t in turns if t.get("speaker")})

        by_speaker = {}
        for name in speakers:
            owned = [t for t in turns if t["speaker"] == name]
            by_speaker[name] = {
                "total_time": round(sum(t["end"] - t["start"] for t in owned), 3),
                "turns": len(owned),
                "text": " ".join(t["text"] for t in owned).strip(),
            }

        return {
            "language": language,
            "duration": round(duration, 3),
            "alignment": aligned,
            "diarization": diarized,
            "num_speakers": len(speakers),
            "speakers": speakers,
            "text": " ".join(s["text"] for s in segments).strip(),
            "turns": turns,
            "by_speaker": by_speaker,
            "segments": segments,
        }


def _build_turns(segments: List[dict]) -> List[dict]:
    """Reagrupa as palavras em turnos contínuos por falante.

    Divide o segmento quando o falante muda no meio dele, o que acontece bastante
    em diálogo rápido.
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
        timed = [w for w in seg["words"] if "start" in w]
        if not timed:
            push(seg.get("speaker"), seg["start"], seg["end"], seg["text"])
            continue

        current = timed[0].get("speaker") or seg.get("speaker")
        buffer, start, end = [], timed[0]["start"], timed[0]["end"]
        for word in timed:
            speaker = word.get("speaker") or current
            if speaker != current:
                push(current, start, end, " ".join(buffer))
                current, buffer, start = speaker, [], word["start"]
            buffer.append(word["word"])
            end = word["end"]
        push(current, start, end, " ".join(buffer))

    for turn in turns:
        turn["start"] = round(turn["start"], 3)
        turn["end"] = round(turn["end"], 3)
    return turns


_instance: Optional[Engine] = None
_init_lock = threading.Lock()


def get_engine() -> Engine:
    global _instance
    if _instance is None:
        with _init_lock:
            if _instance is None:
                _instance = Engine()
    return _instance
