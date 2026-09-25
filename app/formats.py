# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Conversão do resultado para os formatos de download."""

from typing import List

from .subtitles import MAX_DURATION, MAX_LINE, MAX_LINES, build_cues


def _timestamp(seconds: float, comma: bool) -> str:
    # Arredonda o total em milissegundos antes de separar as partes: arredondar só os
    # milésimos transformaria 1,9996 s em "00:00:01,1000".
    total_ms = int(round(max(seconds, 0.0) * 1000))
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _wrap(text: str, max_line: int) -> List[str]:
    lines, cur = [], ""
    for tok in text.split():
        cand = f"{cur} {tok}".strip()
        if len(cand) <= max_line or not cur:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    if cur:
        lines.append(cur)
    return lines


def cues(result: dict, max_line: int = MAX_LINE, max_lines: int = MAX_LINES,
         max_duration: float = MAX_DURATION) -> List[dict]:
    """Blocos de legenda. Usa o instante de cada palavra quando o resultado tem."""
    words = [w for s in result.get("segments", []) for w in (s.get("words") or [])
             if (w.get("word") or "").strip()]
    timed = sum(1 for w in words if w.get("start") is not None)
    if words and timed >= 0.5 * len(words):
        return build_cues(words, max_line=max_line, max_lines=max_lines, max_duration=max_duration,
                          speakers=bool(result.get("diarization")))

    # Sem tempo por palavra: um bloco por turno ou segmento, só com as linhas quebradas.
    out = []
    for block in result.get("turns") or result.get("segments") or []:
        speaker = block.get("speaker")
        text = (f"[{speaker}] " if speaker else "") + block.get("text", "")
        out.append({"start": block["start"], "end": block["end"], "lines": _wrap(text, max_line),
                    "speaker": speaker})
    return out


def to_txt(result: dict) -> str:
    """Texto corrido, ou o diálogo rotulado quando há falantes."""
    turns = result.get("turns")
    if not turns:
        return result.get("text", "")
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def to_srt(result: dict, **limits) -> str:
    blocks = []
    for i, c in enumerate(cues(result, **limits), 1):
        blocks.append(f"{i}\n{_timestamp(c['start'], True)} --> {_timestamp(c['end'], True)}\n"
                      + "\n".join(c["lines"]) + "\n")
    return "\n".join(blocks)


def to_vtt(result: dict, **limits) -> str:
    blocks = ["WEBVTT", ""]
    for c in cues(result, **limits):
        blocks.append(f"{_timestamp(c['start'], False)} --> {_timestamp(c['end'], False)}")
        blocks.extend(c["lines"])
        blocks.append("")
    return "\n".join(blocks)


RENDERERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt}
MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json",
}
