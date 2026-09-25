# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Conversão do resultado para os formatos de download."""

from typing import List


def _timestamp(seconds: float, comma: bool) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _blocks(result: dict) -> List[dict]:
    """Prefere os turnos por falante; sem diarização, usa os segmentos."""
    return result.get("turns") or result.get("segments") or []


def to_txt(result: dict) -> str:
    """Texto corrido, ou o diálogo rotulado quando há falantes."""
    turns = result.get("turns")
    if not turns:
        return result.get("text", "")
    return "\n".join(f"{t['speaker']}: {t['text']}" for t in turns)


def to_srt(result: dict) -> str:
    lines = []
    for i, block in enumerate(_blocks(result), 1):
        speaker = block.get("speaker")
        text = block["text"] if not speaker else f"[{speaker}] {block['text']}"
        lines.append(
            f"{i}\n{_timestamp(block['start'], True)} --> {_timestamp(block['end'], True)}\n{text}\n"
        )
    return "\n".join(lines)


def to_vtt(result: dict) -> str:
    lines = ["WEBVTT", ""]
    for block in _blocks(result):
        speaker = block.get("speaker")
        text = block["text"] if not speaker else f"[{speaker}] {block['text']}"
        lines.append(f"{_timestamp(block['start'], False)} --> {_timestamp(block['end'], False)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


RENDERERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt}
MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json",
}
