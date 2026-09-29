# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Monta legendas curtas a partir do instante de cada palavra.

Um bloco de legenda fecha quando a frase termina, quando há uma pausa longa ou
quando a próxima palavra estouraria o tamanho ou a duração máxima. Nesse último
caso o corte recua até a última vírgula ou pausa curta dentro do bloco, para não
partir uma locução ao meio.

Dentro do bloco, a quebra entre as linhas busca linhas de tamanho parecido e evita
deixar uma palavra de ligação ("de", "que", "para") no fim da primeira linha.
"""

import os

MAX_LINE = int(os.getenv("LEGENDA_MAX_CARACTERES", "42"))
MAX_LINES = int(os.getenv("LEGENDA_MAX_LINHAS", "2"))
MAX_DURATION = float(os.getenv("LEGENDA_MAX_SEGUNDOS", "7.0"))
MIN_DURATION = float(os.getenv("LEGENDA_MIN_SEGUNDOS", "1.0"))
PAUSE_BREAK = 0.7    # pausa que sempre separa blocos
SOFT_PAUSE = 0.25    # pausa curta, boa para cortar quando o bloco estoura
MIN_GAP = 0.08       # intervalo mínimo entre dois blocos, para não piscarem colados
LINGER = 0.5         # quanto a legenda pode ficar depois da fala, se houver espaço
MAX_WORD = 2.5       # nenhuma palavra dura mais que isto na tela

WEAK = {
    "a", "o", "as", "os", "de", "do", "da", "dos", "das", "e", "ou", "que", "em", "no", "na",
    "nos", "nas", "um", "uma", "para", "pra", "com", "por", "pelo", "pela", "se", "ao", "à",
    "às", "aos", "mas", "nem", "lhe", "me", "te", "nosso", "nossa", "seu", "sua",
}
SENTENCE_END = (".", "?", "!", "…")
CLOSERS = "\"'»)]”"


def _bare(word: str) -> str:
    return word.rstrip(CLOSERS)


def _ends_sentence(word: str) -> bool:
    return _bare(word).endswith(SENTENCE_END)


def _ends_clause(word: str) -> bool:
    return _bare(word).endswith((",", ";", ":", "—", "–"))


def _text(words: list, prefix: str = "") -> str:
    return prefix + " ".join(w["word"] for w in words)


def _layout(text: str, max_line: int, max_lines: int):
    """Quebra o texto em até `max_lines` linhas de até `max_line`. None se não couber."""
    if len(text) <= max_line:
        return [text]
    if max_lines < 2:
        return None
    tokens = text.split(" ")
    if max_lines == 2:
        best, best_score = None, None
        for k in range(1, len(tokens)):
            a, b = " ".join(tokens[:k]), " ".join(tokens[k:])
            if len(a) > max_line or len(b) > max_line:
                continue
            last = _bare(tokens[k - 1]).lower().strip(".,;:!?")
            score = abs(len(a) - len(b))
            if last in WEAK:
                score += 12
            if _ends_clause(tokens[k - 1]) or _ends_sentence(tokens[k - 1]):
                score -= 8
            if best_score is None or score < best_score:
                best, best_score = [a, b], score
        return best
    # Mais de duas linhas: preenchimento guloso.
    lines, cur = [], ""
    for tok in tokens:
        cand = f"{cur} {tok}".strip()
        if len(cand) <= max_line:
            cur = cand
        else:
            lines.append(cur)
            cur = tok
    lines.append(cur)
    return lines if len(lines) <= max_lines and all(len(l) <= max_line for l in lines) else None


def _fill_times(words: list) -> list:
    """Palavras sem instante herdam o dos vizinhos, para não sumirem da legenda."""
    out = [dict(w) for w in words if (w.get("word") or "").strip()]
    for i, w in enumerate(out):
        if w.get("start") is None:
            prev_end = next((out[j].get("end") for j in range(i - 1, -1, -1) if out[j].get("end") is not None), None)
            next_start = next((out[j].get("start") for j in range(i + 1, len(out)) if out[j].get("start") is not None), None)
            w["start"] = prev_end if prev_end is not None else (next_start or 0.0)
        if w.get("end") is None or w["end"] < w["start"]:
            nxt = next((out[j].get("start") for j in range(i + 1, len(out)) if out[j].get("start") is not None), None)
            w["end"] = max(w["start"], nxt if nxt is not None else w["start"] + 0.3)
        # Alinhadores às vezes esticam a última palavra de um trecho por todo o
        # silêncio seguinte ("melhor." durando 20 s). Limita pelo tamanho da palavra.
        w["end"] = min(w["end"], w["start"] + min(MAX_WORD, 0.5 + 0.09 * len(w["word"])))
    return out


def build_cues(words: list, max_line: int = MAX_LINE, max_lines: int = MAX_LINES,
               max_duration: float = MAX_DURATION, speakers: bool = False) -> list:
    """Palavras [{word, start, end, speaker?}] -> blocos [{start, end, lines, speaker}]."""
    words = _fill_times(words)
    blocks, cur = [], []

    def prefix(group):
        return f"[{group[0]['speaker']}] " if speakers and group and group[0].get("speaker") else ""

    def fits(group):
        return _layout(_text(group, prefix(group)), max_line, max_lines) is not None

    def emit(group):
        if group:
            blocks.append(group)

    for w in words:
        if cur:
            gap = w["start"] - cur[-1]["end"]
            speaker_change = speakers and w.get("speaker") != cur[-1].get("speaker")
            too_long = not fits(cur + [w]) or (w["end"] - cur[0]["start"]) > max_duration
            if speaker_change or gap >= PAUSE_BREAK:
                emit(cur)
                cur = []
            elif too_long:
                # Recua até a última vírgula ou pausa curta, se houver uma razoável.
                cut = None
                for k in range(len(cur) - 1, 0, -1):
                    if _ends_clause(cur[k - 1]["word"]) or cur[k]["start"] - cur[k - 1]["end"] >= SOFT_PAUSE:
                        if len(_text(cur[:k])) >= 0.35 * max_line * max_lines:
                            cut = k
                        break
                if cut:
                    emit(cur[:cut])
                    cur = cur[cut:]
                    if not fits(cur + [w]) or (w["end"] - cur[0]["start"]) > max_duration:
                        emit(cur)
                        cur = []
                else:
                    emit(cur)
                    cur = []
        cur.append(w)
        if _ends_sentence(w["word"]):
            emit(cur)
            cur = []
    emit(cur)

    # Um bloco curto demais para ler é juntado ao seguinte, se couberem juntos.
    merged = []
    for group in blocks:
        if merged:
            prev = merged[-1]
            short = prev[-1]["end"] - prev[0]["start"] < MIN_DURATION
            close = group[0]["start"] - prev[-1]["end"] < PAUSE_BREAK
            same = not speakers or prev[0].get("speaker") == group[0].get("speaker")
            joined = prev + group
            if short and close and same and fits(joined) and joined[-1]["end"] - joined[0]["start"] <= max_duration:
                merged[-1] = joined
                continue
        merged.append(group)

    cues = []
    for group in merged:
        text = _text(group, prefix(group))
        lines = _layout(text, max_line, max_lines) or [text]  # palavra gigante: não há como quebrar
        cues.append({"start": group[0]["start"], "end": group[-1]["end"], "lines": lines,
                     "speaker": group[0].get("speaker")})

    # Tempo de tela: fica um pouco depois da fala e respeita a duração mínima, mas
    # nunca invade o bloco seguinte.
    for i, c in enumerate(cues):
        limit = cues[i + 1]["start"] - MIN_GAP if i + 1 < len(cues) else None
        target = max(c["end"] + LINGER, c["start"] + MIN_DURATION)
        end = target if limit is None else min(target, limit)
        c["end"] = max(end, c["end"] if limit is None else min(c["end"], limit))
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.01  # blocos colados no tempo: sem espaço possível
    return cues
