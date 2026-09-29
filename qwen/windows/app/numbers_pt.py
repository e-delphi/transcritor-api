# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Converte números escritos por extenso em português para algarismos.

O Qwen3-ASR escreve números por extenso ("setenta"), o que alonga as legendas.
Segue a convenção comum em legendagem: de zero a dez por extenso, acima disso em
algarismos. Manter os pequenos por extenso também evita converter o artigo "um".

O "e" só liga uma parte a outra menor, como na fala: dezena a unidade ("vinte e
cinco"), centena a dezena ("cento e dez"), milhar ao resto ("dois mil e vinte").
Assim "dois e três" continua sendo dois números, e não vira 5.
"""

import os
import re

KEEP_WORDS_UP_TO = int(os.getenv("NUMEROS_POR_EXTENSO_ATE", "10"))

UNITS = {
    "zero": 0, "um": 1, "uma": 1, "dois": 2, "duas": 2, "três": 3, "tres": 3, "quatro": 4,
    "cinco": 5, "seis": 6, "sete": 7, "oito": 8, "nove": 9,
}
TEENS = {
    "dez": 10, "onze": 11, "doze": 12, "treze": 13, "quatorze": 14, "catorze": 14,
    "quinze": 15, "dezesseis": 16, "dezasseis": 16, "dezessete": 17, "dezassete": 17,
    "dezoito": 18, "dezenove": 19, "dezanove": 19,
}
TENS = {
    "vinte": 20, "trinta": 30, "quarenta": 40, "cinquenta": 50, "sessenta": 60,
    "setenta": 70, "oitenta": 80, "noventa": 90,
}
HUNDREDS = {
    "cem": 100, "cento": 100, "duzentos": 200, "duzentas": 200, "trezentos": 300,
    "trezentas": 300, "quatrocentos": 400, "quatrocentas": 400, "quinhentos": 500,
    "quinhentas": 500, "seiscentos": 600, "seiscentas": 600, "setecentos": 700,
    "setecentas": 700, "oitocentos": 800, "oitocentas": 800, "novecentos": 900,
    "novecentas": 900,
}
SCALES = {"mil": 1_000, "milhão": 1_000_000, "milhões": 1_000_000,
          "bilhão": 1_000_000_000, "bilhões": 1_000_000_000}
SCALE_NAMES = {1_000_000: ("milhão", "milhões"), 1_000_000_000: ("bilhão", "bilhões")}

# Separa a palavra da pontuação colada a ela, preservando ambas.
TOKEN = re.compile(r"^([«\"'(\[]*)(.*?)([.,;:!?»\"')\]…]*)$")


def _kind(word: str):
    w = word.lower()
    if w in UNITS:
        return "unit", UNITS[w]
    if w in TEENS:
        return "teen", TEENS[w]
    if w in TENS:
        return "ten", TENS[w]
    if w in HUNDREDS:
        return "hundred", HUNDREDS[w]
    if w in SCALES:
        return "scale", SCALES[w]
    return None, None


# Depois de cada tipo de parte, o maior valor que pode vir ligado por "e".
AFTER_E_LIMIT = {"ten": 10, "hundred": 100, "scale": 1000}


def _parse(words: list) -> tuple:
    """Lê a maior frase numérica no início de `words`.

    Devolve (valor, quantidade de palavras consumidas) ou (None, 0).
    """
    total, group, used, last = 0, 0, 0, None
    i = 0
    while i < len(words):
        _, core, trail = TOKEN.match(words[i]).groups()
        kind, value = _kind(core)
        connector = False
        if kind is None and core.lower() == "e" and last in AFTER_E_LIMIT and i + 1 < len(words):
            _, nxt, _ = TOKEN.match(words[i + 1]).groups()
            nkind, nvalue = _kind(nxt)
            if nkind and nkind != "scale" and nvalue < AFTER_E_LIMIT[last]:
                connector = True
                i += 1
                core, trail, kind, value = nxt, TOKEN.match(words[i]).groups()[2], nkind, nvalue
        if kind is None:
            break
        if not connector and last is not None:
            # Sem "e", só valem: parte seguida de escala ("dois mil") ou escala
            # seguida de centena/dezena/unidade ("mil novecentos").
            if not (kind == "scale" or last == "scale"):
                break
        if kind == "scale":
            # "mil milhões" e "milhão" sem número antes ficam fora do escopo.
            if last == "scale" or (last is None and value > 1000):
                break
            total += (group or 1) * value
            group = 0
        else:
            group += value
        used = i + 1
        last = kind
        i += 1
        if trail and trail[-1] in ".,;:!?…":
            break  # pontuação encerra a frase numérica
    if used == 0:
        return None, 0
    return total + group, used


def _render(value: int, words: list) -> str:
    for scale, (singular, plural) in SCALE_NAMES.items():
        if value >= scale and value % scale == 0 and value < scale * 1000:
            n = value // scale
            return f"{n} {singular if n == 1 else plural}"
    if value >= 10_000:
        return f"{value:,}".replace(",", ".")
    return str(value)


def to_digits(text: str) -> str:
    """Troca por algarismos as frases numéricas acima de KEEP_WORDS_UP_TO."""
    words = text.split(" ")
    out, i = [], 0
    while i < len(words):
        value, used = _parse(words[i:]) if words[i] else (None, 0)
        if value is None:
            out.append(words[i])
            i += 1
            continue
        chunk = words[i:i + used]
        lead = TOKEN.match(chunk[0]).group(1)
        trail = TOKEN.match(chunk[-1]).group(3)
        single_mil = used == 1 and TOKEN.match(chunk[0]).group(2).lower() == "mil"
        if value <= KEEP_WORDS_UP_TO or single_mil:
            out.extend(chunk)  # pequenos e "mil" sozinho ficam por extenso
        else:
            rendered = lead + _render(value, chunk)
            rest = words[i + used:i + used + 2]
            if (not trail and len(rest) == 2 and rest[0].lower() == "por"
                    and TOKEN.match(rest[1]).group(2).lower() == "cento"):
                out.append(rendered + "%" + TOKEN.match(rest[1]).group(3))
                i += used + 2
                continue
            out.append(rendered + trail)
        i += used
    return " ".join(out)
