#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo
#
# Baixa um arquivo e confere o SHA-256; falhando a fonte principal, tenta o
# espelho no Google Drive (mesmo arquivo, mesmo SHA). Usado só no build.
#
#   fetch.sh <destino> <sha256> <url principal> [id no Google Drive]
set -eu

dest="$1"
sha="$2"
url="$3"
drive_id="${4:-}"

ok() {
    echo "$sha  $dest" | sha256sum -c --quiet - 2>/dev/null
}

get() {
    rm -f "$dest"
    curl -fL --retry 3 --retry-delay 5 -o "$dest" "$1" && ok
}

echo "baixando $(basename "$dest")"
if get "$url"; then
    exit 0
fi
if [ -n "$drive_id" ]; then
    echo "  fonte principal falhou; tentando o espelho no Google Drive"
    if get "https://drive.usercontent.google.com/download?id=${drive_id}&export=download&confirm=t"; then
        exit 0
    fi
fi
echo "  ERRO: $(basename "$dest") não baixou ou o SHA-256 não confere" >&2
exit 1
