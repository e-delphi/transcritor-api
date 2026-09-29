#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo
#
# Sobe o servidor audio.cpp e a API. O backend sai de QWEN_BACKEND:
#   auto    (padrão) CUDA na imagem cuda; na vulkan, Vulkan se houver placa de
#           verdade e CPU se não houver
#   vulkan | cuda | cpu   força
# Se o backend escolhido não subir (placa não repassada ao container, driver
# ausente), cai para a CPU em vez de parar: lento, mas transcreve.
set -eu

ENGINE_DIR=/opt/audiocpp
# Na imagem cuda, a queda para a CPU usa uma versão só-CPU à parte: o servidor
# CUDA exige a libcuda.so.1 do driver e não abre sem placa NVIDIA.
CPU_DIR=$ENGINE_DIR
[ -x /opt/audiocpp-cpu/audiocpp_server ] && CPU_DIR=/opt/audiocpp-cpu
PORT=8081
CONFIG=/tmp/audiocpp.json

backend="${QWEN_BACKEND:-auto}"
if [ "$backend" = "auto" ]; then
    if [ "$QWEN_VARIANT" = "cuda" ]; then
        backend=cuda
    elif "$ENGINE_DIR/audiocpp_cli" --list-devices 2>/dev/null \
            | grep '^Vulkan:' | grep -viq 'llvmpipe\|lavapipe\|swiftshader'; then
        # llvmpipe/lavapipe é o Vulkan EM SOFTWARE do Mesa: listado como
        # placa, mas mais lento que o próprio backend de CPU.
        backend=vulkan
    else
        backend=cpu
    fi
fi

write_config() {
    python3 - "$1" <<EOF
import json, os, sys
json.dump({
    "host": "127.0.0.1", "port": $PORT, "backend": sys.argv[1], "device": 0,
    "threads": int(os.environ.get("QWEN_THREADS") or min(os.cpu_count() or 4, 8)),
    "lazy_load": True,
    "max_loaded_models": int(os.environ.get("QWEN_MAX_MODELS", "2")),
    "idle_unload_ms": 0, "min_free_memory_mb": 0,
    "models": [
        {"id": "qwen3-asr", "family": "qwen3_asr", "task": "asr", "mode": "offline",
         "path": "/models/qwen3-asr-1.7b-q8_0.gguf"},
        {"id": "qwen3-align", "family": "qwen3_forced_aligner", "task": "align",
         "mode": "offline", "path": "/models/qwen3-forced-aligner-0.6b-q8_0.gguf"},
    ],
}, open("$CONFIG", "w"))
EOF
}

start_server() {
    # $1 = backend, $2 = pasta do motor
    write_config "$1"
    LD_LIBRARY_PATH="$2" "$2/audiocpp_server" --config "$CONFIG" --no-ui &
    SERVER_PID=$!
    i=0
    while [ $i -lt 120 ]; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return 1
        fi
        if curl -fs "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
        i=$((i + 1))
    done
    kill "$SERVER_PID" 2>/dev/null || true
    return 1
}

echo "audio.cpp: backend $backend"
if [ "$backend" = "cpu" ]; then dir=$CPU_DIR; else dir=$ENGINE_DIR; fi
if ! start_server "$backend" "$dir"; then
    if [ "$backend" = "cpu" ]; then
        echo "ERRO: o servidor audio.cpp não subiu" >&2
        exit 1
    fi
    if [ "$backend" = "cuda" ]; then
        hint="rode com --gpus all e o NVIDIA Container Toolkit"
    else
        hint="repasse a placa com --device /dev/dri"
    fi
    echo "AVISO: o backend $backend não subiu ($hint); usando a CPU, bem mais devagar" >&2
    backend=cpu
    start_server cpu "$CPU_DIR" || { echo "ERRO: o servidor audio.cpp não subiu" >&2; exit 1; }
fi

export AUDIOCPP_URL="http://127.0.0.1:$PORT"
export QWEN_BACKEND_ACTIVE="$backend"
exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 300
