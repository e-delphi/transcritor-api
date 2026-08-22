# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""API HTTP de transcrição com separação por falante. Sem autenticação."""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .audio_io import AudioDecodeError
from .pipeline import get_transcriber

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("transcritor-api")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "512"))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Carrega os modelos na subida para que a primeira requisição não pague o custo.
    get_transcriber()
    yield


app = FastAPI(
    title="Transcritor API",
    description="Transcrição com Whisper + diarização de falantes, offline e sem token.",
    version="1.0.0",
    lifespan=lifespan,
)


def _optional_int(value: Optional[str], field: str) -> Optional[int]:
    """Campos de formulário chegam como texto; string vazia significa 'não informado'."""
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        raise HTTPException(400, f"{field} deve ser um número inteiro") from None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": True}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio ou vídeo"),
    language: Optional[str] = Form(None, description="Código ISO (pt, en...). Vazio = detecta"),
    task: str = Form("transcribe", description="transcribe ou translate"),
    diarization: bool = Form(True, description="Separar por falante"),
    num_speakers: Optional[str] = Form(None, description="Número exato de falantes, se conhecido"),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
    beam_size: Optional[str] = Form(None),
    initial_prompt: Optional[str] = Form(None),
):
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task deve ser 'transcribe' ou 'translate'")

    n_speakers = _optional_int(num_speakers, "num_speakers")
    lo = _optional_int(min_speakers, "min_speakers") or 1
    hi = _optional_int(max_speakers, "max_speakers") or 8
    beam = _optional_int(beam_size, "beam_size") or 5

    if n_speakers is not None and n_speakers < 1:
        raise HTTPException(400, "num_speakers deve ser >= 1")
    if lo < 1 or hi < lo:
        raise HTTPException(400, "intervalo de falantes inválido")
    if beam < 1:
        raise HTTPException(400, "beam_size deve ser >= 1")

    suffix = os.path.splitext(file.filename or "")[1] or ".bin"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"arquivo excede o limite de {MAX_UPLOAD_MB} MB")
                tmp.write(chunk)

        if written == 0:
            raise HTTPException(400, "arquivo vazio")

        logger.info("Transcrevendo %s (%.1f MB)", file.filename, written / 1e6)
        result = get_transcriber().run(
            tmp_path,
            language=language or None,
            task=task,
            diarization=diarization,
            num_speakers=n_speakers,
            min_speakers=lo,
            max_speakers=hi,
            beam_size=beam,
            initial_prompt=initial_prompt or None,
        )
        result["filename"] = file.filename
        return JSONResponse(result)

    except AudioDecodeError as exc:
        raise HTTPException(400, f"não foi possível decodificar o áudio: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha na transcrição")
        raise HTTPException(500, f"erro interno: {exc}") from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/", response_class=PlainTextResponse)
def index() -> str:
    return (
        "Transcritor API\n\n"
        "POST /transcribe   multipart/form-data, campo 'file' -> JSON separado por falante\n"
        "GET  /health\n"
        "GET  /docs         interface interativa\n"
    )
