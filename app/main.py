# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""API HTTP de transcrição e legendas. Sem autenticação."""

import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from .engines import ENGINE, capabilities, get_engine
from .formats import MEDIA_TYPES, RENDERERS
from .jobs import Job, compute_job_id, manager

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger("transcritor-api")

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1024"))


def _worker(job: Job) -> dict:
    """Executa um job da fila. Roda na thread do gerenciador."""
    audio = job.audio_path()
    if audio is None:
        raise RuntimeError("áudio do job não está mais disponível")

    return get_engine().run(
        audio,
        on_progress=lambda stage, fraction: manager.update_progress(job, stage, fraction),
        **job.options,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Carrega os modelos na subida para que a primeira requisição não pague o custo.
    get_engine()
    manager.set_worker(_worker)
    manager.restore()
    manager.start_cleanup_loop()
    yield


app = FastAPI(
    title="Transcritor API",
    description=f"Transcrição e legendas offline, sem token. Motor: {ENGINE}.",
    version="2.1.0",
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


def _options(
    language: Optional[str],
    task: str,
    diarization: Optional[bool],
    alignment: bool,
    num_speakers: Optional[str],
    min_speakers: Optional[str],
    max_speakers: Optional[str],
) -> dict:
    """Parâmetros que influenciam o resultado — e, portanto, o id do job."""
    if task not in ("transcribe", "translate"):
        raise HTTPException(400, "task deve ser 'transcribe' ou 'translate'")

    caps = capabilities()
    if task == "translate" and not caps["translate"]:
        raise HTTPException(400, f"o motor {ENGINE} não traduz; use task=transcribe")
    if diarization is None:
        diarization = caps["default_diarization"]
    if diarization and not caps["diarization"]:
        raise HTTPException(400, f"o motor {ENGINE} não separa falantes; envie diarization=false")

    exact = _optional_int(num_speakers, "num_speakers")
    lo = _optional_int(min_speakers, "min_speakers")
    hi = _optional_int(max_speakers, "max_speakers")

    if exact is not None and exact < 1:
        raise HTTPException(400, "num_speakers deve ser >= 1")
    if lo is not None and lo < 1:
        raise HTTPException(400, "min_speakers deve ser >= 1")
    if lo is not None and hi is not None and hi < lo:
        raise HTTPException(400, "intervalo de falantes inválido")

    return {
        "language": (language or "").strip() or None,
        "task": task,
        "diarization": diarization,
        "alignment": alignment,
        "num_speakers": exact,
        "min_speakers": lo,
        "max_speakers": hi,
    }


async def _save_upload(file: UploadFile) -> Tuple[str, int]:
    """Grava o upload em disco em blocos, respeitando o limite de tamanho."""
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
        return tmp_path, written
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": True, "engine": ENGINE}


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Arquivo de áudio ou vídeo"),
    language: Optional[str] = Form(None, description="Código ISO (pt, en...). Vazio = detecta"),
    task: str = Form("transcribe", description="transcribe ou translate"),
    diarization: Optional[bool] = Form(None, description="Separar por falante (padrão do motor)"),
    alignment: bool = Form(True, description="Alinhar timestamps por palavra"),
    num_speakers: Optional[str] = Form(None, description="Número exato de falantes, se conhecido"),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
):
    """Transcreve e devolve o JSON pronto. Para áudios longos, prefira POST /jobs."""
    opts = _options(language, task, diarization, alignment, num_speakers, min_speakers, max_speakers)
    tmp_path, written = await _save_upload(file)
    try:
        logger.info("Transcrevendo %s (%.1f MB)", file.filename, written / 1e6)
        result = get_engine().run(tmp_path, **opts)
        result["filename"] = file.filename
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha na transcrição")
        raise HTTPException(500, f"erro interno: {exc}") from exc
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/jobs", status_code=202)
async def create_job(
    file: UploadFile = File(..., description="Arquivo de áudio ou vídeo"),
    language: Optional[str] = Form(None),
    task: str = Form("transcribe"),
    diarization: Optional[bool] = Form(None),
    alignment: bool = Form(True),
    num_speakers: Optional[str] = Form(None),
    min_speakers: Optional[str] = Form(None),
    max_speakers: Optional[str] = Form(None),
):
    """Enfileira a transcrição e devolve o id para acompanhar e baixar depois.

    O id é derivado do conteúdo do arquivo: reenviar o mesmo áudio com as mesmas
    opções devolve o resultado pronto imediatamente, sem reprocessar.
    """
    opts = _options(language, task, diarization, alignment, num_speakers, min_speakers, max_speakers)
    tmp_path, written = await _save_upload(file)

    try:
        job_id, file_sha = compute_job_id(tmp_path, {**opts, "engine": ENGINE})
        existing = manager.find(job_id)
        if existing is not None and existing.status != "error":
            logger.info("Job %s reaproveitado (%s)", job_id, existing.status)
            snapshot = existing.snapshot()
            snapshot["cached"] = existing.status == "done"
            snapshot["status_url"] = f"/jobs/{job_id}"
            return JSONResponse(snapshot, status_code=200 if existing.status == "done" else 202)

        job = Job(job_id, file.filename, file_sha, opts)
        os.makedirs(job.dir, exist_ok=True)
        suffix = os.path.splitext(file.filename or "")[1] or ".bin"
        shutil.move(tmp_path, os.path.join(job.dir, f"audio{suffix}"))
        tmp_path = None

        manager.submit(job)
        logger.info("Job %s enfileirado (%s, %.1f MB)", job.id, file.filename, written / 1e6)
        return {**job.snapshot(), "cached": False, "status_url": f"/jobs/{job.id}"}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": manager.list()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, incluir_resultado: bool = Query(False, description="Embutir o resultado")) -> dict:
    job = manager.find(job_id)
    if job is None:
        raise HTTPException(404, "job não encontrado ou expirado")
    return job.snapshot(include_result=incluir_resultado)


@app.get("/jobs/{job_id}/download")
def download_job(
    job_id: str,
    formato: str = Query("json", description="json, txt, srt ou vtt"),
    max_caracteres: Optional[int] = Query(None, ge=10, le=120, description="Caracteres por linha de legenda"),
    max_linhas: Optional[int] = Query(None, ge=1, le=4, description="Linhas por bloco de legenda"),
    max_segundos: Optional[float] = Query(None, ge=1, le=20, description="Duração máxima de um bloco"),
):
    """Baixa a transcrição pronta. Pode ser chamado quantas vezes quiser.

    As legendas (srt, vtt) são montadas na hora a partir do instante de cada
    palavra, então dá para baixar o mesmo job em tamanhos de legenda diferentes.
    """
    job = manager.find(job_id)
    if job is None:
        raise HTTPException(404, "job não encontrado ou expirado")
    if job.status == "error":
        raise HTTPException(409, f"job terminou em erro: {job.error}")
    if job.status != "done":
        raise HTTPException(409, f"job ainda não concluído (status: {job.status})")

    result = job.load_result()
    if result is None:
        raise HTTPException(410, "resultado não está mais disponível")

    formato = formato.lower()
    if formato not in MEDIA_TYPES:
        raise HTTPException(400, f"formato inválido; use um de: {', '.join(MEDIA_TYPES)}")

    base = os.path.splitext(job.filename or job.id)[0]
    if formato == "json":
        body = json.dumps(result, ensure_ascii=False, indent=2)
    elif formato in ("srt", "vtt"):
        limits = {k: v for k, v in (("max_line", max_caracteres), ("max_lines", max_linhas),
                                    ("max_duration", max_segundos)) if v is not None}
        body = RENDERERS[formato](result, **limits)
    else:
        body = RENDERERS[formato](result)
    return Response(
        content=body,
        media_type=MEDIA_TYPES[formato],
        headers={"Content-Disposition": f'attachment; filename="{base}.{formato}"'},
    )


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    if not manager.remove(job_id):
        raise HTTPException(404, "job não encontrado ou expirado")
    return {"deleted": job_id}


@app.get("/", response_class=PlainTextResponse)
def index() -> str:
    return (
        f"Transcritor API (motor: {ENGINE})\n\n"
        "POST   /transcribe           multipart, campo 'file' -> JSON com o instante de cada palavra\n"
        "POST   /jobs                 igual, porem assincrono -> devolve job_id (sha256)\n"
        "GET    /jobs/{id}            status, progresso e ETA\n"
        "GET    /jobs/{id}/download   baixa o resultado (formato=json|txt|srt|vtt)\n"
        "                             legendas: max_caracteres=42 max_linhas=2 max_segundos=7\n"
        "GET    /jobs                 lista os jobs\n"
        "DELETE /jobs/{id}            remove um job e seus arquivos\n"
        "GET    /health               status do servico\n"
        "GET    /docs                 interface interativa\n"
    )
