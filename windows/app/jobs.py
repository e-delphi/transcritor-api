# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Fila de transcrições persistida em disco, com deduplicação por conteúdo.

O identificador do job é o SHA-256 do áudio combinado com os parâmetros que
influenciam o resultado. Assim reenviar o mesmo arquivo com as mesmas opções
devolve a transcrição pronta na hora, sem reprocessar, enquanto mudar
`num_speakers` ou o idioma gera um job novo — se o id fosse apenas o hash do
arquivo, o cache responderia com um resultado calculado sob outros parâmetros.

Tudo vive em disco: os jobs sobrevivem a reinício do container, e uma transcrição
interrompida no meio volta para a fila em vez de se perder. Áudio e resultado são
apagados juntos quando o job passa da validade.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

STORE_DIR = os.getenv("JOBS_DIR", "/data/jobs")
JOB_TTL_SECONDS = float(os.getenv("JOB_TTL_SECONDS", str(6 * 3600)))
CLEANUP_INTERVAL = float(os.getenv("JOB_CLEANUP_INTERVAL", "600"))

STAGE_LABELS = {
    "queued": "na fila",
    "decoding": "decodificando o áudio",
    "transcribing": "transcrevendo",
    "aligning": "alinhando palavras",
    "diarizing": "separando falantes",
    "done": "concluído",
    "error": "erro",
}

# Abaixo deste progresso a extrapolação do tempo restante não é confiável.
MIN_PROGRESS_FOR_ETA = 0.10
# Gravar a cada atualização de progresso castigaria o disco sem necessidade.
META_FLUSH_SECONDS = 5.0


def compute_job_id(audio_path: str, options: Dict[str, Any]) -> Tuple[str, str]:
    """SHA-256 do conteúdo do áudio somado aos parâmetros que mudam o resultado.

    Devolve (id do job, sha256 puro do arquivo).
    """
    digest = hashlib.sha256()
    with open(audio_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    file_sha = digest.hexdigest()

    canonical = json.dumps(options, sort_keys=True, ensure_ascii=False)
    combined = hashlib.sha256(f"{file_sha}|{canonical}".encode()).hexdigest()
    return combined, file_sha


class Job:
    def __init__(self, job_id: str, filename: Optional[str], file_sha256: str, options: dict):
        self.id = job_id
        self.filename = filename
        self.file_sha256 = file_sha256
        self.options = options
        self.status = "queued"
        self.stage = "queued"
        self.progress = 0.0
        self.created_at = time.time()
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None
        self.error: Optional[str] = None
        self._last_flush = 0.0

    @property
    def dir(self) -> str:
        return os.path.join(STORE_DIR, self.id)

    @property
    def result_path(self) -> str:
        return os.path.join(self.dir, "result.json")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.dir, "meta.json")

    def audio_path(self) -> Optional[str]:
        for name in os.listdir(self.dir):
            if name.startswith("audio"):
                return os.path.join(self.dir, name)
        return None

    def expires_at(self) -> float:
        return self.created_at + JOB_TTL_SECONDS

    def load_result(self) -> Optional[dict]:
        if not os.path.exists(self.result_path):
            return None
        with open(self.result_path, encoding="utf-8") as handle:
            return json.load(handle)

    def snapshot(self, include_result: bool = False) -> Dict[str, Any]:
        now = time.time()
        elapsed = (self.finished_at or now) - (self.started_at or self.created_at)
        data: Dict[str, Any] = {
            "job_id": self.id,
            "file_sha256": self.file_sha256,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, self.stage),
            "progress": round(self.progress, 4),
            "progress_percent": round(self.progress * 100, 1),
            "elapsed_seconds": round(max(elapsed, 0.0), 1),
            "expires_at": round(self.expires_at()),
            "expires_in_seconds": round(max(self.expires_at() - now, 0.0)),
        }
        if self.status == "running" and MIN_PROGRESS_FOR_ETA < self.progress < 1.0:
            data["eta_seconds"] = round(elapsed * (1 - self.progress) / self.progress, 1)
        if self.status == "done":
            data["download_url"] = f"/jobs/{self.id}/download"
            if include_result:
                data["result"] = self.load_result()
        if self.status == "error":
            data["error"] = self.error
        return data

    def to_meta(self) -> dict:
        return {
            "job_id": self.id,
            "file_sha256": self.file_sha256,
            "filename": self.filename,
            "options": self.options,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "Job":
        job = cls(meta["job_id"], meta.get("filename"), meta.get("file_sha256", ""), meta.get("options", {}))
        job.status = meta.get("status", "queued")
        job.stage = meta.get("stage", "queued")
        job.progress = meta.get("progress", 0.0)
        job.created_at = meta.get("created_at", time.time())
        job.started_at = meta.get("started_at")
        job.finished_at = meta.get("finished_at")
        job.error = meta.get("error")
        return job

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < META_FLUSH_SECONDS:
            return
        self._last_flush = now
        os.makedirs(self.dir, exist_ok=True)
        tmp = f"{self.meta_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(self.to_meta(), handle, ensure_ascii=False)
        os.replace(tmp, self.meta_path)


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # Um worker só: os modelos são serializados de qualquer forma, então mais
        # threads apenas criariam disputa sem ganho de throughput.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")
        self._worker: Optional[Callable[[Job], dict]] = None
        os.makedirs(STORE_DIR, exist_ok=True)

    def set_worker(self, worker: Callable[[Job], dict]) -> None:
        """Define a função que executa o trabalho; usada também ao reenfileirar."""
        self._worker = worker

    # ------------------------------------------------------------------ carga --
    def restore(self) -> None:
        """Recarrega os jobs do disco e reenfileira o que ficou pela metade."""
        if not os.path.isdir(STORE_DIR):
            return
        restored = requeued = 0
        for job_id in sorted(os.listdir(STORE_DIR)):
            meta_path = os.path.join(STORE_DIR, job_id, "meta.json")
            if not os.path.exists(meta_path):
                continue
            try:
                with open(meta_path, encoding="utf-8") as handle:
                    job = Job.from_meta(json.load(handle))
            except (OSError, ValueError, KeyError):
                logger.warning("Job %s tem metadados ilegíveis; descartando", job_id)
                shutil.rmtree(os.path.join(STORE_DIR, job_id), ignore_errors=True)
                continue

            with self._lock:
                self._jobs[job.id] = job
            restored += 1

            # Reiniciar o container mata o worker: o que estava rodando volta à fila
            # em vez de ficar preso em "running" para sempre.
            if job.status in ("running", "queued") and job.audio_path():
                job.status = "queued"
                job.stage = "queued"
                job.progress = 0.0
                job.flush(force=True)
                self._pool.submit(self._run, job)
                requeued += 1

        self._evict_expired()
        if restored:
            logger.info("Restaurados %d job(s) do disco (%d reenfileirado[s])", restored, requeued)

    # --------------------------------------------------------------- execução --
    def find(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def submit(self, job: Job) -> Job:
        with self._lock:
            self._jobs[job.id] = job
        job.flush(force=True)
        self._pool.submit(self._run, job)
        return job

    def _run(self, job: Job) -> None:
        if self._worker is None:
            job.status, job.stage, job.error = "error", "error", "worker não configurado"
            job.flush(force=True)
            return

        job.status = "running"
        job.started_at = time.time()
        job.flush(force=True)
        try:
            result = self._worker(job)
            tmp = f"{job.result_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False)
            os.replace(tmp, job.result_path)
            job.status, job.stage, job.progress = "done", "done", 1.0
        except Exception as exc:  # noqa: BLE001 - a falha vai para o cliente
            logger.exception("Job %s falhou", job.id)
            job.status, job.stage, job.error = "error", "error", str(exc)
        finally:
            job.finished_at = time.time()
            job.flush(force=True)

    def update_progress(self, job: Job, stage: str, fraction: float) -> None:
        job.stage = stage
        job.progress = fraction
        job.flush()

    # ----------------------------------------------------------------- listagem --
    def list(self) -> List[Dict[str, Any]]:
        self._evict_expired()
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.snapshot() for job in jobs]

    def remove(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.pop(job_id, None)
        if job is None:
            return False
        shutil.rmtree(job.dir, ignore_errors=True)
        return True

    # ------------------------------------------------------------------ limpeza --
    def _evict_expired(self) -> None:
        now = time.time()
        with self._lock:
            expired = [j for j in self._jobs.values() if j.expires_at() < now and j.status != "running"]
            for job in expired:
                del self._jobs[job.id]
        for job in expired:
            # Áudio e resultado saem juntos: o diretório inteiro do job é removido.
            shutil.rmtree(job.dir, ignore_errors=True)
        if expired:
            logger.info("Removidos %d job(s) vencido(s) com seus arquivos", len(expired))

    def start_cleanup_loop(self) -> None:
        def loop() -> None:
            while True:
                time.sleep(CLEANUP_INTERVAL)
                try:
                    self._evict_expired()
                except Exception:  # noqa: BLE001 - a limpeza nunca derruba o serviço
                    logger.exception("Falha na limpeza periódica")

        threading.Thread(target=loop, daemon=True, name="job-cleanup").start()


manager = JobManager()
