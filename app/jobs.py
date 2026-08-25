# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Eduardo

"""Fila de transcrições em segundo plano com acompanhamento de progresso.

Áudios longos levam mais tempo que o razoável para uma requisição HTTP síncrona
(com `large-v3` a transcrição roda a ~1,5x o tempo real), então o cliente cria um
job, recebe um identificador na hora e consulta o andamento quando quiser.

O estado vive em memória: reiniciar o container descarta os jobs. Para várias
réplicas seria preciso um armazenamento compartilhado.
"""

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Jobs concluídos somem depois disto, para a memória não crescer sem limite.
JOB_TTL_SECONDS = float(os.getenv("JOB_TTL_SECONDS", "3600"))
MAX_JOBS = int(os.getenv("MAX_JOBS", "200"))
# Abaixo deste progresso a extrapolação do tempo restante não é confiável.
MIN_PROGRESS_FOR_ETA = 0.10

STAGE_LABELS = {
    "queued": "na fila",
    "decoding": "decodificando o áudio",
    "transcribing": "transcrevendo",
    "diarizing": "separando falantes",
    "done": "concluído",
    "error": "erro",
}


@dataclass
class Job:
    id: str
    filename: Optional[str]
    status: str = "queued"  # queued | running | done | error
    stage: str = "queued"
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        elapsed = (self.finished_at or now) - (self.started_at or self.created_at)
        data: Dict[str, Any] = {
            "job_id": self.id,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "stage_label": STAGE_LABELS.get(self.stage, self.stage),
            "progress": round(self.progress, 4),
            "progress_percent": round(self.progress * 100, 1),
            "elapsed_seconds": round(max(elapsed, 0.0), 1),
        }
        # Estimativa de tempo restante. Exige progresso já significativo: extrapolar
        # a partir dos primeiros por cento produz números sem sentido.
        if self.status == "running" and MIN_PROGRESS_FOR_ETA < self.progress < 1.0:
            data["eta_seconds"] = round(elapsed * (1 - self.progress) / self.progress, 1)
        if self.status == "done":
            data["result"] = self.result
        if self.status == "error":
            data["error"] = self.error
        return data


class JobManager:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        # Um worker só: os modelos já são serializados por lock, então mais threads
        # apenas criariam disputa sem ganho de throughput.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="job")

    def submit(self, filename: Optional[str], work: Callable[[Job], dict]) -> Job:
        self._evict_expired()
        job = Job(id=uuid.uuid4().hex[:16], filename=filename)
        with self._lock:
            if len(self._jobs) >= MAX_JOBS:
                raise RuntimeError(f"limite de {MAX_JOBS} jobs atingido; consulte ou remova os antigos")
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, work)
        return job

    def _run(self, job: Job, work: Callable[[Job], dict]) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            job.result = work(job)
            job.status = "done"
            job.stage = "done"
            job.progress = 1.0
        except Exception as exc:  # noqa: BLE001 - a falha vai para o cliente
            logger.exception("Job %s falhou", job.id)
            job.status = "error"
            job.stage = "error"
            job.error = str(exc)
        finally:
            job.finished_at = time.time()

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        out = []
        for job in jobs:
            item = job.snapshot()
            item.pop("result", None)  # a listagem não carrega transcrições inteiras
            out.append(item)
        return out

    def remove(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None

    def _evict_expired(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        with self._lock:
            expired = [
                jid
                for jid, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
            for jid in expired:
                del self._jobs[jid]
        if expired:
            logger.info("Removidos %d job(s) expirado(s)", len(expired))


manager = JobManager()
