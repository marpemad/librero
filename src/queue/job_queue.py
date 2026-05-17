"""
JobQueue — cola FIFO con un único worker.

Por qué: el LLM local (qwen3.6:27b en Ollama) usa toda la RAM/GPU del Mac.
Dos síntesis en paralelo = OOM, swap, 5x más lentas, o cuelgue. Con esta cola
garantizamos que solo se procesa una a la vez, y los demás jobs esperan en orden
con feedback al usuario.

Diseño:
  - Una `asyncio.Queue` y un único `worker` task que la consume.
  - `enqueue(job)`  → encola y devuelve la posición que ocupa el job.
  - `cancel(job_id)`→ marca un job PENDING como cancelado (no aborta el RUNNING).
  - `pending_jobs()`/`current_job()` para mostrar `/queue`.
  - El handler que procesa cada job lo inyecta `start(handler)` (closure con
    acceso al bot de Telegram, etc.) para no acoplar la cola al bot.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    title: str                       # mostrado al usuario, p.ej. "Trabajo Profundo (epub)"
    chat_id: int
    user_id: int
    progress_msg_id: int             # id del mensaje de progreso en Telegram (para editar)
    payload: Any                     # lo que el handler necesite (dict con content, plan, etc.)
    enqueued_at: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def short_id(self) -> str:
        return self.id[:6]

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        end = self.finished_at or time.time()
        return end - self.started_at


JobHandler = Callable[[Job], Awaitable[None]]


def new_job_id() -> str:
    return uuid.uuid4().hex[:10]


class JobQueue:
    """Cola FIFO con un único worker."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Job] = asyncio.Queue()
        # Registro de TODOS los jobs vistos (PENDING/RUNNING/DONE/FAILED/CANCELLED).
        # Permite mostrar histórico breve y resolver IDs en /cancel.
        self._jobs: dict[str, Job] = {}
        self._current: Optional[Job] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._handler: Optional[JobHandler] = None

    # ------------ API pública ------------

    def start(self, handler: JobHandler) -> None:
        """Lanza el worker en background. Llamar UNA sola vez (en post_init)."""
        if self._worker_task is not None:
            logger.warning("JobQueue.start() llamado dos veces — ignorando.")
            return
        self._handler = handler
        loop = asyncio.get_event_loop()
        self._worker_task = loop.create_task(self._worker(), name="job-queue-worker")
        logger.info("🧵 JobQueue worker iniciado")

    async def enqueue(self, job: Job) -> int:
        """Encola y devuelve la posición 1-indexed (1 = el siguiente en ejecutarse)."""
        self._jobs[job.id] = job
        await self._queue.put(job)
        return self.position_of(job.id)

    def position_of(self, job_id: str) -> int:
        """1 = el siguiente en ejecutarse. 0 si ya terminó / no existe / está corriendo."""
        job = self._jobs.get(job_id)
        if not job or job.status != JobStatus.PENDING:
            return 0
        # Posición = número de PENDINGs anteriores (por enqueued_at) + 1
        # + 1 más si hay un job RUNNING (porque el RUNNING también va por delante)
        pending_before = sum(
            1 for j in self._jobs.values()
            if j.status == JobStatus.PENDING and j.enqueued_at < job.enqueued_at
        )
        running_offset = 1 if self._current and self._current.status == JobStatus.RUNNING else 0
        return pending_before + 1 + running_offset

    def pending_jobs(self) -> list[Job]:
        """Lista de jobs PENDING en orden cronológico de encolado."""
        return sorted(
            (j for j in self._jobs.values() if j.status == JobStatus.PENDING),
            key=lambda j: j.enqueued_at,
        )

    def current_job(self) -> Optional[Job]:
        """Job RUNNING actualmente, o None."""
        if self._current and self._current.status == JobStatus.RUNNING:
            return self._current
        return None

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        """Cancela un job PENDING. NO aborta el RUNNING (peligroso a medio LLM).
        Devuelve True si la cancelación se aplicó.
        """
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status != JobStatus.PENDING:
            return False
        job.status = JobStatus.CANCELLED
        logger.info("⏹️  Job %s cancelado por el usuario", job.short_id())
        return True

    def stats(self) -> dict[str, int]:
        out = {s.value: 0 for s in JobStatus}
        for j in self._jobs.values():
            out[j.status.value] += 1
        return out

    # ------------ Worker interno ------------

    async def _worker(self) -> None:
        assert self._handler is not None, "Llama a start(handler) antes de encolar"
        while True:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                logger.info("Worker cancelado.")
                break

            try:
                # Si fue cancelado mientras esperaba, lo saltamos.
                if job.status == JobStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                self._current = job
                job.status = JobStatus.RUNNING
                job.started_at = time.time()
                logger.info("▶️  Job %s iniciado (%s)", job.short_id(), job.title)

                try:
                    await self._handler(job)
                    if job.status == JobStatus.RUNNING:
                        job.status = JobStatus.DONE
                except Exception as exc:
                    logger.exception("❌ Job %s falló", job.short_id())
                    job.status = JobStatus.FAILED
                    job.error = f"{type(exc).__name__}: {exc}"
                finally:
                    job.finished_at = time.time()
                    logger.info(
                        "⏹  Job %s terminado en %.1fs [%s]",
                        job.short_id(), job.elapsed, job.status.value,
                    )
                    self._current = None
                    self._queue.task_done()
            except Exception:
                # Defensa total: el worker NUNCA debe morir
                logger.exception("Error inesperado en worker (sigo vivo)")
