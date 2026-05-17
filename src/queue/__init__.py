"""Cola de trabajos serializada — evita saturar el LLM local con jobs concurrentes."""
from .job_queue import Job, JobQueue, JobStatus

__all__ = ["Job", "JobQueue", "JobStatus"]
