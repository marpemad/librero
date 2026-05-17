"""
MessageRouter — clasifica el mensaje entrante en un IngestionPayload.

NO conoce los extractores concretos. Solo identifica el tipo (TEXT/URL/FILE).
La selección de la estrategia la hace el ExtractorFactory.
"""
from __future__ import annotations

from pathlib import Path

from src.extractors import IngestionPayload, SourceKind
from src.extractors._helpers import find_first_url
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MessageRouter:
    @staticmethod
    def from_text(text: str) -> IngestionPayload:
        url = find_first_url(text)
        if url:
            logger.info("Router: detectada URL → %s", url)
            return IngestionPayload(kind=SourceKind.URL, raw=url, metadata={"original_text": text})
        logger.info("Router: texto sin URL → CONCEPTO")
        return IngestionPayload(kind=SourceKind.TEXT, raw=text)

    @staticmethod
    def from_file(local_path: Path, mime: str | None, filename: str) -> IngestionPayload:
        logger.info("Router: archivo %s (mime=%s)", filename, mime)
        return IngestionPayload(
            kind=SourceKind.FILE,
            raw=str(local_path),
            metadata={"mime": mime, "filename": filename},
        )
