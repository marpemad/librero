"""
Caché simple en SQLite para resultados de extracciones costosas.

Usado por:
  - whisper_runner: hash del archivo de audio → texto transcrito
  - document_extractor: hash del PDF → texto parseado

Diseño:
  - Una sola tabla `extractions(key TEXT PRIMARY KEY, value TEXT, created_at TEXT)`.
  - `key` se construye como `<scope>:<hash>` para namespacing (ej. "whisper:abcd1234").
  - Las operaciones síncronas se envuelven en `asyncio.to_thread` cuando se llaman desde el bot.

Ventajas para el caso de uso del usuario:
  - Si un audio de 2h se procesa, queda cacheado. Si vuelves a mandar la misma URL/archivo,
    se salta la transcripción (que es lo que más duele perder).
  - SQLite es file-based, no requiere servicio extra, y es perfecto para un solo usuario.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExtractionCache:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or settings.cache_db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self._path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS extractions (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
                """
            )
            con.commit()

    def get(self, scope: str, key: str) -> Optional[str]:
        full = f"{scope}:{key}"
        with sqlite3.connect(self._path) as con:
            row = con.execute(
                "SELECT value FROM extractions WHERE key = ?", (full,)
            ).fetchone()
        if row:
            logger.info("Cache HIT  [%s] %s", scope, key[:12])
            return row[0]
        logger.info("Cache MISS [%s] %s", scope, key[:12])
        return None

    def set(self, scope: str, key: str, value: str) -> None:
        full = f"{scope}:{key}"
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self._path) as con:
            con.execute(
                "INSERT OR REPLACE INTO extractions(key, value, created_at) VALUES (?, ?, ?)",
                (full, value, now),
            )
            con.commit()
        logger.info("Cache SET  [%s] %s (%d chars)", scope, key[:12], len(value))


# --- Helpers de hashing ---
def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA1 streaming — funciona con audios y PDFs grandes sin cargar todo en RAM."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def hash_text(text: str) -> str:
    """Para URLs o cadenas (cachear por URL canónica)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# Singleton del proceso
cache = ExtractionCache()
