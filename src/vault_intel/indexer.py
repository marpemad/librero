"""Índice semántico del Vault: escanea notas Markdown y guarda embeddings en SQLite.

Uso:
    idx = VaultIndexer()
    idx.rebuild()           # escaneo incremental al arrancar
    idx.index_note(path)    # re-indexar una nota recién escrita
    results = idx.search(embedding, top_k=5)
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from src.config.settings import settings
from src.utils.logger import get_logger
from .embedder import create_embedder

logger = get_logger(__name__)

_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)
_TAG_RE = re.compile(r"#([\w/\-áéíóúüñÁÉÍÓÚÜÑ]+)")


class VaultIndexer:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or settings.vault_index_db_path
        self._embedder = create_embedder()
        self._conn = self._open_db()

    # ── Setup ──────────────────────────────────────────────────────────────────

    def _open_db(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                path      TEXT PRIMARY KEY,
                mtime     REAL    NOT NULL,
                title     TEXT    NOT NULL,
                tags      TEXT    NOT NULL DEFAULT '[]',
                embedding BLOB    NOT NULL
            )
        """)
        conn.commit()
        return conn

    # ── API pública ────────────────────────────────────────────────────────────

    def rebuild(self) -> int:
        """Escaneo incremental del Vault. Solo re-embeda notas nuevas o modificadas."""
        vault = settings.obsidian_vault_path
        md_files = list(vault.rglob("*.md"))
        updated = 0

        for path in md_files:
            try:
                mtime = path.stat().st_mtime
                row = self._conn.execute(
                    "SELECT mtime FROM notes WHERE path=?", (str(path),)
                ).fetchone()
                if row and abs(row[0] - mtime) < 0.01:
                    continue  # sin cambios
                self._index_note(path, mtime)
                updated += 1
            except Exception as exc:
                logger.warning("Error indexando %s: %s", path.name, exc)

        # Borrar notas eliminadas del Vault
        existing = {str(p) for p in md_files}
        stored = {r[0] for r in self._conn.execute("SELECT path FROM notes")}
        for removed in stored - existing:
            self._conn.execute("DELETE FROM notes WHERE path=?", (removed,))

        self._conn.commit()
        total = self.note_count()
        logger.info("Vault indexado: %d notas totales, %d actualizadas", total, updated)
        return updated

    def index_note(self, path: Path) -> None:
        """Indexa o re-indexa una sola nota (llamar tras escribir en el Vault)."""
        try:
            mtime = path.stat().st_mtime
            self._index_note(path, mtime)
            self._conn.commit()
            logger.debug("Nota re-indexada: %s", path.name)
        except Exception as exc:
            logger.warning("No se pudo indexar %s: %s", path.name, exc)

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[float, str, str, list[str]]]:
        """Búsqueda coseno. Devuelve lista de (similitud, path, title, tags)."""
        rows = self._conn.execute(
            "SELECT path, title, tags, embedding FROM notes"
        ).fetchall()

        results: list[tuple[float, str, str, list[str]]] = []
        for path, title, tags_json, emb_bytes in rows:
            emb = np.frombuffer(emb_bytes, dtype=np.float32).copy()
            sim = float(np.dot(query_embedding, emb))
            results.append((sim, path, title, json.loads(tags_json)))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    def embed_text(self, text: str) -> np.ndarray:
        return self._embedder.embed(text)

    def get_all_tags(self) -> list[str]:
        """Tags únicos del Vault ordenados por frecuencia (máx. 50)."""
        counter: Counter[str] = Counter()
        for (tags_json,) in self._conn.execute("SELECT tags FROM notes"):
            counter.update(json.loads(tags_json))
        return [tag for tag, _ in counter.most_common(50)]

    def note_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM notes").fetchone()[0]

    # ── Internos ───────────────────────────────────────────────────────────────

    def _index_note(self, path: Path, mtime: float) -> None:
        text = path.read_text(encoding="utf-8", errors="ignore")
        title = self._extract_title(text, path)
        tags = _TAG_RE.findall(text)
        snippet = f"{title}. {self._extract_summary(text)}"
        embedding = self._embedder.embed(snippet)
        self._conn.execute(
            "INSERT OR REPLACE INTO notes (path, mtime, title, tags, embedding) "
            "VALUES (?, ?, ?, ?, ?)",
            (str(path), mtime, title, json.dumps(tags), embedding.tobytes()),
        )

    @staticmethod
    def _extract_title(text: str, path: Path) -> str:
        m = _H1_RE.search(text)
        if m:
            return m.group(1).strip()
        return path.stem

    @staticmethod
    def _extract_summary(text: str) -> str:
        """Extrae ## Resumen si existe; si no, primeros 300 chars del cuerpo."""
        idx = text.find("## Resumen")
        if idx != -1:
            after = text[idx + 10:].strip()
            end = after.find("\n##")
            return (after[:end] if end != -1 else after)[:400].strip()
        # Quitar frontmatter YAML
        if text.startswith("---"):
            end = text.find("---", 3)
            if end != -1:
                text = text[end + 3:]
        return text.strip()[:300]
