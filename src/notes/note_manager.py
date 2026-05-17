"""NoteManager — gestor de listas curadas persistidas como Markdown en el Vault.

Cada "tipo" de lista (libros, series, películas…) vive en un fichero propio
dentro de la subcarpeta de listas del Vault:

    Vault/
      Listas/
        Libros.md
        Series.md
        Películas.md
        ...

Formato del fichero:

    ---
    updated: "2026-05-03T10:00:00"
    type: "list_note"
    kind: "libros"
    ---

    # 📚 Libros

    ## Lista (3 elementos)

    - Isaac Asimov _(2026-05-03)_
    - Marcos Vázquez _(2026-05-03)_
    - Arthur C. Clarke _(2026-05-04)_

El formato es legible y editable a mano en Obsidian.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Tipos conocidos: canonical → {emoji, aliases} ─────────────────────────────

_KIND_CONFIG: dict[str, dict] = {
    "libros":     {"emoji": "📚", "aliases": ["libro", "book", "books", "lectura"]},
    "series":     {"emoji": "📺", "aliases": ["serie", "tv", "show", "shows", "series"]},
    "películas":  {"emoji": "🎬", "aliases": [
        "pelicula", "peliculas", "película", "movie", "movies", "film", "films", "cine",
    ]},
    "podcasts":   {"emoji": "🎙", "aliases": ["podcast"]},
    "música":     {"emoji": "🎵", "aliases": [
        "musica", "música", "music", "album", "albums", "artista", "artistas",
    ]},
    "juegos":     {"emoji": "🎮", "aliases": ["juego", "juegos", "game", "games", "videojuego"]},
    "artículos":  {"emoji": "📰", "aliases": [
        "articulo", "articulos", "artículo", "article", "articles",
    ]},
    "personas":   {"emoji": "👤", "aliases": ["persona", "people", "person", "autor", "autores"]},
    "lugares":    {"emoji": "📍", "aliases": ["lugar", "lugares", "place", "places", "viaje"]},
    "cursos":     {"emoji": "🎓", "aliases": ["curso", "course", "courses", "formacion"]},
    "otros":      {"emoji": "📌", "aliases": ["otro", "otras", "other", "misc", "varios"]},
}

# Índice inverso alias → canonical (case-insensitive)
_ALIAS_TO_KIND: dict[str, str] = {}
for _kind, _cfg in _KIND_CONFIG.items():
    _ALIAS_TO_KIND[_kind] = _kind
    for _alias in _cfg["aliases"]:
        _ALIAS_TO_KIND[_alias.lower()] = _kind


def normalize_kind(raw: str) -> str:
    """Normaliza a canonical kind. Si no está en el catálogo, lo devuelve en
    minúsculas (tipos ad-hoc son válidos)."""
    key = raw.strip().lower()
    return _ALIAS_TO_KIND.get(key, key)


def kind_emoji(kind: str) -> str:
    """Devuelve el emoji asociado al tipo, o 📌 si es un tipo ad-hoc."""
    return _KIND_CONFIG.get(kind, {}).get("emoji", "📌")


def kind_title(kind: str) -> str:
    """Título para mostrar: primera letra en mayúscula."""
    return kind[0].upper() + kind[1:] if kind else "Lista"


# ── Modelo de ítem ─────────────────────────────────────────────────────────────

@dataclass
class ListItem:
    text: str
    added_at: datetime = field(default_factory=datetime.now)

    def to_md_line(self) -> str:
        date_str = self.added_at.strftime("%Y-%m-%d")
        return f"- {self.text} _({date_str})_"


# ── Parsers ────────────────────────────────────────────────────────────────────

_ITEM_RE = re.compile(
    r"^-\s+(.+?)(?:\s+_\((\d{4}-\d{2}-\d{2})\)_)?\s*$"
)


# ── Clase principal ────────────────────────────────────────────────────────────

class NoteManager:
    """Gestiona listas curadas persistidas como Markdown en el Vault."""

    def __init__(self, base_path: Path) -> None:
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    # ── Rutas ────────────────────────────────────────────────────────────────

    def _note_path(self, kind: str) -> Path:
        """Ruta del fichero .md para un tipo dado (ej. "libros" → Libros.md)."""
        stem = kind_title(kind)
        return self._base / f"{stem}.md"

    def note_path_for(self, kind: str) -> Path:
        return self._note_path(normalize_kind(kind))

    # ── API pública ──────────────────────────────────────────────────────────

    def add_items(self, kind: str, items: list[str]) -> tuple[list[str], bool]:
        """Añade ítems a la lista dada.

        Returns:
            (items_añadidos, fue_creada_la_nota)
        """
        kind = normalize_kind(kind)
        path = self._note_path(kind)
        was_created = not path.exists()
        existing = self._load_items(path)
        existing_lower = {i.text.lower() for i in existing}

        added: list[str] = []
        for raw in items:
            text = raw.strip().rstrip(".")
            if not text:
                continue
            if text.lower() not in existing_lower:
                existing.append(ListItem(text=text))
                existing_lower.add(text.lower())
                added.append(text)

        if added:
            self._save(path, kind, existing)
            logger.info(
                "%s '%s': +%d ítem(s) [total=%d]",
                "Creada" if was_created else "Actualizada",
                kind,
                len(added),
                len(existing),
            )
        return added, was_created

    def list_items(self, kind: str) -> list[ListItem]:
        """Devuelve todos los ítems de la lista."""
        kind = normalize_kind(kind)
        return self._load_items(self._note_path(kind))

    def remove_item(self, kind: str, query: str) -> Optional[str]:
        """Elimina el primer ítem cuyo texto contenga `query` (insensible a mayús).

        Returns:
            El texto del ítem eliminado, o None si no se encontró.
        """
        kind = normalize_kind(kind)
        path = self._note_path(kind)
        items = self._load_items(path)
        q = query.strip().lower()
        to_remove = next((i for i in items if q in i.text.lower()), None)
        if to_remove is None:
            return None
        items.remove(to_remove)
        self._save(path, kind, items)
        logger.info("Eliminado de '%s': %s", kind, to_remove.text)
        return to_remove.text

    def list_kinds(self) -> list[tuple[str, int]]:
        """Devuelve [(kind, n_items), …] de todas las listas existentes en disco."""
        result: list[tuple[str, int]] = []
        for f in sorted(self._base.glob("*.md")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            if 'type: "list_note"' not in text:
                continue
            m = re.search(r'^kind:\s*["\']?([^"\'|\n]+)["\']?', text, re.MULTILINE)
            kind = m.group(1).strip() if m else f.stem.lower()
            items = self._load_items(f)
            result.append((kind, len(items)))
        return result

    # ── Render para Telegram ─────────────────────────────────────────────────

    def render_telegram(self, kind: str, max_items: int = 40) -> str:
        """Texto formateado (Markdown) para mostrar la lista en Telegram."""
        kind = normalize_kind(kind)
        items = self._load_items(self._note_path(kind))
        emoji = kind_emoji(kind)
        title = kind_title(kind)
        n = len(items)

        if not items:
            return (
                f"{emoji} *{title}*\n\n"
                f"_Lista vacía._\n"
                f"Añade con: `/note {kind} <ítem1>, <ítem2>`"
            )

        lines = [f"{emoji} *{title}* ({n} elemento{'s' if n != 1 else ''})\n"]
        for i, item in enumerate(items[:max_items], start=1):
            date_str = item.added_at.strftime("%d/%m/%y")
            lines.append(f"  `{i}.` {item.text} _({date_str})_")
        if n > max_items:
            lines.append(f"\n  _… y {n - max_items} más (ver nota en Obsidian)_")
        lines.append(f"\n_Borrar:_ `/notedel {kind} <nombre_parcial>`")
        return "\n".join(lines)

    # ── I/O interno ──────────────────────────────────────────────────────────

    def _load_items(self, path: Path) -> list[ListItem]:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        items: list[ListItem] = []
        in_list = False
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^##\s+Lista", stripped):
                in_list = True
                continue
            if in_list and stripped.startswith("## "):
                break  # siguiente sección
            if not in_list:
                continue
            m = _ITEM_RE.match(stripped)
            if not m:
                continue
            item_text = m.group(1).strip()
            date_raw = m.group(2)
            added_at = datetime.now()
            if date_raw:
                try:
                    added_at = datetime.fromisoformat(date_raw)
                except ValueError:
                    pass
            items.append(ListItem(text=item_text, added_at=added_at))
        return items

    def _save(self, path: Path, kind: str, items: list[ListItem]) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        emoji = kind_emoji(kind)
        title = kind_title(kind)
        n = len(items)
        lines = [
            "---",
            f'updated: "{now}"',
            'type: "list_note"',
            f'kind: "{kind}"',
            "---",
            "",
            f"# {emoji} {title}",
            "",
            f"## Lista ({n} elemento{'s' if n != 1 else ''})",
            "",
        ]
        for item in items:
            lines.append(item.to_md_line())
        if not items:
            lines.append("_(vacía)_")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
