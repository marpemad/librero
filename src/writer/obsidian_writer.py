"""ObsidianWriter — guarda la nota con frontmatter en el Vault.

El frontmatter ahora es rico y se construye automáticamente desde
`ExtractedContent.extra`, volcando todo lo que hayan extraído los
extractores (autor, editorial, ISBN, fecha, canal, duración, etc.).

Cada nota recibe un `id` corto y estable (8 hex), que sobrevive a edits
y rename del fichero (siempre que el frontmatter se respete).
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from slugify import slugify

from src.config.settings import settings
from src.extractors import ExtractedContent
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _new_note_id() -> str:
    """ID corto único por nota (8 chars hex de un UUID v4)."""
    return uuid.uuid4().hex[:8]


# Mapeo: clave de extra → clave de YAML en el frontmatter.
# El orden importa; los campos aparecerán en el frontmatter en este orden.
_EXTRA_TO_YAML: list[tuple[str, str]] = [
    ("doc_kind",          "kind"),            # book / paper / video / podcast / article / web / image / document
    ("book_title",        "original_title"),
    ("doc_title",         "original_title"),
    ("author",            "author"),
    ("authors",           "authors"),
    ("publisher",         "publisher"),
    ("year",              "year"),
    ("published",         "published"),
    ("isbn",              "isbn"),
    ("doi",               "doi"),
    ("original_language", "original_language"),
    ("site_name",         "site"),
    ("url",               "url"),
    ("channel_url",       "channel"),
    ("video_id",          "video_id"),
    ("duration",          "duration"),
    ("view_count",        "views"),
    ("abstract",          "abstract"),
    ("description",       "description"),
    ("chapters",          "chapters"),
    ("dimensions",        "dimensions"),
    ("image_format",      "image_format"),
    ("source_tags",       "source_tags"),
    ("sources",           "sources"),
    ("query",             "query"),
]


class ObsidianWriter:
    def __init__(self, base_folder: Path | None = None) -> None:
        self._base = base_folder or settings.inbox_path
        self._base.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        title: str,
        body_md: str,
        content: ExtractedContent,
        language: Optional[str] = None,
        needs_review: bool = False,
        review_notes: Optional[list[str]] = None,
        related: Optional[list[str]] = None,
        note_id: Optional[str] = None,
    ) -> tuple[Path, str]:
        """Escribe la nota y devuelve `(path, note_id)`.

        Si no se pasa `note_id`, se genera uno nuevo (8 hex chars). El id queda
        persistido en el frontmatter y permite recuperar la nota con `/read <id>`.
        """
        if note_id is None:
            note_id = _new_note_id()

        stem = slugify(title, lowercase=False, separator=" ", max_length=120) or "Nota"
        if needs_review:
            stem = f"⚠ {stem}"
        target = self._base / f"{stem}.md"
        i = 2
        while target.exists():
            target = self._base / f"{stem} ({i}).md"
            i += 1

        frontmatter = self._build_frontmatter(
            content,
            language=language,
            needs_review=needs_review,
            review_notes=review_notes,
            related=related,
            note_id=note_id,
        )
        full = f"{frontmatter}\n\n{body_md.strip()}\n"
        target.write_text(full, encoding="utf-8")
        logger.info("✅ Nota escrita: %s (id=%s, review=%s)", target.name, note_id, needs_review)
        return target, note_id

    def overwrite(
        self,
        existing_path: Path,
        title: str,
        body_md: str,
        content: ExtractedContent,
        language: Optional[str] = None,
        needs_review: bool = False,
        review_notes: Optional[list[str]] = None,
        related: Optional[list[str]] = None,
    ) -> tuple[Path, str]:
        """Sobreescribe una nota existente preservando su note_id original.

        Si la nota no tiene id en el frontmatter, se genera uno nuevo.
        """
        existing_id: Optional[str] = None
        if existing_path.exists():
            old_text = existing_path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'^id:\s*["\']?([0-9a-f]{8})["\']?', old_text, re.MULTILINE)
            if m:
                existing_id = m.group(1)

        note_id = existing_id or _new_note_id()
        frontmatter = self._build_frontmatter(
            content,
            language=language,
            needs_review=needs_review,
            review_notes=review_notes,
            related=related,
            note_id=note_id,
        )
        full = f"{frontmatter}\n\n{body_md.strip()}\n"
        existing_path.write_text(full, encoding="utf-8")
        logger.info("♻️ Nota sobreescrita: %s (id=%s)", existing_path.name, note_id)
        return existing_path, note_id

    def append_update(
        self,
        existing_path: Path,
        new_body_md: str,
    ) -> tuple[Path, str]:
        """Añade una sección de actualización al final de una nota existente.

        Extrae el note_id del frontmatter para devolverlo. Si la nota no existe
        (raro), simplemente escribe el cuerpo limpio.
        """
        existing_id = ""
        existing_text = ""
        if existing_path.exists():
            existing_text = existing_path.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r'^id:\s*["\']?([0-9a-f]{8})["\']?', existing_text, re.MULTILINE)
            if m:
                existing_id = m.group(1)

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        section = f"\n\n---\n\n## 🔄 Actualización — {now_str}\n\n{new_body_md.strip()}\n"
        updated = (existing_text.rstrip() + section) if existing_text else new_body_md.strip() + "\n"
        existing_path.write_text(updated, encoding="utf-8")
        logger.info("🔄 Nota actualizada (append): %s", existing_path.name)
        return existing_path, existing_id

    @staticmethod
    def _build_frontmatter(
        content: ExtractedContent,
        language: Optional[str],
        needs_review: bool,
        review_notes: Optional[list[str]],
        related: Optional[list[str]] = None,
        note_id: Optional[str] = None,
    ) -> str:
        now = datetime.now().isoformat(timespec="seconds")
        ref = _yaml_escape(content.source_ref)
        lines = ["---", f'created: "{now}"']
        if note_id:
            lines.append(f'id: "{note_id}"')
        lines += [
            f'source_type: "{content.source_type}"',
            f'source_ref: "{ref}"',
        ]
        if language:
            lines.append(f'language: "{language}"')

        # Volcado automático de extra
        seen: set[str] = set()
        extra = content.extra or {}
        for src_key, yaml_key in _EXTRA_TO_YAML:
            if yaml_key in seen:
                continue
            val = extra.get(src_key)
            if val is None or val == "" or val == []:
                continue
            rendered = _render_yaml_value(yaml_key, val)
            if rendered:
                lines.extend(rendered)
                seen.add(yaml_key)

        # Notas relacionadas (Vault Intel)
        if related:
            lines.append("related:")
            for r in related:
                lines.append(f'  - "[[{_yaml_escape(r)}]]"')

        if needs_review:
            lines.append("needs_review: true")
            if review_notes:
                lines.append("review_problems:")
                for p in review_notes:
                    lines.append(f'  - "{_yaml_escape(p)}"')

        lines.append("---")
        return "\n".join(lines)


def _yaml_escape(value: str) -> str:
    """Escape para YAML entre comillas dobles: comillas y saltos de línea."""
    return str(value).replace('"', "'").replace("\n", " ").replace("\r", " ")


def _render_yaml_value(yaml_key: str, val: Any) -> list[str]:
    """Devuelve las líneas YAML para un valor — escalar, lista, número o booleano."""
    # Booleanos / enteros / floats: sin comillas
    if isinstance(val, bool):
        return [f"{yaml_key}: {'true' if val else 'false'}"]
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return [f"{yaml_key}: {val}"]
    # Listas: bloque YAML con guiones
    if isinstance(val, (list, tuple)):
        items = [v for v in val if v not in (None, "", [])]
        if not items:
            return []
        out = [f"{yaml_key}:"]
        for item in items[:30]:  # límite defensivo
            out.append(f'  - "{_yaml_escape(str(item))}"')
        return out
    # String: una sola línea (recortada si es enorme)
    s = str(val).strip()
    if not s:
        return []
    if len(s) > 800:
        s = s[:800].rstrip() + "…"
    return [f'{yaml_key}: "{_yaml_escape(s)}"']
