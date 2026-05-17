"""
Lectura analítica del Vault — alimenta /weekly y /random.

No depende de Ollama ni de embeddings. Solo lee el sistema de archivos y parsea
frontmatter de las notas Markdown que escribimos con `ObsidianWriter`.
"""
from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

# Mapeo kind → emoji (alineado con doc_kind que ponen los extractores)
KIND_EMOJI: dict[str, str] = {
    "book":     "📚",
    "paper":    "🎓",
    "video":    "📺",
    "podcast":  "🎙",
    "article":  "📰",
    "web":      "🌐",
    "image":    "🖼",
    "document": "📄",
    "concept":  "💡",
    "voice":    "🎤",
    "tweet":    "🐦",
    "sheet":    "📊",
    "pdf":      "📄",
    "epub":     "📚",
    "docx":     "📝",
    "txt":      "📝",
    "md":       "📝",
}


@dataclass
class NoteSummary:
    """Información ligera de una nota. NO carga el body entero."""
    path: Path
    mtime: datetime
    title: str
    note_id: str = ""        # del frontmatter (`id: "abcdef12"`)
    kind: str = ""           # del frontmatter (book/paper/video/...)
    source_type: str = ""    # del frontmatter (epub/youtube/web/...)
    author: str = ""
    language: str = ""
    tags: list[str] = field(default_factory=list)  # tags reales del body (#tag1 #tag2)
    needs_review: bool = False

    @property
    def emoji(self) -> str:
        return KIND_EMOJI.get(self.kind) or KIND_EMOJI.get(self.source_type) or "📝"


# ─── Parsers ───────────────────────────────────────────────────────────────────

_FM_BOUNDARY = re.compile(r"^---\s*$", re.MULTILINE)
_FM_KEY_VAL  = re.compile(r"^([A-Za-z_][\w]*):\s*(.*)$")
_FM_LIST_ITEM = re.compile(r'^\s+-\s+"?([^"]*)"?\s*$')
_H1_RE       = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_TAGS_LINE   = re.compile(r"\*\*Etiquetas:\*\*\s*([^\n]+)")
_TAG_RE      = re.compile(r"#[\wáéíóúüñÁÉÍÓÚÜÑ\-]+")


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Devuelve (frontmatter_dict, body). Si no hay frontmatter, dict vacío y body=text."""
    if not text.startswith("---"):
        return {}, text
    # Buscamos la segunda línea "---"
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx == -1:
        return {}, text

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:]).lstrip("\n")
    fm = _parse_yaml_simple(fm_lines)
    return fm, body


def _parse_yaml_simple(lines: list[str]) -> dict:
    """Mini parser YAML para el frontmatter que escribimos. Soporta:
      key: "value"
      key: value
      key: 123
      key: true
      key:
        - "item1"
        - "item2"
    """
    out: dict = {}
    current_list_key: Optional[str] = None
    for raw in lines:
        if not raw.strip():
            current_list_key = None
            continue
        # Item de lista
        if raw.startswith("  -") or raw.startswith("\t-"):
            m = _FM_LIST_ITEM.match(raw)
            if m and current_list_key is not None:
                out.setdefault(current_list_key, [])
                out[current_list_key].append(m.group(1).strip())
            continue
        # key: value
        m = _FM_KEY_VAL.match(raw)
        if not m:
            current_list_key = None
            continue
        key, val = m.group(1), m.group(2).strip()
        if not val:
            # Inicio de lista
            current_list_key = key
            out[key] = []
            continue
        current_list_key = None
        # Quitar comillas
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        # Tipos
        if val.lower() == "true":
            out[key] = True
        elif val.lower() == "false":
            out[key] = False
        elif val.lstrip("-").isdigit():
            out[key] = int(val)
        else:
            out[key] = val
    return out


def parse_note(path: Path) -> Optional[NoteSummary]:
    """Lee y parsea una nota Markdown. Devuelve None si está corrupta."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    fm, body = _split_frontmatter(text)
    h1_match = _H1_RE.search(body)
    title = (h1_match.group(1).strip() if h1_match else path.stem)

    # Tags reales de la línea **Etiquetas:**, con fallback a source_tags del frontmatter
    tags: list[str] = []
    if m := _TAGS_LINE.search(body):
        tags = [t.lstrip("#") for t in _TAG_RE.findall(m.group(1))]
    if not tags and fm.get("source_tags"):
        tags = [str(t).lstrip("#") for t in fm["source_tags"]]

    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        mtime = datetime.fromtimestamp(0)

    return NoteSummary(
        path=path,
        mtime=mtime,
        title=title,
        note_id=str(fm.get("id", "")),
        kind=str(fm.get("kind", "")),
        source_type=str(fm.get("source_type", "")),
        author=str(fm.get("author", "")) or _first_author(fm),
        language=str(fm.get("language", "")),
        tags=tags,
        needs_review=bool(fm.get("needs_review", False)),
    )


def _first_author(fm: dict) -> str:
    a = fm.get("authors")
    if isinstance(a, list) and a:
        return str(a[0])
    return ""


# ─── Recorrido del Vault ──────────────────────────────────────────────────────

def scan_vault(base: Path, since: Optional[datetime] = None) -> Iterable[NoteSummary]:
    """Itera todas las notas .md del Vault. Si `since` está dado, solo las modificadas desde entonces."""
    if not base.exists():
        return
    for md in base.rglob("*.md"):
        if not md.is_file():
            continue
        # Saltar notas dentro de carpetas ocultas (.obsidian, .trash, etc.)
        if any(part.startswith(".") for part in md.relative_to(base).parts):
            continue
        try:
            mtime = datetime.fromtimestamp(md.stat().st_mtime)
        except OSError:
            continue
        if since and mtime < since:
            continue
        summary = parse_note(md)
        if summary:
            yield summary


# ─── /weekly ──────────────────────────────────────────────────────────────────

# Aliases comunes — el usuario puede escribir /weekly libro y entendemos book
KIND_ALIASES: dict[str, str] = {
    "libro":   "book",
    "libros":  "book",
    "book":    "book",
    "books":   "book",
    "video":   "video",
    "videos":  "video",
    "vídeo":   "video",
    "vídeos":  "video",
    "youtube": "video",
    "podcast": "podcast",
    "podcasts": "podcast",
    "paper":   "paper",
    "papers":  "paper",
    "articulo": "article",
    "artículo": "article",
    "articulos": "article",
    "artículos": "article",
    "article": "article",
    "articles": "article",
    "web":     "web",
    "webs":    "web",
    "imagen":  "image",
    "imagenes": "image",
    "imágenes": "image",
    "image":   "image",
    "images":  "image",
    "concept": "concept",
    "concepto": "concept",
    "voice":   "voice",
    "voz":     "voice",
    "tweet":   "tweet",
    "tweets":  "tweet",
    "doc":     "document",
    "document": "document",
    "documento": "document",
}


def normalize_kind(raw: str) -> Optional[str]:
    """Devuelve el `kind` canónico para un alias dado (lowercase). None si no reconocido."""
    if not raw:
        return None
    return KIND_ALIASES.get(raw.strip().lower())


def collect_weekly_notes(
    base: Path,
    days: int = 7,
    kind_filter: Optional[str] = None,
) -> list[NoteSummary]:
    """Devuelve las notas modificadas en los últimos `days` días, opcionalmente filtradas por kind."""
    cutoff = datetime.now() - timedelta(days=days)
    notes = list(scan_vault(base, since=cutoff))
    if kind_filter:
        notes = [
            n for n in notes
            if (n.kind == kind_filter) or (n.source_type == kind_filter)
        ]
    notes.sort(key=lambda n: n.mtime, reverse=True)
    return notes


def weekly_digest(
    base: Path,
    days: int = 7,
    kind_filter: Optional[str] = None,
) -> str:
    """Genera un resumen Markdown de las notas creadas/actualizadas en los últimos `days` días.

    Si `kind_filter` está dado, solo se incluyen notas de ese tipo (book, video, paper, ...).
    """
    notes = collect_weekly_notes(base, days=days, kind_filter=kind_filter)

    title_suffix = ""
    if kind_filter:
        emoji = KIND_EMOJI.get(kind_filter, "📝")
        title_suffix = f" — solo {emoji} {kind_filter}"

    if not notes:
        return (
            f"📅 *Resumen últimos {days} días*{title_suffix}\n\n"
            "_No hay notas nuevas ni actualizadas en este periodo._"
        )

    # Agrupar por kind / source_type
    by_kind: dict[str, list[NoteSummary]] = {}
    for n in notes:
        key = n.kind or n.source_type or "otros"
        by_kind.setdefault(key, []).append(n)

    # Top tags
    tag_counter: Counter[str] = Counter()
    for n in notes:
        tag_counter.update(n.tags)
    top_tags = tag_counter.most_common(8)

    review_count = sum(1 for n in notes if n.needs_review)

    lines: list[str] = [
        f"📅 *Resumen últimos {days} días*{title_suffix}",
        f"",
        f"*{len(notes)}* nota{'s' if len(notes) != 1 else ''} ingestada{'s' if len(notes) != 1 else ''}/actualizada{'s' if len(notes) != 1 else ''}.",
    ]
    if review_count:
        lines.append(f"⚠️ {review_count} pendiente{'s' if review_count != 1 else ''} de revisión.")
    lines.append("")

    # Sección por kind, ordenado por cantidad descendente
    KIND_ORDER = ["book", "paper", "video", "podcast", "article", "web", "image", "document", "concept", "voice", "tweet", "sheet"]
    sorted_kinds = sorted(
        by_kind.items(),
        key=lambda kv: (-len(kv[1]), KIND_ORDER.index(kv[0]) if kv[0] in KIND_ORDER else 99),
    )
    for kind, items in sorted_kinds:
        emoji = KIND_EMOJI.get(kind, "📝")
        label = kind.capitalize()
        lines.append(f"{emoji} *{label}* ({len(items)})")
        for n in items[:8]:  # cap por sección
            review_flag = " ⚠️" if n.needs_review else ""
            author_part = f" — _{n.author}_" if n.author else ""
            lines.append(f"  • {n.title}{author_part}{review_flag}")
        if len(items) > 8:
            lines.append(f"  _… y {len(items) - 8} más_")
        lines.append("")

    if top_tags:
        lines.append("#️⃣ *Tags más frecuentes:*")
        tag_strs = [f"#{t} ({c})" for t, c in top_tags]
        lines.append("  " + "   ".join(tag_strs))

    return "\n".join(lines).rstrip()


# ─── /random ──────────────────────────────────────────────────────────────────

def pick_random(base: Path, kind_filter: Optional[str] = None) -> Optional[NoteSummary]:
    """Devuelve una nota aleatoria del Vault. Opcionalmente filtra por kind."""
    candidates = list(scan_vault(base))
    if kind_filter:
        candidates = [
            n for n in candidates
            if (n.kind == kind_filter) or (n.source_type == kind_filter)
        ]
    if not candidates:
        return None
    return random.choice(candidates)


# ─── Meta-resumen LLM del digest semanal ──────────────────────────────────────

def build_llm_input(notes: list[NoteSummary], days: int) -> str:
    """Construye el input para que el LLM identifique temas dominantes."""
    if not notes:
        return ""
    lines = [
        f"Lista de notas creadas/actualizadas en los últimos {days} días "
        f"(total: {len(notes)}):"
    ]
    for n in notes[:80]:  # cap defensivo si la semana fue prolífica
        author = f" — {n.author}" if n.author else ""
        kind = f"[{n.kind or n.source_type}]" if (n.kind or n.source_type) else ""
        tags = (" — " + " ".join(f"#{t}" for t in n.tags[:5])) if n.tags else ""
        lines.append(f"- {kind} {n.title}{author}{tags}")
    return "\n".join(lines)


META_SYSTEM_PROMPT = """Eres un analista que recibe la lista de notas tomadas durante una semana
y devuelve un meta-resumen en {language}.

REGLAS:
- Devuelve SOLO Markdown, sin prefacio ni cierre.
- Estructura EXACTA:
  ## 🧠 Temas dominantes
  (2-4 temas en bullets — cada uno: nombre del tema en negrita y 1 frase explicativa que conecte
   notas concretas; cita títulos entre comillas dobles cuando aporte)

  ## 🔗 Conexiones interesantes
  (2-3 bullets identificando pares/grupos de notas que se relacionan entre sí
   y por qué; usa los títulos exactos)

  ## 💡 Sugerencias para profundizar
  (2-3 preguntas o líneas de investigación que emergen del conjunto)

- NO inventes notas. Trabaja solo con la lista proporcionada.
- Sé conciso, denso, sin relleno conversacional."""


META_USER_TEMPLATE = """{notes_list}

Genera el meta-resumen siguiendo el formato del system prompt."""


def llm_meta_summary(
    notes: list[NoteSummary],
    days: int,
    llm_client,               # BaseLLMClient — tipado suelto para evitar import circular
    language: str = "Español",
) -> str:
    """Llama al LLM (Ollama o Gemini) para generar un párrafo de temas dominantes
    basado en las notas. Compatible con cualquier proveedor que herede de BaseLLMClient.

    Devuelve el bloque Markdown listo para concatenar al digest, o string vacío si falla.
    """
    if not notes:
        return ""
    user = META_USER_TEMPLATE.format(notes_list=build_llm_input(notes, days))
    system = META_SYSTEM_PROMPT.format(language=language)
    try:
        # `_chat` es async — lo ejecutamos en un loop aislado para mantener la API síncrona.
        import asyncio as _asyncio
        return _asyncio.run(llm_client._chat(system=system, user=user, num_predict=1500))
    except Exception:
        return ""


# ─── /read <id> — buscar una nota por su id de frontmatter o por slug ───────

def find_note_by_id(base: Path, target: str) -> Optional[NoteSummary]:
    """Resuelve `target` a una nota concreta del Vault.

    Estrategia:
      1. Match exacto del campo `id` del frontmatter (`abcdef12`).
      2. Match parcial por prefijo del id (si es único).
      3. Match exacto del slug del filename (sin `.md`).
      4. Match parcial del slug (substring, case-insensitive) si es único.

    Devuelve `None` si no se encuentra o hay ambigüedad.
    """
    if not target:
        return None
    target_norm = target.strip().lstrip("#").lower()
    if not target_norm:
        return None

    notes = list(scan_vault(base))

    # 1. id exacto
    exact = [n for n in notes if n.note_id and n.note_id.lower() == target_norm]
    if len(exact) == 1:
        return exact[0]

    # 2. id por prefijo único
    if not exact:
        prefix = [n for n in notes if n.note_id and n.note_id.lower().startswith(target_norm)]
        if len(prefix) == 1:
            return prefix[0]

    # 3. slug del filename exacto
    slug_exact = [n for n in notes if n.path.stem.lower() == target_norm]
    if len(slug_exact) == 1:
        return slug_exact[0]

    # 4. slug substring único
    sub = [n for n in notes if target_norm in n.path.stem.lower()]
    if len(sub) == 1:
        return sub[0]

    return None


# ─── /find <keywords> — búsqueda por palabras clave (sustring AND) ──────────

@dataclass
class KeywordHit:
    note: NoteSummary
    score: int                 # más alto = más relevante
    snippet: str               # contexto del primer match

def find_by_keywords(
    base: Path,
    keywords: list[str],
    *,
    top_k: int = 8,
    snippet_len: int = 120,
) -> list[KeywordHit]:
    """Busca notas que contengan TODAS las palabras (AND) en título o cuerpo.

    Scoring sencillo y local (sin embeddings):
      • +5 por cada keyword que aparece en el TÍTULO
      • +1 por cada match en el cuerpo
      • +3 si todas las keywords aparecen en una ventana de 200 chars
    """
    if not keywords:
        return []
    kw_norm = [k.lower() for k in keywords if k.strip()]
    if not kw_norm:
        return []

    hits: list[KeywordHit] = []
    for n in scan_vault(base):
        try:
            text = n.path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        text_low = text.lower()
        title_low = n.title.lower()

        # AND: TODAS las keywords deben aparecer (en título o cuerpo)
        if not all((k in title_low or k in text_low) for k in kw_norm):
            continue

        # Score
        score = 0
        for k in kw_norm:
            if k in title_low:
                score += 5
            score += text_low.count(k)

        # Bonus de proximidad
        first_idxs = [text_low.find(k) for k in kw_norm if text_low.find(k) >= 0]
        if first_idxs and (max(first_idxs) - min(first_idxs)) < 200:
            score += 3

        # Snippet alrededor del primer match
        first_kw = kw_norm[0]
        idx = text_low.find(first_kw)
        if idx < 0:
            idx = title_low.find(first_kw)
            snippet = n.title
        else:
            start = max(0, idx - snippet_len // 2)
            end = min(len(text), idx + snippet_len)
            snippet = text[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet = snippet + "…"

        hits.append(KeywordHit(note=n, score=score, snippet=snippet))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]


def format_note_for_preview(path: Path, max_chars: int = 3500) -> tuple[str, bool]:
    """Devuelve (texto_renderizable, truncado). Sin frontmatter, listo para Telegram.

    Si el cuerpo cabe en `max_chars`, devuelve entero. Si no, recorta y añade aviso.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "_(No se pudo leer la nota.)_", False

    _, body = _split_frontmatter(text)
    body = body.strip()
    if len(body) <= max_chars:
        return body, False
    return body[:max_chars].rstrip() + "\n\n_… (truncado — abre el archivo en Obsidian)_", True
