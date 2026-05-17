"""
Parser de queries estructuradas para `ConceptExtractor`.

Permite que el usuario escriba en Telegram cosas como:

    Libro. Robert Kiyosaki. Padre rico, padre pobre
    Película: Inception - Christopher Nolan
    Podcast | Lex Fridman | Entrevista a Carmack

…y extraigamos `kind`, `author` y `title` para:
  • Construir una búsqueda DDG mucho más específica.
  • Pasar al LLM datos factuales que NO debe inventar.
  • Volcar todo al frontmatter (igual que un libro/vídeo cualquiera).

Si la query no encaja con ningún patrón, se trata como concepto libre
(comportamiento de siempre).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# Mapeo de palabras clave de tipo (incluye variantes en español/inglés)
# Cualquier alias debe estar en minúsculas.
KIND_KEYWORDS: dict[str, str] = {
    # Libros
    "libro": "book", "libros": "book", "book": "book", "books": "book",
    # Vídeos / YouTube
    "video": "video", "vídeo": "video", "videos": "video", "vídeos": "video",
    "youtube": "video",
    # Podcasts
    "podcast": "podcast", "podcasts": "podcast", "episodio": "podcast",
    # Papers / artículos académicos
    "paper": "paper", "papers": "paper",
    "artículo": "article", "articulo": "article",
    "articulos": "article", "artículos": "article",
    # Películas
    "pelicula": "movie", "película": "movie",
    "peliculas": "movie", "películas": "movie",
    "film": "movie", "movie": "movie",
    # Series
    "serie": "tv", "series": "tv", "show": "tv",
    # Música
    "cancion": "song", "canción": "song",
    "canciones": "song", "song": "song",
    "album": "album", "álbum": "album",
    # Juegos
    "juego": "game", "videojuego": "game", "game": "game",
    # Personas
    "persona": "person", "person": "person",
    "autor": "person", "author": "person",
    # Lugares
    "lugar": "place", "place": "place",
    "ciudad": "place", "city": "place",
    # Conceptos puros (forzar la sección concept)
    "concepto": "concept", "concept": "concept",
    "idea": "concept",
}


@dataclass
class ConceptQuery:
    """Resultado del parseo del input del usuario."""
    raw: str                          # input original
    kind: Optional[str] = None        # "book", "video", "movie", …
    author: Optional[str] = None
    title: Optional[str] = None
    has_metadata: bool = False        # True si se reconoció estructura

    def search_query(self) -> str:
        """Construye la query óptima para DuckDuckGo.

        Prioriza título + autor (lo más específico). Si no hay metadata
        estructurada, devuelve el input crudo.
        """
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        if self.author:
            parts.append(self.author)
        return " ".join(parts) if parts else self.raw


# ---------------------------------------------------------------------------

# Separadores principales entre tipo / autor / título.
# Aceptamos: "." " | " " : " (sin pegar). Permite que el título contenga "."
# sin romperlo (lo arreglamos con la lógica de unión más abajo).
_PRIMARY_SPLIT_RE = re.compile(r"\s*\|\s*|\s*:\s+|\s*\.\s+")

# Separadores dentro de un solo bloque (cuando hay un único tramo tras el tipo)
# y queremos partirlo en (título, autor). Casos: "Inception by Nolan",
# "Inception - Nolan", "Inception — Nolan".
_INNER_SPLIT_RE = re.compile(
    r"\s+(?:by|de|por)\s+|\s+[-–—]\s+",
    re.IGNORECASE,
)


def parse_concept(text: str) -> ConceptQuery:
    """Parsea una entrada como 'Libro. Robert Kiyosaki. Padre rico'.

    Reglas:
      • Si el primer token coincide con un alias de KIND_KEYWORDS, hay metadata.
      • Tras el tipo, el orden por defecto es:  AUTOR, TÍTULO.
        - 2 partes → (autor, título).
        - 1 parte  → solo título; intentamos partir por ' - ', ' by ', ' de '
          para detectar autor secundariamente.
      • Sin tipo reconocible → query libre (`has_metadata=False`).
    """
    raw = (text or "").strip()
    if not raw:
        return ConceptQuery(raw=raw)

    # Spliteamos por separador primario
    parts = [p.strip() for p in _PRIMARY_SPLIT_RE.split(raw) if p.strip()]
    if not parts:
        return ConceptQuery(raw=raw)

    # ¿La primera parte es un tipo conocido?
    first_token = parts[0].lower().rstrip(".:")
    kind = KIND_KEYWORDS.get(first_token)

    if not kind:
        # Free-form: sin metadata estructurada
        return ConceptQuery(raw=raw)

    rest = parts[1:]
    author: Optional[str] = None
    title: Optional[str] = None

    if len(rest) >= 2:
        author = rest[0]
        # El resto se une (puede tener "." dentro del título)
        title = ". ".join(rest[1:]).strip()
    elif len(rest) == 1:
        chunk = rest[0]
        # ¿Hay un separador "interno" tipo ' - ' o ' by '?
        m = _INNER_SPLIT_RE.search(chunk)
        if m:
            title = chunk[: m.start()].strip()
            author = chunk[m.end():].strip()
        else:
            # Solo título
            title = chunk
    # else: solo el tipo, sin más → no aporta nada útil

    # Sanity: ignorar valores vacíos
    if author == "":
        author = None
    if title == "":
        title = None

    return ConceptQuery(
        raw=raw,
        kind=kind,
        author=author,
        title=title,
        has_metadata=True,
    )
