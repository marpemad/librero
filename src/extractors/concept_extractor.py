"""ConceptExtractor — texto plano sin URL ⇒ buscar en DuckDuckGo y agregar top-N páginas.

Ahora con parser de queries estructuradas:
  "Libro. Robert Kiyosaki. Padre rico, padre pobre"
  → kind=book, author=Robert Kiyosaki, title=Padre rico, padre pobre
  → búsqueda DDG con título + autor (mucho más afilada)
  → metadata factual al frontmatter y al LLM
"""
from __future__ import annotations

import asyncio
import time

from src.extractors.base import (
    Extractor,
    ExtractedContent,
    IngestionPayload,
    SourceKind,
)
from src.extractors._helpers import fetch_html, html_to_main_text
from src.extractors.concept_parser import parse_concept
from src.utils.logger import get_logger

logger = get_logger(__name__)

TOP_N = 3
MAX_CHARS_PER_PAGE = 6_000
# Reintentos ante rate-limit de DDG
_SEARCH_RETRIES = 3
_SEARCH_RETRY_DELAY = 4  # segundos entre intentos


class ConceptExtractor(Extractor):
    name = "concept"

    def can_handle(self, payload: IngestionPayload) -> bool:
        return payload.kind is SourceKind.TEXT

    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        raw_query = payload.raw.strip()
        parsed = parse_concept(raw_query)

        # Si hay metadata estructurada, la query de búsqueda es título + autor
        # (mucho más específica que el input crudo)
        search_q = parsed.search_query()
        logger.info(
            "ConceptExtractor: investigando '%s' (kind=%s, author=%s, title=%s)",
            search_q, parsed.kind, parsed.author, parsed.title,
        )

        results = await asyncio.to_thread(self._search, search_q)
        urls = [r["href"] for r in results if r.get("href")]
        logger.info("DDG devolvió %d URLs", len(urls))

        # Descarga concurrente de las top-N páginas
        pages = await asyncio.gather(
            *[self._fetch_one(u) for u in urls[:TOP_N]],
            return_exceptions=True,
        )

        # Encabezado del texto: si hay metadata, mostramos un bloque factual primero
        chunks: list[str] = []
        if parsed.has_metadata:
            header_lines = ["# Investigación estructurada"]
            if parsed.title:
                header_lines.append(f"**Título:** {parsed.title}")
            if parsed.author:
                header_lines.append(f"**Autor:** {parsed.author}")
            if parsed.kind:
                header_lines.append(f"**Tipo:** {parsed.kind}")
            chunks.append("\n".join(header_lines) + "\n")
        else:
            chunks.append(f"# Concepto investigado: {raw_query}\n")

        sources: list[str] = []
        for url, page in zip(urls[:TOP_N], pages):
            if isinstance(page, Exception) or not page:
                logger.warning("Falló fetch de %s: %s", url, page)
                continue
            sources.append(url)
            chunks.append(f"\n## Fuente: {url}\n\n{page[:MAX_CHARS_PER_PAGE]}\n")

        if len(sources) == 0:
            chunks.append(
                "\n_(No se pudo extraer contenido de ninguna fuente. "
                "El LLM responderá con su conocimiento general.)_\n"
            )

        # Construir extra con toda la metadata disponible
        extra: dict = {"sources": sources, "query": raw_query}
        if parsed.has_metadata:
            extra["concept_parsed"] = True
            if parsed.kind:
                extra["doc_kind"] = parsed.kind
            if parsed.author:
                # Para books usamos `authors` (lista, consistente con EPUB);
                # para el resto, `author` (escalar).
                if parsed.kind == "book":
                    extra["authors"] = [parsed.author]
                else:
                    extra["author"] = parsed.author
            if parsed.title:
                # `book_title` para libros — el writer lo mapea a `original_title`
                if parsed.kind == "book":
                    extra["book_title"] = parsed.title
                else:
                    extra["doc_title"] = parsed.title

        # title_hint: prioriza el título parseado para la nota
        title_hint = parsed.title or raw_query

        return ExtractedContent(
            title_hint=title_hint,
            source_type="concept",
            source_ref=f"query:{raw_query}",
            text="\n".join(chunks),
            extra=extra,
        )

    @staticmethod
    def _search(query: str) -> list[dict]:
        """Busca con ddgs (sucesor oficial de duckduckgo-search).
        Reintenta ante RatelimitException con backoff lineal.
        """
        try:
            from ddgs import DDGS
            _DDGSClass = DDGS
        except ImportError:
            # Fallback al paquete renombrado por si el entorno es viejo
            from duckduckgo_search import DDGS  # type: ignore[no-redef]
            _DDGSClass = DDGS

        last_exc: Exception | None = None
        for attempt in range(1, _SEARCH_RETRIES + 1):
            try:
                results = list(_DDGSClass().text(query, max_results=TOP_N * 2))
                if results:
                    return results
                # Resultado vacío — esperar y reintentar
                logger.warning("DDG devolvió 0 resultados (intento %d/%d)", attempt, _SEARCH_RETRIES)
            except Exception as exc:
                last_exc = exc
                logger.warning("DDG error intento %d/%d: %s", attempt, _SEARCH_RETRIES, exc)
            if attempt < _SEARCH_RETRIES:
                time.sleep(_SEARCH_RETRY_DELAY * attempt)
        logger.error("DDG falló tras %d intentos: %s", _SEARCH_RETRIES, last_exc)
        return []

    @staticmethod
    async def _fetch_one(url: str) -> str:
        try:
            html = await fetch_html(url)
            return html_to_main_text(html, url=url)
        except Exception as e:
            logger.warning("fetch fallido %s: %s", url, e)
            return ""
