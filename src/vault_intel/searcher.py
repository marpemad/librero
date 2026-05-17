"""Búsqueda semántica sobre el Vault: duplicados, wikilinks reales, tags coherentes.

VaultSearcher.analyze() devuelve un VaultContext que se pasa al LLM y al Writer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config.settings import settings
from src.utils.logger import get_logger
from .indexer import VaultIndexer

logger = get_logger(__name__)


@dataclass
class SimilarNote:
    similarity: float
    path: str
    title: str
    tags: list[str]


@dataclass
class VaultContext:
    """Contexto del Vault para enriquecer síntesis y frontmatter."""
    # Nota con similitud > dup_threshold_high → bot pregunta al usuario
    duplicate: SimilarNote | None = None
    # Notas con similitud media → se añaden automáticamente como `related:`
    related: list[SimilarNote] = field(default_factory=list)
    # Títulos de notas existentes sugeridos como wikilinks válidos para el LLM
    suggested_wikilinks: list[str] = field(default_factory=list)
    # Tags ya usados en el Vault (vocabulario coherente para el LLM)
    existing_tags: list[str] = field(default_factory=list)


class VaultSearcher:
    def __init__(self, indexer: VaultIndexer) -> None:
        self._idx = indexer

    def analyze(self, title_hint: str, text_snippet: str) -> VaultContext:
        """Analiza contenido entrante contra el Vault. Llamar antes de sintetizar.

        Bloquea (hace embedding vía Ollama) → usar con asyncio.to_thread.
        """
        if self._idx.note_count() == 0:
            return VaultContext(existing_tags=self._idx.get_all_tags())

        query = f"{title_hint}. {text_snippet[:500]}"
        qemb = self._idx.embed_text(query)
        results = self._idx.search(qemb, top_k=10)

        high = settings.dup_threshold_high
        low = settings.dup_threshold_low

        duplicate: SimilarNote | None = None
        related: list[SimilarNote] = []

        for sim, path, title, tags in results:
            note = SimilarNote(similarity=sim, path=path, title=title, tags=tags)
            if sim >= high:
                if duplicate is None:
                    duplicate = note
                    logger.info(
                        "Posible duplicado: '%s' (sim=%.2f)", title, sim
                    )
            elif sim >= low:
                related.append(note)
                logger.debug("Nota relacionada: '%s' (sim=%.2f)", title, sim)

        # Wikilinks válidos: títulos de las notas más cercanas con sim >= 0.40
        wikilinks = [r[2] for r in results[:6] if r[0] >= 0.40]
        existing_tags = self._idx.get_all_tags()

        return VaultContext(
            duplicate=duplicate,
            related=related[:3],
            suggested_wikilinks=wikilinks,
            existing_tags=existing_tags[:30],
        )
