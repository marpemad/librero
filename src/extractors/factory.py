"""
ExtractorFactory — implementa el Strategy Pattern.

⭐ Para añadir una nueva fuente:
   1. Crea una clase que herede de Extractor.
   2. Regístrala en `_REGISTRY` en el orden adecuado (más específico primero).
El router NO se modifica jamás.
"""
from __future__ import annotations

from src.extractors.base import Extractor, IngestionPayload
from src.extractors.concept_extractor import ConceptExtractor
from src.extractors.document_extractor import DocumentExtractor
from src.extractors.media_extractor import MediaExtractor
from src.extractors.social_extractor import SocialExtractor
from src.extractors.voice_extractor import VoiceExtractor
from src.extractors.web_extractor import GoogleSheetsExtractor, WebScraperExtractor
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ⚠️ ORDEN IMPORTA: del más específico al más genérico.
# Por ejemplo, SocialExtractor debe ir antes que WebScraperExtractor,
# si no las URLs de twitter caerían en el scraper genérico.
# VoiceExtractor antes de DocumentExtractor para que .ogg/.m4a vayan a Whisper, no a un parser.
_REGISTRY: list[Extractor] = [
    SocialExtractor(),
    MediaExtractor(),
    GoogleSheetsExtractor(),
    VoiceExtractor(),
    DocumentExtractor(),
    WebScraperExtractor(),     # fallback para cualquier URL HTTP/S
    ConceptExtractor(),        # fallback final: texto puro → investigar
]


class ExtractorFactory:
    @staticmethod
    def select(payload: IngestionPayload) -> Extractor:
        for extractor in _REGISTRY:
            if extractor.can_handle(payload):
                logger.info("Factory eligió: %s", extractor.name)
                return extractor
        raise ValueError(
            f"Ningún extractor puede manejar el payload kind={payload.kind}"
        )

    @staticmethod
    def registered() -> list[str]:
        return [e.name for e in _REGISTRY]
