"""
Detección de idioma para adaptar el system prompt al contenido.

Usa lingua-py — más fiable que langdetect para textos cortos y mezclados,
y muy bueno para distinguir idiomas latinos (ES/IT/FR/PT) que se confunden.

Carga perezosa: el detector se inicializa solo al primer uso (~1s).
"""
from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

_detector = None


def _get_detector():
    global _detector
    if _detector is not None:
        return _detector
    from lingua import Language, LanguageDetectorBuilder

    # Limitamos a los idiomas que el usuario realmente usa (ES + EN principalmente).
    # Esto mejora velocidad y precisión enormemente vs habilitar los 75 idiomas.
    langs = [Language.SPANISH, Language.ENGLISH, Language.FRENCH, Language.ITALIAN]
    _detector = (
        LanguageDetectorBuilder.from_languages(*langs)
        .with_preloaded_language_models()
        .build()
    )
    logger.info("Lingua detector cargado para: %s", [l.name for l in langs])
    return _detector


# Mapeo a nombres legibles para el prompt (en el propio idioma).
_NAMES = {
    "es": "español",
    "en": "English",
    "fr": "français",
    "it": "italiano",
}


def detect_language(text: str, sample_chars: int = 4000) -> str:
    """
    Devuelve el código ISO 639-1 ('es', 'en', 'fr', 'it'), o 'es' por defecto.
    Solo analiza los primeros `sample_chars` para velocidad.
    """
    if not text or len(text.strip()) < 30:
        return "es"
    sample = text[:sample_chars]
    try:
        result = _get_detector().detect_language_of(sample)
        if result is None:
            return "es"
        return result.iso_code_639_1.name.lower()
    except Exception as e:
        logger.warning("Fallo detección de idioma: %s — fallback ES", e)
        return "es"


def language_label(code: str) -> str:
    """Nombre legible del idioma para inyectar en el prompt."""
    return _NAMES.get(code, "español")
