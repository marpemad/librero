"""
Base del Strategy Pattern para extracción de contenido.

Cada nueva fuente = una nueva clase que herede de `Extractor`
y se registre en el `ExtractorFactory`. El router NUNCA se toca.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class SourceKind(Enum):
    """Tipos de payload que llegan al router."""
    TEXT = auto()         # Texto puro (sin URL) → Concepto a investigar
    URL = auto()          # Una o varias URLs en el mensaje
    FILE = auto()         # Adjunto en Telegram (PDF, EPUB, DOCX, CSV, TXT, MD)


@dataclass
class IngestionPayload:
    """Lo que el router le pasa al extractor seleccionado."""
    kind: SourceKind
    raw: str                           # texto del mensaje, URL, o ruta al archivo descargado
    metadata: dict[str, Any] = field(default_factory=dict)
    # Por ejemplo, para FILE: {"mime": "application/pdf", "filename": "paper.pdf"}


@dataclass
class ExtractedContent:
    """Resultado de cualquier extractor — formato uniforme para el LLM."""
    title_hint: str               # Pista de título (la URL, el nombre del archivo, la idea…)
    source_type: str              # "youtube", "pdf", "tweet", "concept"…
    source_ref: str               # URL original o ruta — irá en el frontmatter
    text: str                     # Contenido textual ya limpio
    extra: dict[str, Any] = field(default_factory=dict)  # autor, duración, fecha, etc.


class Extractor(ABC):
    """
    Clase base. Cada extractor implementa:
      - can_handle(): ¿esta estrategia aplica a este payload?
      - extract():    devuelve ExtractedContent
    """

    #: Nombre legible para logs y para el frontmatter
    name: str = "abstract"

    @abstractmethod
    def can_handle(self, payload: IngestionPayload) -> bool:
        ...

    @abstractmethod
    async def extract(self, payload: IngestionPayload) -> ExtractedContent:
        ...
