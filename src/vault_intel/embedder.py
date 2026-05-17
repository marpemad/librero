"""Embeddings locales via Ollama (nomic-embed-text, multilingüe).

Devuelve vectores float32 normalizados — coseno = producto escalar.
"""
from __future__ import annotations

import numpy as np
import ollama

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OllamaEmbedder:
    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.embed_model

    def embed(self, text: str) -> np.ndarray:
        """Embedding normalizado. Trunca a 2 000 chars para velocidad."""
        resp = self._client.embeddings(model=self._model, prompt=text[:2000])
        vec = np.array(resp["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
