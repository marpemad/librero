"""Embeddings: Ollama (local) o Gemini (nube).

Devuelve vectores float32 normalizados — coseno = producto escalar.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import google.generativeai as genai

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEmbedder(ABC):
    """Interfaz común para todos los embedders."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Devuelve embedding normalizado (float32)."""
        pass


class OllamaEmbedder(BaseEmbedder):
    """Embeddings locales via Ollama (nomic-embed-text, multilingüe)."""

    def __init__(self) -> None:
        # Lazy import: solo cuando se usa Ollama
        import ollama
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


class GeminiEmbedder(BaseEmbedder):
    """Embeddings via Google Gemini API (nube)."""

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY no está configurada en .env")
        genai.configure(api_key=settings.gemini_api_key)
        # Modelo de embeddings disponible en la API de Gemini
        self._model = "models/gemini-embedding-2"

    def embed(self, text: str) -> np.ndarray:
        """Embedding normalizado via Gemini API. Trunca a 2 000 chars."""
        try:
            result = genai.embed_content(
                model=self._model,
                content=text[:2000],
                task_type="RETRIEVAL_DOCUMENT"
            )
            vec = np.array(result["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            return vec
        except Exception as e:
            logger.error(f"Error en GeminiEmbedder: {e}")
            raise


def create_embedder() -> BaseEmbedder:
    """Factory: devuelve el embedder según LLM_PROVIDER."""
    provider = settings.llm_provider.lower().strip()
    
    if provider == "gemini":
        logger.info("Usando GeminiEmbedder para embeddings")
        return GeminiEmbedder()
    elif provider == "ollama":
        logger.info("Usando OllamaEmbedder para embeddings")
        return OllamaEmbedder()
    else:
        raise ValueError(f"Proveedor LLM desconocido: {provider}. Usa 'gemini' u 'ollama'.")
