"""OllamaClient — proveedor LLM local vía Ollama."""
from __future__ import annotations

import asyncio

import ollama
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.llm.base_client import BaseLLMClient, SynthResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Re-exportamos SynthResult para que los imports existentes no se rompan.
__all__ = ["OllamaClient", "SynthResult"]


class OllamaClient(BaseLLMClient):
    provider_name = "ollama"

    def __init__(self) -> None:
        self._client = ollama.Client(host=settings.ollama_host)
        self._model = settings.ollama_model
        self.model_name = settings.ollama_model

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        reraise=True,
    )
    async def _chat(self, system: str, user: str, num_predict: int) -> str:
        response = await asyncio.to_thread(
            self._client.chat,
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            think=False,
            options={
                "temperature": 0.3,
                "num_predict": num_predict,
                "top_p": 0.9,
            },
        )
        msg = response["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("thinking") or "").strip()
            if content:
                logger.warning("⚠️ content vacío — usando thinking como fallback (%d chars)", len(content))
        return self._post_process(content)
