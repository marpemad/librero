"""GeminiClient — proveedor LLM en la nube vía Google Gemini API."""
from __future__ import annotations

import asyncio

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.llm.base_client import BaseLLMClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GeminiClient(BaseLLMClient):
    provider_name = "gemini"

    def __init__(self) -> None:
        import google.generativeai as genai  # type: ignore[import-untyped]
        if not settings.gemini_api_key:
            raise RuntimeError(
                "Falta GEMINI_API_KEY en .env. Obtén una en https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=settings.gemini_api_key)
        self._genai = genai
        self._model_name = settings.gemini_model
        self.model_name = settings.gemini_model
        logger.info("GeminiClient iniciado con modelo %s", self._model_name)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=2, min=4, max=15),
        reraise=True,
    )
    async def _chat(self, system: str, user: str, num_predict: int) -> str:
        def _call() -> str:
            model = self._genai.GenerativeModel(
                model_name=self._model_name,
                system_instruction=system,
                generation_config=self._genai.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=num_predict,
                    top_p=0.9,
                ),
            )
            response = model.generate_content(user)
            return response.text or ""

        text = await asyncio.to_thread(_call)
        return self._post_process(text)
