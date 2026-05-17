from src.llm.chunker import ChunkPlan, make_plan
from src.llm.base_client import BaseLLMClient, SynthResult
from src.llm.ollama_client import OllamaClient
from src.llm.validator import ValidationResult, validate_note


def make_llm_client() -> BaseLLMClient:
    """Devuelve el cliente LLM configurado en .env (LLM_PROVIDER=ollama|gemini)."""
    from src.config.settings import settings
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient()
    return OllamaClient()


__all__ = [
    "BaseLLMClient",
    "OllamaClient",
    "SynthResult",
    "ChunkPlan",
    "make_plan",
    "make_llm_client",
    "ValidationResult",
    "validate_note",
]
