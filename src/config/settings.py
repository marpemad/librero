"""Configuración global cargada desde .env"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _parse_user_ids(raw: str | None) -> set[int]:
    if not raw:
        return set()
    return {int(x.strip()) for x in raw.split(",") if x.strip()}


@dataclass(frozen=True)
class Settings:
    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    allowed_user_ids: set[int] = field(
        default_factory=lambda: _parse_user_ids(os.getenv("ALLOWED_USER_IDS"))
    )

    # LLM — proveedor activo: "ollama" (local) o "gemini" (nube)
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama (solo si LLM_PROVIDER=ollama)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.6:27b-q4_K_M")

    # Gemini (solo si LLM_PROVIDER=gemini)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    # Modelo de visión para describir imágenes (debe ser multimodal).
    # Recomendados: qwen2.5vl:7b · llama3.2-vision:11b · gemma3:4b
    # Cadena vacía → desactivado, solo OCR.
    vision_model: str = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

    # Obsidian
    obsidian_vault_path: Path = Path(
        os.getenv("OBSIDIAN_VAULT_PATH", str(Path.home() / "ObsidianVault"))
    )
    obsidian_inbox_folder: str = os.getenv("OBSIDIAN_INBOX_FOLDER", "00_Inbox")

    # Whisper
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "auto")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

    # Temp
    temp_dir: Path = Path(os.getenv("TEMP_DIR", "/tmp/librero"))

    # Chunking (Fase 1)
    chunk_threshold_chars: int = int(os.getenv("CHUNK_THRESHOLD_CHARS", "20000"))
    chunk_size_chars: int = int(os.getenv("CHUNK_SIZE_CHARS", "12000"))
    chunk_overlap_chars: int = int(os.getenv("CHUNK_OVERLAP_CHARS", "500"))
    confirm_threshold_chars: int = int(os.getenv("CONFIRM_THRESHOLD_CHARS", "50000"))

    # Caché
    cache_db_path: Path = Path(os.getenv("CACHE_DB_PATH", "/tmp/librero/cache.db"))

    # Vault Intel (Fase 2)
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    vault_index_db_path: Path = Path(
        os.getenv("VAULT_INDEX_DB_PATH", "/tmp/librero/vault_index.db")
    )
    dup_threshold_high: float = float(os.getenv("DUP_THRESHOLD_HIGH", "0.85"))
    dup_threshold_low: float = float(os.getenv("DUP_THRESHOLD_LOW", "0.65"))

    # Tareas — un único markdown actuando de tablón Kanban dentro del Vault
    tasks_file_path: Path = Path(
        os.getenv(
            "TASKS_FILE",
            str(Path(os.getenv("OBSIDIAN_VAULT_PATH", str(Path.home() / "ObsidianVault"))) / "Tablón.md"),
        )
    )

    # Listas curadas — cada tipo vive en su propio .md dentro de esta carpeta
    lists_folder: str = os.getenv("LISTS_FOLDER", "Listas")

    # Google Calendar
    gcal_client_id: str = os.getenv("GCAL_CLIENT_ID", "")
    gcal_client_secret: str = os.getenv("GCAL_CLIENT_SECRET", "")
    gcal_token_path: Path = Path(
        os.getenv("GCAL_TOKEN_PATH", str(Path.home() / ".config" / "librero" / "gcal_token.json"))
    )
    gcal_notify_db_path: Path = Path(
        os.getenv("GCAL_NOTIFY_DB_PATH", "/tmp/librero/gcal_notify.db")
    )
    gcal_auth_port: int = int(os.getenv("GCAL_AUTH_PORT", "8765"))
    gcal_daily_summary_time: str = os.getenv("GCAL_DAILY_SUMMARY_TIME", "08:00")

    def validate(self) -> None:
        if not self.telegram_token:
            raise RuntimeError("Falta TELEGRAM_BOT_TOKEN en .env")
        if not self.allowed_user_ids:
            raise RuntimeError(
                "Falta ALLOWED_USER_IDS en .env (no exponer el bot al público)."
            )
        provider = self.llm_provider.lower()
        if provider not in ("ollama", "gemini"):
            raise RuntimeError(
                f"LLM_PROVIDER='{self.llm_provider}' no es válido. Usa 'ollama' o 'gemini'."
            )
        if provider == "gemini" and not self.gemini_api_key:
            raise RuntimeError(
                "LLM_PROVIDER=gemini pero falta GEMINI_API_KEY en .env "
                "(obtén una en https://aistudio.google.com/apikey)."
            )
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        (self.obsidian_vault_path / self.obsidian_inbox_folder).mkdir(
            parents=True, exist_ok=True
        )

    @property
    def inbox_path(self) -> Path:
        return self.obsidian_vault_path / self.obsidian_inbox_folder

    @property
    def lists_path(self) -> Path:
        return self.obsidian_vault_path / self.lists_folder


settings = Settings()
