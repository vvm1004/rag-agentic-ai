"""Configuration for the PDF RAG Gradio application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()

    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer, received {value!r}."
        ) from error


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default)).strip()

    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be a number, received {value!r}."
        ) from error


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(
        name,
        "1" if default else "0",
    ).strip().lower()

    return value in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_placeholder(value: str) -> bool:
    normalized = value.strip().lower()

    return (
        not normalized
        or normalized.startswith("your_")
        or normalized.startswith("<your")
        or "replace_me" in normalized
    )


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    google_api_key: str
    chat_model: str
    embedding_model: str

    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_output_tokens: int
    max_pdf_mb: int

    server_name: str
    server_port: int
    share: bool

    @property
    def api_key_configured(self) -> bool:
        """Return whether the Google API key looks usable."""
        return not _is_placeholder(
            self.google_api_key
        )


def load_settings() -> Settings:
    """Load settings from environment variables."""
    chunk_size = _read_int(
        "RAG_CHUNK_SIZE",
        1000,
    )
    chunk_overlap = _read_int(
        "RAG_CHUNK_OVERLAP",
        150,
    )
    top_k = _read_int(
        "RAG_TOP_K",
        4,
    )

    if chunk_size <= 0:
        raise RuntimeError(
            "RAG_CHUNK_SIZE must be greater than zero."
        )

    if (
        chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise RuntimeError(
            "RAG_CHUNK_OVERLAP must be non-negative and "
            "smaller than RAG_CHUNK_SIZE."
        )

    if top_k <= 0:
        raise RuntimeError(
            "RAG_TOP_K must be greater than zero."
        )

    return Settings(
        google_api_key=os.getenv(
            "GOOGLE_API_KEY",
            "",
        ).strip(),
        chat_model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.1-flash-lite",
        ).strip(),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-2",
        ).strip(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        temperature=_read_float(
            "RAG_TEMPERATURE",
            0.1,
        ),
        max_output_tokens=_read_int(
            "RAG_MAX_OUTPUT_TOKENS",
            1024,
        ),
        max_pdf_mb=_read_int(
            "RAG_MAX_PDF_MB",
            20,
        ),
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        ).strip(),
        server_port=_read_int(
            "GRADIO_SERVER_PORT",
            7860,
        ),
        share=_read_bool(
            "GRADIO_SHARE",
            False,
        ),
    )


settings = load_settings()


def require_api_key() -> None:
    """Raise a clear setup error when the API key is missing."""
    if not settings.api_key_configured:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a "
            "placeholder value in the shared .env file."
        )
