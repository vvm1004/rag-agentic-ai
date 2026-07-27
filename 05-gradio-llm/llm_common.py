from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

BASE_DIR = Path(__file__).resolve().parent

# Prefer the shared .env file in the parent project.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Configuration used across the Gradio lessons."""

    google_api_key: str
    gemini_model: str
    temperature: float
    max_tokens: int
    server_name: str
    server_port: int


def _read_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default)).strip()

    try:
        return float(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be a number, received {value!r}."
        ) from error


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()

    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer, received {value!r}."
        ) from error


def load_settings() -> Settings:
    """Load shared environment settings."""
    return Settings(
        google_api_key=os.getenv(
            "GOOGLE_API_KEY",
            "",
        ).strip(),
        gemini_model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.1-flash-lite",
        ).strip(),
        temperature=_read_float(
            "LLM_TEMPERATURE",
            0.5,
        ),
        max_tokens=_read_int(
            "LLM_MAX_TOKENS",
            256,
        ),
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        ).strip(),
        server_port=_read_int(
            "GRADIO_SERVER_PORT",
            7860,
        ),
    )


settings = load_settings()


def api_key_is_configured() -> bool:
    """Return whether GOOGLE_API_KEY looks usable."""
    normalized = settings.google_api_key.lower()

    return bool(
        settings.google_api_key
        and not normalized.startswith("your_")
        and "replace_me" not in normalized
    )


def require_api_key() -> None:
    """Raise a clear error when the Google API key is missing."""
    if not api_key_is_configured():
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a placeholder "
            "value in the main project's .env file."
        )


def create_chat_model(
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model for one lesson."""
    require_api_key()

    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        temperature=(
            settings.temperature
            if temperature is None
            else temperature
        ),
        max_tokens=(
            settings.max_tokens
            if max_tokens is None
            else max_tokens
        ),
        max_retries=2,
    )


def message_to_text(content: object) -> str:
    """Convert text-model response content into a normal string."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for item in content:
            if isinstance(item, str):
                text_parts.append(item)

            elif isinstance(item, dict):
                text = item.get("text")

                if isinstance(text, str):
                    text_parts.append(text)

        return "\n".join(text_parts).strip()

    return str(content)


def generate_text(
    prompt: str,
    *,
    max_tokens: int | None = None,
) -> str:
    """Send one prompt to Gemini and return plain text."""
    cleaned_prompt = prompt.strip()

    if not cleaned_prompt:
        return "Please enter a question."

    model = create_chat_model(
        max_tokens=max_tokens
    )

    response = model.invoke(
        cleaned_prompt
    )

    return message_to_text(
        response.content
    ).strip()
