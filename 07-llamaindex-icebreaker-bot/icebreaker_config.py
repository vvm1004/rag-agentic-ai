"""Central configuration for the LlamaIndex Icebreaker Bot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MOCK_PROFILE_PATH = DATA_DIR / "mock_linkedin_profile.json"

# Reuse the shared .env in the parent course project.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)

INITIAL_FACTS_TEMPLATE = """
You are an AI networking assistant.

Use ONLY the profile context below.
Do not use outside knowledge and do not invent facts.

Profile context:
---------------------
{context_str}
---------------------

Generate exactly 3 interesting, specific conversation starters based on the
person's career, education, projects, skills, or interests.

For each item:
1. State the relevant fact briefly.
2. Suggest one natural icebreaker question connected to that fact.

Keep the tone professional and friendly.
"""

USER_QUESTION_TEMPLATE = """
You answer questions about one professional profile.

Use ONLY the profile context below.
If the answer is not supported by the context, say:
"I don't know based on the available profile data."

Profile context:
---------------------
{context_str}
---------------------

User question:
{query_str}

Answer clearly and concisely.
"""


def _read_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default)).strip()

    try:
        return int(value)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer, received {value!r}."
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


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    google_api_key: str
    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    similarity_top_k: int
    server_name: str
    server_port: int
    share: bool

    @property
    def api_key_configured(self) -> bool:
        normalized = self.google_api_key.lower()

        return bool(
            self.google_api_key
            and not normalized.startswith("your_")
            and "replace_me" not in normalized
        )


def load_settings() -> Settings:
    """Load and validate environment configuration."""
    chunk_size = _read_int(
        "ICEBREAKER_CHUNK_SIZE",
        500,
    )
    chunk_overlap = _read_int(
        "ICEBREAKER_CHUNK_OVERLAP",
        50,
    )
    similarity_top_k = _read_int(
        "ICEBREAKER_TOP_K",
        5,
    )

    if chunk_size <= 0:
        raise RuntimeError(
            "ICEBREAKER_CHUNK_SIZE must be greater than zero."
        )

    if (
        chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise RuntimeError(
            "ICEBREAKER_CHUNK_OVERLAP must be non-negative and smaller "
            "than ICEBREAKER_CHUNK_SIZE."
        )

    if similarity_top_k <= 0:
        raise RuntimeError(
            "ICEBREAKER_TOP_K must be greater than zero."
        )

    return Settings(
        google_api_key=os.getenv(
            "GOOGLE_API_KEY",
            "",
        ).strip(),
        llm_model=os.getenv(
            "GEMINI_MODEL",
            "gemini-3.1-flash-lite",
        ).strip(),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-2",
        ).strip(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        similarity_top_k=similarity_top_k,
        server_name=os.getenv(
            "ICEBREAKER_SERVER_NAME",
            "127.0.0.1",
        ).strip(),
        server_port=_read_int(
            "ICEBREAKER_SERVER_PORT",
            5002,
        ),
        share=_read_bool(
            "ICEBREAKER_SHARE",
            False,
        ),
    )


settings = load_settings()


def require_google_api_key() -> None:
    """Fail with a clear message if the shared Google API key is missing."""
    if not settings.api_key_configured:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a placeholder "
            "value in the parent project's .env file."
        )
