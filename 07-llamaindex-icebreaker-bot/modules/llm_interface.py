"""Create the Gemini LLM and embedding model for LlamaIndex."""

from __future__ import annotations

import os

from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

import icebreaker_config as config


def _prepare_google_environment() -> None:
    """Expose the shared API key to Google GenAI integrations."""
    config.require_google_api_key()
    os.environ["GOOGLE_API_KEY"] = (
        config.settings.google_api_key
    )


def create_google_embedding() -> GoogleGenAIEmbedding:
    """Create the embedding model used for semantic retrieval."""
    _prepare_google_environment()

    return GoogleGenAIEmbedding(
        model_name=config.settings.embedding_model
    )


def create_google_llm(
    model_name: str | None = None,
) -> GoogleGenAI:
    """Create the Gemini model used for response generation."""
    _prepare_google_environment()

    return GoogleGenAI(
        model=(
            model_name
            or config.settings.llm_model
        )
    )
