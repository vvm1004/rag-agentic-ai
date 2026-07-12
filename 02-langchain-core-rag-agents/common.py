"""Shared Gemini configuration."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()


def require_api_key() -> None:
    """Ensure that a Gemini API key is available."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY or GEMINI_API_KEY. "
            "Create a .env file at the project root and add your API key."
        )


def get_chat_model(
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model."""
    require_api_key()

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_retries=2,
    )


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create a Gemini embedding model."""
    require_api_key()

    model_name = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-2-preview",
    )

    return GoogleGenerativeAIEmbeddings(
        model=model_name,
    )


def get_message_text(message: Any) -> str:
    """Extract plain text from a model response."""
    text = getattr(message, "text", None)

    if isinstance(text, str) and text:
        return text

    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)

            elif isinstance(block, dict):
                block_text = block.get("text")

                if isinstance(block_text, str):
                    text_parts.append(block_text)

        return "\n".join(text_parts)

    return str(content)