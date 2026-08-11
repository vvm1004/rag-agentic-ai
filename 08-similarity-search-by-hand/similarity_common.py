"""Shared Gemini embedding helpers for Similarity Search by Hand."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

# Reuse the shared .env from the parent course project.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)

DOCUMENTS = [
    (
        "Bugs introduced by the intern had to be squashed "
        "by the lead developer."
    ),
    (
        "Bugs found by the quality assurance engineer were "
        "difficult to debug."
    ),
    (
        "Bugs are common throughout the warm summer months, "
        "according to the entomologist."
    ),
    (
        "Bugs, in particular spiders, are extensively studied "
        "by arachnologists."
    ),
]

DEFAULT_QUERY = (
    "Who is responsible for a coding project and fixing "
    "others' mistakes?"
)


@dataclass(frozen=True)
class Settings:
    """Environment configuration used by this folder."""

    google_api_key: str
    embedding_model: str
    embedding_dimensions: int

    @property
    def api_key_configured(self) -> bool:
        normalized = self.google_api_key.strip().lower()

        return bool(
            normalized
            and not normalized.startswith("your_")
            and "replace_me" not in normalized
        )


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()

    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be an integer, received {raw!r}."
        ) from error

    if value <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return value


def load_settings() -> Settings:
    """Load Gemini embedding settings from .env."""
    return Settings(
        google_api_key=os.getenv(
            "GOOGLE_API_KEY",
            "",
        ).strip(),
        embedding_model=os.getenv(
            "GEMINI_EMBEDDING_MODEL",
            "gemini-embedding-2",
        ).strip(),
        embedding_dimensions=_read_int(
            "GEMINI_EMBEDDING_DIMENSIONS",
            768,
        ),
    )


settings = load_settings()


def require_api_key() -> None:
    """Fail clearly if the shared Google API key is missing."""
    if not settings.api_key_configured:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a placeholder "
            "value in the parent project's .env file."
        )


def get_client() -> genai.Client:
    """Create a Google GenAI client using the shared API key."""
    require_api_key()

    return genai.Client(
        api_key=settings.google_api_key
    )


def _extract_vectors(
    result: object,
) -> np.ndarray:
    """Convert a Gemini embedding response to a NumPy matrix."""
    embeddings = getattr(
        result,
        "embeddings",
        None,
    )

    if not embeddings:
        raise RuntimeError(
            "Gemini returned no embeddings."
        )

    vectors: list[list[float]] = []

    for embedding in embeddings:
        values = getattr(
            embedding,
            "values",
            None,
        )

        if not values:
            raise RuntimeError(
                "Gemini returned an embedding without vector values."
            )

        vectors.append(
            [
                float(value)
                for value in values
            ]
        )

    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise RuntimeError(
            f"Expected a 2D embedding matrix, got shape {matrix.shape}."
        )

    return matrix


def embed_texts_for_similarity(
    texts: Iterable[str],
) -> np.ndarray:
    """Embed text for symmetric sentence-to-sentence similarity.

    Gemini Embedding 2 recommends putting the task instruction into the
    content for text-only semantic-similarity use cases.
    """
    # Wrap each string in a list so the SDK treats them as separate documents
    # instead of multiple parts of a single document.
    prepared: list[Any] = [
        [f"task: sentence similarity | query: {text}"]
        for text in texts
    ]

    client = get_client()

    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=prepared,
        config=types.EmbedContentConfig(
            output_dimensionality=(
                settings.embedding_dimensions
            ),
        ),
    )

    return _extract_vectors(
        result
    )


def embed_documents_for_search(
    documents: Iterable[str],
) -> np.ndarray:
    """Embed documents using Gemini Embedding 2 retrieval formatting."""
    # Wrap each string in a list so the SDK treats them as separate documents
    prepared: list[Any] = [
        [f"title: none | text: {document}"]
        for document in documents
    ]

    client = get_client()

    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=prepared,
        config=types.EmbedContentConfig(
            output_dimensionality=(
                settings.embedding_dimensions
            ),
        ),
    )

    return _extract_vectors(
        result
    )


def embed_query_for_search(
    query: str,
) -> np.ndarray:
    """Embed one retrieval query using Gemini Embedding 2 formatting."""
    prepared = (
        f"task: search result | query: {query}"
    )

    client = get_client()

    result = client.models.embed_content(
        model=settings.embedding_model,
        contents=prepared,
        config=types.EmbedContentConfig(
            output_dimensionality=(
                settings.embedding_dimensions
            ),
        ),
    )

    matrix = _extract_vectors(
        result
    )

    if matrix.shape[0] != 1:
        raise RuntimeError(
            "Expected exactly one query embedding."
        )

    return matrix


def euclidean_distance(
    vector1: np.ndarray,
    vector2: np.ndarray,
) -> float:
    """Calculate Euclidean/L2 distance manually."""
    squared_sum = sum(
        (float(x) - float(y)) ** 2
        for x, y in zip(
            vector1,
            vector2,
            strict=True,
        )
    )

    return math.sqrt(
        squared_sum
    )


def dot_product(
    vector1: np.ndarray,
    vector2: np.ndarray,
) -> float:
    """Calculate a dot product manually."""
    return sum(
        float(x) * float(y)
        for x, y in zip(
            vector1,
            vector2,
            strict=True,
        )
    )


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    """Normalize each row vector to L2 norm 1."""
    norms = np.sqrt(
        np.sum(
            matrix ** 2,
            axis=1,
            keepdims=True,
        )
    )

    if np.any(norms == 0):
        raise ValueError(
            "Cannot normalize a zero-length vector."
        )

    return matrix / norms


def print_documents() -> None:
    """Print the corpus with stable document IDs."""
    print("DOCUMENTS")
    print("=" * 72)

    for index, document in enumerate(
        DOCUMENTS
    ):
        print(
            f"[{index}] {document}"
        )


def print_matrix(
    matrix: np.ndarray,
    *,
    title: str,
    precision: int = 4,
) -> None:
    """Print one matrix in a readable form."""
    print("\n" + title)
    print("=" * 72)

    with np.printoptions(
        precision=precision,
        suppress=True,
    ):
        print(matrix)
