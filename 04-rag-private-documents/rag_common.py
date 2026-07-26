from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

BASE_DIR = Path(__file__).resolve().parent

# Prefer the shared .env in the main project.
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Configuration used by every lesson script."""

    source_path: Path
    chroma_path: Path
    collection_name: str

    api_key: str
    chat_model: str
    embedding_model: str

    chunk_size: int
    chunk_overlap: int
    top_k: int
    temperature: float
    max_output_tokens: int


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


def load_settings() -> Settings:
    """Load settings from the parent project's environment."""
    chunk_size = _read_int(
        "RAG_CHUNK_SIZE",
        1000,
    )
    chunk_overlap = _read_int(
        "RAG_CHUNK_OVERLAP",
        150,
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

    return Settings(
        source_path=(
            BASE_DIR
            / "data"
            / "company_policies.txt"
        ),
        chroma_path=BASE_DIR / "chroma_db",
        collection_name="folder04_private_documents",
        api_key=os.getenv(
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
        top_k=_read_int(
            "RAG_TOP_K",
            4,
        ),
        temperature=_read_float(
            "RAG_TEMPERATURE",
            0.1,
        ),
        max_output_tokens=_read_int(
            "RAG_MAX_OUTPUT_TOKENS",
            1024,
        ),
    )


settings = load_settings()


def require_api_key() -> None:
    """Raise a clear error when the API key is missing."""
    normalized = settings.api_key.lower()

    if (
        not settings.api_key
        or normalized.startswith("your_")
        or "replace_me" in normalized
    ):
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a "
            "placeholder value in the parent project's .env file."
        )


def normalize_embedding_model_name(
    model_name: str,
) -> str:
    """Return a model name accepted by the Gemini integration."""
    cleaned = model_name.strip()

    if cleaned.startswith("models/"):
        return cleaned

    return f"models/{cleaned}"


def load_documents() -> list[Document]:
    """Load the sample text file as LangChain Documents."""
    if not settings.source_path.exists():
        raise FileNotFoundError(
            f"Source document not found: {settings.source_path}"
        )

    loader = TextLoader(
        str(settings.source_path),
        autodetect_encoding=True,
    )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = (
            settings.source_path.name
        )

    return documents


def split_documents(
    documents: list[Document] | None = None,
) -> list[Document]:
    """Split documents into overlapping semantic chunks."""
    source_documents = (
        documents
        if documents is not None
        else load_documents()
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        source_documents
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        chunk.metadata["chunk_number"] = index

    return chunks


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create the Gemini embedding client."""
    require_api_key()

    return GoogleGenerativeAIEmbeddings(
        model=normalize_embedding_model_name(
            settings.embedding_model
        )
    )


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    require_api_key()

    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        max_tokens=settings.max_output_tokens,
        max_retries=2,
    )


def build_vector_store(
    *,
    reset: bool = True,
) -> Chroma:
    """Create and persist a Chroma collection from all chunks."""
    if reset and settings.chroma_path.exists():
        shutil.rmtree(
            settings.chroma_path
        )

    settings.chroma_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunks = split_documents()

    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=settings.collection_name,
        persist_directory=str(
            settings.chroma_path
        ),
    )


def load_vector_store() -> Chroma:
    """Open the persisted Chroma collection."""
    database_files = [
        path
        for path in settings.chroma_path.rglob("*")
        if (
            path.is_file()
            and path.name != ".gitkeep"
        )
    ]

    if not database_files:
        raise RuntimeError(
            "The Chroma database has not been built. "
            "Run 04_build_vector_store.py first."
        )

    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(
            settings.chroma_path
        ),
    )

    stored_ids = vector_store.get().get("ids", [])

    if not stored_ids:
        raise RuntimeError(
            "The Chroma collection is empty. "
            "Run 04_build_vector_store.py again."
        )

    return vector_store


def format_context(
    documents: Iterable[Document],
) -> str:
    """Format retrieved documents for an LLM prompt."""
    sections: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )
        chunk_number = document.metadata.get(
            "chunk_number",
            "?",
        )

        sections.append(
            (
                f"[Document {index}: "
                f"source={source}, chunk={chunk_number}]\n"
                f"{document.page_content.strip()}"
            )
        )

    return "\n\n".join(sections)


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    """Calculate cosine similarity without external math packages."""
    if len(first) != len(second):
        raise ValueError(
            "Vectors must have the same number of dimensions."
        )

    dot_product = sum(
        left * right
        for left, right in zip(
            first,
            second,
            strict=True,
        )
    )

    first_norm = math.sqrt(
        sum(value * value for value in first)
    )
    second_norm = math.sqrt(
        sum(value * value for value in second)
    )

    if first_norm == 0 or second_norm == 0:
        return 0.0

    return dot_product / (
        first_norm * second_norm
    )


def print_document(
    document: Document,
    *,
    number: int | None = None,
    max_characters: int = 500,
) -> None:
    """Print one document or chunk in a readable format."""
    heading = (
        f"DOCUMENT {number}"
        if number is not None
        else "DOCUMENT"
    )

    print("\n" + "=" * 72)
    print(heading)
    print("=" * 72)
    print("Metadata:", document.metadata)
    print(
        "Characters:",
        len(document.page_content),
    )
    print("\nContent preview:")
    print(
        document.page_content[
            :max_characters
        ].strip()
    )

    if len(document.page_content) > max_characters:
        print("...")
