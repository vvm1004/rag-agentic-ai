"""Shared helpers for document chunking and vector storage."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from common import get_embeddings

LESSON_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_TEXT_PATH = (
    LESSON_DIRECTORY
    / "data"
    / "langchain_notes.txt"
)


def load_chunks() -> list[Document]:
    """Load the lesson document and split it into chunks."""
    loader = TextLoader(
        str(DEFAULT_TEXT_PATH),
        encoding="utf-8",
    )

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )

    return splitter.split_documents(
        documents
    )


def build_vector_store() -> tuple[
    Chroma,
    list[Document],
]:
    """Create an in-memory Chroma vector store."""
    chunks = load_chunks()
    embeddings = get_embeddings()

    collection_name = (
        f"langchain_lesson_{uuid4().hex}"
    )

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
    )

    document_ids = [
        str(uuid4())
        for _ in chunks
    ]

    vector_store.add_documents(
        documents=chunks,
        ids=document_ids,
    )

    return vector_store, chunks