"""PDF loading, indexing, retrieval, and grounded answer generation."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import require_api_key, settings


class RagError(RuntimeError):
    """Raised when PDF indexing or question answering fails."""


@dataclass
class PdfIndex:
    """An indexed PDF stored in memory for the current process."""

    file_name: str
    page_count: int
    chunk_count: int
    vector_store: Chroma


class PdfRagService:
    """Manage uploaded PDF indexes and answer grounded questions."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._indexes: dict[str, PdfIndex] = {}
        self._embeddings: GoogleGenerativeAIEmbeddings | None = None
        self._llm: ChatGoogleGenerativeAI | None = None

    def _get_embeddings(
        self,
    ) -> GoogleGenerativeAIEmbeddings:
        require_api_key()

        if self._embeddings is None:
            model_name = settings.embedding_model

            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"

            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=model_name
            )

        return self._embeddings

    def _get_llm(
        self,
    ) -> ChatGoogleGenerativeAI:
        require_api_key()

        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.chat_model,
                temperature=settings.temperature,
                max_tokens=settings.max_output_tokens,
                max_retries=2,
            )

        return self._llm

    @staticmethod
    def _validate_pdf_path(
        file_path: str | None,
    ) -> Path:
        if not file_path:
            raise RagError(
                "Please upload a PDF file first."
            )

        path = Path(file_path)

        if not path.exists():
            raise RagError(
                "The uploaded PDF file could not be found."
            )

        if path.suffix.lower() != ".pdf":
            raise RagError(
                "Only PDF files are supported in this lab."
            )

        maximum_bytes = (
            settings.max_pdf_mb * 1024 * 1024
        )

        if path.stat().st_size > maximum_bytes:
            raise RagError(
                "The PDF is too large. "
                f"The current limit is {settings.max_pdf_mb} MB."
            )

        return path

    @staticmethod
    def _load_pdf(
        path: Path,
    ) -> list[Document]:
        try:
            documents = PyPDFLoader(
                str(path)
            ).load()
        except Exception as error:
            raise RagError(
                f"Could not read the PDF: {error}"
            ) from error

        readable_documents = [
            document
            for document in documents
            if document.page_content.strip()
        ]

        if not readable_documents:
            raise RagError(
                "No readable text was found in the PDF. "
                "Scanned image-only PDFs require OCR, which is "
                "not included in this lab."
            )

        for document in readable_documents:
            raw_page = document.metadata.get("page")

            document.metadata.update(
                {
                    "source": path.name,
                    "page_number": (
                        raw_page + 1
                        if isinstance(raw_page, int)
                        else None
                    ),
                }
            )

        return readable_documents

    @staticmethod
    def _split_documents(
        documents: list[Document],
    ) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                "",
            ],
        )

        chunks = splitter.split_documents(
            documents
        )

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):
            chunk.metadata["chunk_number"] = index

        return chunks

    def index_pdf(
        self,
        file_path: str | None,
    ) -> tuple[str, str]:
        """Load, split, embed, and store one PDF in memory."""
        path = self._validate_pdf_path(
            file_path
        )
        documents = self._load_pdf(path)
        chunks = self._split_documents(
            documents
        )

        if not chunks:
            raise RagError(
                "The PDF did not produce any text chunks."
            )

        index_id = str(uuid4())

        try:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self._get_embeddings(),
                collection_name=(
                    "pdf_"
                    + index_id.replace("-", "")
                ),
            )
        except Exception as error:
            raise RagError(
                "The PDF could not be embedded. Check the "
                "Gemini API key, embedding model, quota, and "
                f"network connection. Details: {error}"
            ) from error

        pdf_index = PdfIndex(
            file_name=path.name,
            page_count=len(documents),
            chunk_count=len(chunks),
            vector_store=vector_store,
        )

        with self._lock:
            self._indexes[index_id] = pdf_index

        status = (
            f"Indexed **{path.name}** successfully.\n\n"
            f"- Readable pages: **{len(documents)}**\n"
            f"- Chunks: **{len(chunks)}**\n"
            f"- Embedding model: **{settings.embedding_model}**\n\n"
            "You can now ask multiple questions without "
            "embedding the PDF again."
        )

        return index_id, status

    def _get_index(
        self,
        index_id: str | None,
    ) -> PdfIndex:
        if not index_id:
            raise RagError(
                "Index the uploaded PDF before asking a question."
            )

        with self._lock:
            pdf_index = self._indexes.get(
                index_id
            )

        if pdf_index is None:
            raise RagError(
                "This PDF index is no longer available. "
                "Re-index the PDF. In-memory indexes disappear "
                "when the application restarts."
            )

        return pdf_index

    @staticmethod
    def _format_context(
        documents: list[Document],
    ) -> str:
        sections: list[str] = []

        for index, document in enumerate(
            documents,
            start=1,
        ):
            page_number = document.metadata.get(
                "page_number"
            )
            chunk_number = document.metadata.get(
                "chunk_number"
            )
            source = document.metadata.get(
                "source",
                "uploaded.pdf",
            )

            location = f"source={source}"

            if page_number:
                location += f", page={page_number}"

            if chunk_number:
                location += (
                    f", chunk={chunk_number}"
                )

            sections.append(
                (
                    f"[Document {index}: {location}]\n"
                    f"{document.page_content.strip()}"
                )
            )

        return "\n\n".join(sections)

    @staticmethod
    def _format_sources(
        documents: list[Document],
    ) -> str:
        if not documents:
            return "No source chunks were retrieved."

        lines = ["### Retrieved sources"]

        for index, document in enumerate(
            documents,
            start=1,
        ):
            source = str(
                document.metadata.get(
                    "source",
                    "uploaded.pdf",
                )
            )
            page_number = document.metadata.get(
                "page_number"
            )
            chunk_number = document.metadata.get(
                "chunk_number"
            )

            label = f"**Document {index} — {source}**"

            if page_number:
                label += f", page {page_number}"

            if chunk_number:
                label += f", chunk {chunk_number}"

            excerpt = " ".join(
                document.page_content.split()
            )[:420]

            lines.append(
                f"\n{label}\n\n> {excerpt}"
            )

        return "\n".join(lines)

    def answer_question(
        self,
        index_id: str | None,
        question: str,
    ) -> tuple[str, str]:
        """Retrieve relevant chunks and generate a grounded answer."""
        cleaned_question = question.strip()

        if not cleaned_question:
            raise RagError(
                "Please enter a question."
            )

        pdf_index = self._get_index(
            index_id
        )

        try:
            documents = (
                pdf_index.vector_store
                .similarity_search(
                    cleaned_question,
                    k=settings.top_k,
                )
            )
        except Exception as error:
            raise RagError(
                f"Document retrieval failed: {error}"
            ) from error

        if not documents:
            return (
                "I could not find relevant information in the PDF.",
                "No source chunks were retrieved.",
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a PDF question-answering assistant.\n\n"
                        "Answer only from the supplied PDF context.\n"
                        "Rules:\n"
                        "- Do not use outside knowledge.\n"
                        "- Do not invent facts.\n"
                        "- If the context is insufficient, clearly say "
                        "that the PDF does not contain enough information.\n"
                        "- Answer in the same language as the question.\n"
                        "- Cite evidence inline with labels such as "
                        "[Document 1].\n"
                        "- Keep the answer clear and reasonably concise."
                    ),
                ),
                (
                    "human",
                    (
                        "PDF context:\n{context}\n\n"
                        "Question:\n{question}"
                    ),
                ),
            ]
        )

        chain = (
            prompt
            | self._get_llm()
            | StrOutputParser()
        )

        try:
            answer = chain.invoke(
                {
                    "context": self._format_context(
                        documents
                    ),
                    "question": cleaned_question,
                }
            ).strip()
        except Exception as error:
            raise RagError(
                "Gemini could not generate the answer. "
                "Check the API key, chat model, quota, and "
                f"network connection. Details: {error}"
            ) from error

        return (
            answer,
            self._format_sources(
                documents
            ),
        )

    def remove_index(
        self,
        index_id: str | None,
    ) -> None:
        """Delete one in-memory index from the service."""
        if not index_id:
            return

        with self._lock:
            pdf_index = self._indexes.pop(
                index_id,
                None,
            )

        if pdf_index is not None:
            try:
                pdf_index.vector_store.delete_collection()
            except Exception:
                pass

    def clear_and_reset(
        self,
        index_id: str | None,
    ) -> tuple[None, str, str, str]:
        """Remove the current index and clear all UI outputs."""
        self.remove_index(
            index_id
        )

        return (
            None,
            "Upload a PDF and click **Index PDF**.",
            "",
            "",
        )
