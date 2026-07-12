from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

LESSON_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_TEXT_PATH = (
    LESSON_DIRECTORY
    / "data"
    / "langchain_notes.txt"
)


def load_documents(
    pdf_path: str | None,
    url: str | None,
) -> list[Document]:
    """Load the default text file and optional external sources."""
    documents: list[Document] = []

    text_loader = TextLoader(
        str(DEFAULT_TEXT_PATH),
        encoding="utf-8",
    )

    documents.extend(
        text_loader.load()
    )

    if pdf_path:
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            raise FileNotFoundError(
                f"PDF file not found: {pdf_file}"
            )

        pdf_loader = PyPDFLoader(
            str(pdf_file)
        )

        documents.extend(
            pdf_loader.load()
        )

    if url:
        web_loader = WebBaseLoader(url)

        documents.extend(
            web_loader.load()
        )

    return documents


def display_document_statistics(
    name: str,
    documents: list[Document],
) -> None:
    """Display basic statistics for document chunks."""
    if not documents:
        print(f"{name}: no documents found")
        return

    lengths = [
        len(document.page_content)
        for document in documents
    ]

    metadata_keys = sorted(
        {
            key
            for document in documents
            for key in document.metadata
        }
    )

    average_length = (
        sum(lengths)
        / len(lengths)
    )

    print(f"\n=== {name} ===")
    print("Number of chunks:", len(documents))
    print(
        "Average chunk length:",
        f"{average_length:.2f}",
    )
    print(
        "Minimum chunk length:",
        min(lengths),
    )
    print(
        "Maximum chunk length:",
        max(lengths),
    )
    print(
        "Metadata keys:",
        metadata_keys,
    )

    example = documents[0]

    print("\nExample metadata:")
    print(example.metadata)

    print("\nExample content:")
    print(
        example.page_content[:300].replace(
            "\n",
            " ",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load documents and compare text splitters."
        )
    )

    parser.add_argument(
        "--pdf",
        help="Optional path to a local PDF file.",
    )

    parser.add_argument(
        "--url",
        help="Optional website URL to load.",
    )

    arguments = parser.parse_args()

    documents = load_documents(
        pdf_path=arguments.pdf,
        url=arguments.url,
    )

    print(
        f"Loaded {len(documents)} source documents."
    )

    character_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,
        chunk_overlap=30,
    )

    recursive_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
    )

    character_chunks = (
        character_splitter.split_documents(
            documents
        )
    )

    recursive_chunks = (
        recursive_splitter.split_documents(
            documents
        )
    )

    display_document_statistics(
        "CharacterTextSplitter",
        character_chunks,
    )

    display_document_statistics(
        "RecursiveCharacterTextSplitter",
        recursive_chunks,
    )


if __name__ == "__main__":
    main()