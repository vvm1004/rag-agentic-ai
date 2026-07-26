from __future__ import annotations

from rag_common import (
    print_document,
    settings,
    split_documents,
)


def main() -> None:
    """Split the source document and inspect chunk boundaries."""
    chunks = split_documents()

    print("LESSON 02 — SPLIT DOCUMENT")
    print("Chunk size:", settings.chunk_size)
    print("Chunk overlap:", settings.chunk_overlap)
    print("Chunks created:", len(chunks))

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print_document(
            chunk,
            number=index,
            max_characters=500,
        )

    print(
        "\nKey idea: overlap keeps some context on both sides "
        "of a chunk boundary."
    )


if __name__ == "__main__":
    main()
