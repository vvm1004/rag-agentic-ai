from __future__ import annotations

from similarity_common import (
    DOCUMENTS,
    embed_texts_for_similarity,
    print_documents,
    settings,
)


def main() -> None:
    """Create Gemini embeddings and inspect their shape."""
    print("LESSON 01 — CREATE GEMINI EMBEDDINGS")
    print("=" * 72)
    print(
        "Model:",
        settings.embedding_model,
    )
    print(
        "Configured dimensions:",
        settings.embedding_dimensions,
    )
    print()

    print_documents()

    embeddings = embed_texts_for_similarity(
        DOCUMENTS
    )

    print("\nEMBEDDINGS")
    print("=" * 72)
    print(
        "Shape:",
        embeddings.shape,
    )
    print(
        "Number of documents:",
        embeddings.shape[0],
    )
    print(
        "Vector dimensions:",
        embeddings.shape[1],
    )
    print(
        "\nFirst 10 values of document 0:"
    )
    print(
        embeddings[0][:10]
    )

    print(
        "\nKey idea: Gemini changed the text into numerical "
        "vectors. The rest of this folder calculates similarity "
        "with NumPy/Python rather than a vector database."
    )


if __name__ == "__main__":
    main()
