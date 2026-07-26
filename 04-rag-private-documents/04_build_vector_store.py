from __future__ import annotations

from rag_common import (
    build_vector_store,
    settings,
    split_documents,
)


def main() -> None:
    """Embed every chunk and save it in Chroma."""
    chunks = split_documents()

    print("LESSON 04 — BUILD VECTOR STORE")
    print("Chunks to index:", len(chunks))
    print("Chroma path:", settings.chroma_path)
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print("\nBuilding the index...")

    vector_store = build_vector_store(
        reset=True
    )

    collection_count = len(
        vector_store.get().get("ids", [])
    )

    print(
        "Indexed Chroma records:",
        collection_count,
    )
    print(
        "\nKey idea: Chroma stores each chunk together with "
        "its vector and metadata so it can be retrieved later."
    )


if __name__ == "__main__":
    main()
