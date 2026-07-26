from __future__ import annotations

from rag_common import settings


def main() -> None:
    """Print local configuration and prerequisite status."""
    api_key_configured = bool(
        settings.api_key
        and not settings.api_key.lower().startswith(
            "your_"
        )
    )

    database_files = [
        path
        for path in settings.chroma_path.rglob("*")
        if (
            path.is_file()
            and path.name != ".gitkeep"
        )
    ]

    print("FOLDER 04 — SETUP CHECK")
    print("=" * 72)
    print(
        "Source document exists:",
        settings.source_path.exists(),
    )
    print(
        "Source path:",
        settings.source_path,
    )
    print(
        "Google API key configured:",
        api_key_configured,
    )
    print(
        "Chat model:",
        settings.chat_model,
    )
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print(
        "Chunk size:",
        settings.chunk_size,
    )
    print(
        "Chunk overlap:",
        settings.chunk_overlap,
    )
    print(
        "Chroma database built:",
        bool(database_files),
    )


if __name__ == "__main__":
    main()
