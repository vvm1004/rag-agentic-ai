"""Check the environment without calling Gemini."""

from __future__ import annotations

import gradio as gr

from config import settings


def main() -> None:
    """Print local configuration status."""
    print("FOLDER 06 — PDF RAG GRADIO APP")
    print("=" * 72)
    print(
        "Google API key configured:",
        settings.api_key_configured,
    )
    print("Chat model:", settings.chat_model)
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print("Chunk size:", settings.chunk_size)
    print(
        "Chunk overlap:",
        settings.chunk_overlap,
    )
    print("Top K:", settings.top_k)
    print(
        "Maximum PDF size:",
        f"{settings.max_pdf_mb} MB",
    )
    print("Gradio version:", gr.__version__)
    print(
        "Application URL:",
        (
            f"http://{settings.server_name}:"
            f"{settings.server_port}"
        ),
    )
    print(
        "\nThis command does not call the Gemini API."
    )


if __name__ == "__main__":
    main()
