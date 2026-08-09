"""Check Folder 07 configuration without calling Gemini."""

from __future__ import annotations

import importlib.util

import icebreaker_config as config


REQUIRED_MODULES = {
    "gradio": "gradio",
    "llama_index.core": "llama-index-core",
    "llama_index.llms.google_genai": (
        "llama-index-llms-google-genai"
    ),
    "llama_index.embeddings.google_genai": (
        "llama-index-embeddings-google-genai"
    ),
}


def main() -> None:
    """Print local setup information."""
    print("FOLDER 07 — LLAMAINDEX ICEBREAKER BOT")
    print("=" * 72)
    print(
        "Google API key configured:",
        config.settings.api_key_configured,
    )
    print(
        "Gemini model:",
        config.settings.llm_model,
    )
    print(
        "Embedding model:",
        config.settings.embedding_model,
    )
    print(
        "Chunk size:",
        config.settings.chunk_size,
    )
    print(
        "Chunk overlap:",
        config.settings.chunk_overlap,
    )
    print(
        "Similarity top K:",
        config.settings.similarity_top_k,
    )
    print(
        "Mock profile:",
        config.MOCK_PROFILE_PATH,
    )
    print(
        "Mock profile exists:",
        config.MOCK_PROFILE_PATH.exists(),
    )

    print("\nPython packages")
    print("-" * 72)

    for module_name, package_name in REQUIRED_MODULES.items():
        installed = (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )

        print(
            f"{package_name}:",
            "OK" if installed else "MISSING",
        )

    print(
        "\nApplication URL:",
        (
            f"http://{config.settings.server_name}:"
            f"{config.settings.server_port}"
        ),
    )
    print(
        "\nThis script does not call the Gemini API."
    )


if __name__ == "__main__":
    main()
