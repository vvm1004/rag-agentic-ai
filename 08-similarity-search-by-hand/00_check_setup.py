from __future__ import annotations

import importlib.util

from similarity_common import settings


REQUIRED_MODULES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "torch": "torch",
    "dotenv": "python-dotenv",
    "google.genai": "google-genai",
}


def module_exists(module_name: str) -> bool:
    """Return whether an import path can be resolved."""
    try:
        return (
            importlib.util.find_spec(
                module_name
            )
            is not None
        )
    except (
        ImportError,
        ModuleNotFoundError,
        AttributeError,
    ):
        return False


def main() -> None:
    """Print local setup information."""
    print("FOLDER 08 — SIMILARITY SEARCH BY HAND WITH GEMINI")
    print("=" * 72)

    print(
        "Google API key configured:",
        settings.api_key_configured,
    )
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print(
        "Embedding dimensions:",
        settings.embedding_dimensions,
    )

    print("\nPython packages")
    print("-" * 72)

    for module_name, package_name in REQUIRED_MODULES.items():
        print(
            f"{package_name}:",
            (
                "OK"
                if module_exists(module_name)
                else "MISSING"
            ),
        )

    print(
        "\nThis script does not call the Gemini API."
    )


if __name__ == "__main__":
    main()
