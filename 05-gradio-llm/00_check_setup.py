from __future__ import annotations

import gradio as gr

from llm_common import (
    api_key_is_configured,
    settings,
)


def main() -> None:
    """Print local setup information."""
    print("FOLDER 05 — GRADIO LAB SETUP")
    print("=" * 72)
    print(
        "Google API key configured:",
        api_key_is_configured(),
    )
    print(
        "Gemini model:",
        settings.gemini_model,
    )
    print(
        "Temperature:",
        settings.temperature,
    )
    print(
        "Max tokens:",
        settings.max_tokens,
    )
    print(
        "Gradio version:",
        gr.__version__,
    )
    print(
        "Local URL:",
        (
            f"http://{settings.server_name}:"
            f"{settings.server_port}"
        ),
    )
    print(
        "\nThis script does not call the Gemini API."
    )


if __name__ == "__main__":
    main()
