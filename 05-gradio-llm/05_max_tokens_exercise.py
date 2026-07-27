from __future__ import annotations

import gradio as gr

from llm_common import (
    generate_text,
    settings,
)


def generate_short_response(prompt_text: str) -> str:
    """Generate with a smaller output-token limit."""
    return generate_text(
        prompt_text,
        max_tokens=256,
    )


def generate_long_response(prompt_text: str) -> str:
    """Generate with a larger output-token limit."""
    return generate_text(
        prompt_text,
        max_tokens=512,
    )


def compare_output_limits(
    prompt_text: str,
) -> tuple[str, str]:
    """Return responses produced with two token limits."""
    return (
        generate_short_response(prompt_text),
        generate_long_response(prompt_text),
    )


demo = gr.Interface(
    fn=compare_output_limits,
    inputs=gr.Textbox(
        label="Prompt",
        lines=3,
        value=(
            "Explain in detail how a beginner can become "
            "a strong data scientist."
        ),
    ),
    outputs=[
        gr.Textbox(
            label="Max 256 tokens",
            lines=12,
        ),
        gr.Textbox(
            label="Max 512 tokens",
            lines=12,
        ),
    ],
    title="Output Length Exercise",
    description=(
        "Lesson 05: Compare responses with different maximum "
        "output-token limits. A larger limit permits a longer "
        "answer but does not force the model to use every token."
    ),
)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
