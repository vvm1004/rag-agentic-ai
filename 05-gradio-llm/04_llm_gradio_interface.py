from __future__ import annotations

import gradio as gr

from llm_common import (
    generate_text,
    settings,
)


def generate_response(prompt_text: str) -> str:
    """Generate one single-turn LLM response."""
    return generate_text(prompt_text)


chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(
        label="Input",
        lines=3,
        placeholder="Type your question here...",
    ),
    outputs=gr.Textbox(
        label="Output",
        lines=10,
    ),
    title="Gemini Q&A with Gradio",
    description=(
        "Lesson 04: Gradio sends Textbox input to a Python "
        "function, which calls Gemini and returns text."
    ),
    examples=[
        ["How can I become a good data scientist?"],
        ["Explain retrieval-augmented generation simply."],
        ["What is the difference between Flask and Gradio?"],
    ],
)


if __name__ == "__main__":
    chat_application.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
