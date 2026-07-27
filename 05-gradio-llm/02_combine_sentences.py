from __future__ import annotations

import gradio as gr

from llm_common import settings


def combine_sentences(first_sentence: str, second_sentence: str) -> str:
    """Join two sentences with one space."""
    return f"{first_sentence.strip()} {second_sentence.strip()}".strip()


demo = gr.Interface(
    fn=combine_sentences,
    inputs=[
        gr.Textbox(
            label="Input 1",
            placeholder="Enter the first sentence...",
        ),
        gr.Textbox(
            label="Input 2",
            placeholder="Enter the second sentence...",
        ),
    ],
    outputs=gr.Textbox(label="Combined output"),
    title="Combine Two Sentences",
    description=(
        "Lesson 02: Multiple Gradio inputs become function arguments."
    ),
    examples=[
        ["Hello", "world!"],
        ["Gradio creates web interfaces.", "Python handles the logic."],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
