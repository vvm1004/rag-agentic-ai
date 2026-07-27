from __future__ import annotations

import gradio as gr

from llm_common import settings


def add_numbers(first_number: float, second_number: float) -> float:
    """Return the sum of two numbers."""
    return first_number + second_number


demo = gr.Interface(
    fn=add_numbers,
    inputs=[
        gr.Number(label="Number 1"),
        gr.Number(label="Number 2"),
    ],
    outputs=gr.Number(label="Sum"),
    title="Simple Sum Calculator",
    description=(
        "Lesson 01: Gradio maps browser inputs to a normal Python function."
    ),
    examples=[
        [3, 4],
        [10, 25],
        [-5, 12],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
