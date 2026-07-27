from __future__ import annotations

import gradio as gr

from llm_common import settings


def sentence_builder(
    quantity: int,
    tech_worker_type: str,
    countries: list[str],
    place: str,
    activity_list: list[str],
    morning: bool,
) -> str:
    """Build one sentence from several Gradio component values."""
    country_text = (
        " and ".join(countries)
        if countries
        else "an unspecified country"
    )

    activity_text = (
        " and ".join(activity_list)
        if activity_list
        else "worked"
    )

    time_of_day = "morning" if morning else "night"

    return (
        f"The {quantity} {tech_worker_type}s from {country_text} "
        f"went to the {place}, where they {activity_text} "
        f"until the {time_of_day}."
    )


demo = gr.Interface(
    fn=sentence_builder,
    inputs=[
        gr.Slider(
            minimum=3,
            maximum=20,
            value=4,
            step=1,
            label="Count",
            info="Choose between 3 and 20.",
        ),
        gr.Dropdown(
            [
                "Data Scientist",
                "Software Developer",
                "Software Engineer",
            ],
            value="Software Developer",
            label="Tech worker type",
        ),
        gr.CheckboxGroup(
            ["Canada", "Japan", "France"],
            label="Countries",
        ),
        gr.Radio(
            ["office", "restaurant", "meeting room"],
            value="office",
            label="Location",
        ),
        gr.Dropdown(
            ["partied", "brainstormed", "coded", "fixed bugs"],
            value=["brainstormed", "fixed bugs"],
            multiselect=True,
            label="Activities",
        ),
        gr.Checkbox(
            label="Morning",
            value=True,
        ),
    ],
    outputs=gr.Textbox(label="Generated sentence"),
    title="Common Gradio Input Types",
    description=(
        "Lesson 03: Slider, Dropdown, CheckboxGroup, Radio, "
        "multiselect Dropdown, and Checkbox."
    ),
    examples=[
        [
            3,
            "Software Developer",
            ["Canada", "Japan"],
            "restaurant",
            ["coded", "fixed bugs"],
            True,
        ],
        [
            4,
            "Data Scientist",
            ["Japan"],
            "office",
            ["brainstormed", "partied"],
            False,
        ],
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
