from __future__ import annotations

import gradio as gr
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from llm_common import (
    create_chat_model,
    message_to_text,
    settings,
)


SYSTEM_PROMPT = (
    "You are a helpful learning assistant. "
    "Answer clearly and concisely."
)


def extract_text(content: object) -> str:
    """Support both Gradio 5 and Gradio 6 chat-history formats."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []

        for block in content:
            if isinstance(block, str):
                parts.append(block)

            elif isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts).strip()

    return ""


def chat(
    message: str,
    history: list[dict[str, object]],
) -> str:
    """Generate a reply using the visible Gradio chat history."""
    model = create_chat_model()

    messages: list[BaseMessage] = [
        SystemMessage(
            content=SYSTEM_PROMPT
        )
    ]

    for item in history:
        role = item.get("role")
        content = extract_text(
            item.get("content", "")
        )

        if not content:
            continue

        if role == "user":
            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":
            messages.append(
                AIMessage(content=content)
            )

    messages.append(
        HumanMessage(
            content=message
        )
    )

    response = model.invoke(messages)

    return message_to_text(
        response.content
    )


demo = gr.ChatInterface(
    fn=chat,
    title="Bonus: Conversational Gemini Chatbot",
    description=(
        "Unlike Lesson 04, this example sends previous chat "
        "messages back to the model."
    ),
    examples=[
        "Explain Gradio in one paragraph.",
        "What is LangChain?",
    ],
)


if __name__ == "__main__":
    demo.launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
    )
