from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)

from common import get_chat_model


def demonstrate_string_prompt() -> None:
    """Create and run a reusable string prompt."""
    prompt = PromptTemplate.from_template(
        """
Explain {topic} to a beginner.

Requirements:
- Use exactly {bullet_count} bullet points.
- Keep each bullet point concise.
- Avoid unnecessary technical jargon.
""".strip()
    )

    prompt_value = prompt.invoke(
        {
            "topic": "LangChain Expression Language",
            "bullet_count": 3,
        }
    )

    print("=== FORMATTED PROMPT ===")
    print(prompt_value.to_string())

    chain = (
        prompt
        | get_chat_model(temperature=0.2)
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "topic": "LangChain Expression Language",
            "bullet_count": 3,
        }
    )

    print("\n=== CHAIN RESULT ===")
    print(result)


def demonstrate_chat_prompt() -> None:
    """Create a chat prompt with conversation history."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a concise programming tutor. "
                    "Use simple explanations and practical examples."
                ),
            ),
            MessagesPlaceholder("history"),
            (
                "human",
                "{question}",
            ),
        ]
    )

    history = [
        HumanMessage(
            content="What is LangChain?"
        ),
        AIMessage(
            content=(
                "LangChain is a framework for building "
                "applications powered by language models."
            )
        ),
    ]

    chain = (
        prompt
        | get_chat_model(temperature=0.2)
        | StrOutputParser()
    )

    result = chain.invoke(
        {
            "history": history,
            "question": (
                "What does the pipe operator do in LCEL?"
            ),
        }
    )

    print("\n=== CHAT PROMPT RESULT ===")
    print(result)


def main() -> None:
    demonstrate_string_prompt()
    demonstrate_chat_prompt()


if __name__ == "__main__":
    main()