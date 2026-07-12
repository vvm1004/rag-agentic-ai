from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from common import get_chat_model, get_message_text


def compare_temperatures() -> None:
    """Compare precise and creative model configurations."""
    prompt = (
        "Write a two-line slogan for an AI-powered study assistant."
    )

    precise_model = get_chat_model(temperature=0.1)
    creative_model = get_chat_model(temperature=0.9)

    precise_response = precise_model.invoke(prompt)
    creative_response = creative_model.invoke(prompt)

    print("=== LOW TEMPERATURE ===")
    print(get_message_text(precise_response))

    print("\n=== HIGH TEMPERATURE ===")
    print(get_message_text(creative_response))


def demonstrate_chat_messages() -> None:
    """Demonstrate system, human, and AI messages."""
    model = get_chat_model(temperature=0.2)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a concise book recommendation assistant. "
                "Recommend one book and explain the reason in no more "
                "than two sentences."
            )
        ),
        HumanMessage(
            content=(
                "I enjoy mystery novels with clever detectives."
            )
        ),
    ]

    first_response = model.invoke(messages)

    print("\n=== FIRST RESPONSE ===")
    print(get_message_text(first_response))

    # Preserve the original AIMessage in the conversation history.
    messages.append(first_response)

    messages.append(
        HumanMessage(
            content=(
                "Recommend another mystery novel, "
                "but this time the story must be set in Japan."
            )
        )
    )

    second_response = model.invoke(messages)

    print("\n=== FOLLOW-UP RESPONSE ===")
    print(get_message_text(second_response))


def main() -> None:
    compare_temperatures()
    demonstrate_chat_messages()


if __name__ == "__main__":
    main()