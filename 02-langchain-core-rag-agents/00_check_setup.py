"""Verify the Gemini API configuration."""

from common import get_chat_model, get_message_text


def main() -> None:
    model = get_chat_model(temperature=0.1)

    response = model.invoke(
        "Reply with exactly this text: Gemini connection works"
    )

    print(get_message_text(response))


if __name__ == "__main__":
    main()