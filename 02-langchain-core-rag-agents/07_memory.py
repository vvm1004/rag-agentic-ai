from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from common import (
    get_chat_model,
    get_message_text,
)


def demonstrate_manual_history() -> None:
    """Preserve conversation history manually."""
    model = get_chat_model(temperature=0.2)

    messages: list[BaseMessage] = [
        SystemMessage(
            content=(
                "You are a concise personal assistant."
            )
        ),
        HumanMessage(
            content=(
                "My name is Alice. "
                "My favorite color is blue, "
                "and I enjoy mountain hiking."
            )
        ),
    ]

    first_response = model.invoke(messages)

    messages.append(first_response)

    messages.append(
        HumanMessage(
            content=(
                "What are my name, favorite color, "
                "and favorite activity?"
            )
        )
    )

    second_response = model.invoke(messages)

    print("=== MANUAL MESSAGE HISTORY ===")
    print(
        get_message_text(second_response)
    )


def demonstrate_checkpointer_memory() -> None:
    """Preserve conversation state with an agent checkpointer."""
    agent = create_agent(
        model=get_chat_model(temperature=0.2),
        tools=[],
        system_prompt=(
            "You are a concise personal assistant."
        ),
        checkpointer=InMemorySaver(),
    )

    thread_id = str(uuid4())

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "My name is Alice. "
                        "My favorite color is blue, "
                        "and I enjoy mountain hiking."
                    ),
                }
            ]
        },
        config=config,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What are my name, favorite color, "
                        "and favorite activity?"
                    ),
                }
            ]
        },
        config=config,
    )

    final_message = result["messages"][-1]

    print("\n=== CHECKPOINTER MEMORY ===")
    print(
        get_message_text(final_message)
    )


def main() -> None:
    demonstrate_manual_history()
    demonstrate_checkpointer_memory()


if __name__ == "__main__":
    main()