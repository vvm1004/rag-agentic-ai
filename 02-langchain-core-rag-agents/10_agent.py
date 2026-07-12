from __future__ import annotations

import argparse
from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from common import (
    get_chat_model,
    get_message_text,
)
from tools_lib import (
    calculator,
    format_text,
)


def build_agent():
    """Create an agent with calculation and text formatting tools."""
    return create_agent(
        model=get_chat_model(temperature=0.1),
        tools=[
            calculator,
            format_text,
        ],
        system_prompt=(
            "You are a concise assistant. "
            "Use the calculator tool for arithmetic requests. "
            "Use the format_text tool for uppercase, lowercase, "
            "or title case requests. "
            "After using a tool, briefly explain the result."
        ),
        checkpointer=InMemorySaver(),
    )


def ask_agent(
    agent,
    question: str,
    config: dict,
) -> None:
    """Send one user question to the agent."""
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        },
        config=config,
    )

    final_message = result["messages"][-1]

    print(
        get_message_text(final_message)
    )


def run_single_question(
    question: str,
) -> None:
    """Run one question and exit."""
    agent = build_agent()

    config = {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }

    print("Question:", question)
    print("Answer:", end=" ")

    ask_agent(
        agent=agent,
        question=question,
        config=config,
    )


def run_interactive_chat() -> None:
    """Run an interactive agent session."""
    agent = build_agent()

    config = {
        "configurable": {
            "thread_id": str(uuid4()),
        }
    }

    print(
        "Interactive agent started. "
        "Type exit, quit, or bye to stop."
    )

    while True:
        question = input("\nYou: ").strip()

        if question.lower() in {
            "exit",
            "quit",
            "bye",
        }:
            print("Agent session ended.")
            break

        if not question:
            continue

        print("Agent:", end=" ")

        ask_agent(
            agent=agent,
            question=question,
            config=config,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a Gemini agent with arithmetic "
            "and text formatting tools."
        )
    )

    parser.add_argument(
        "--question",
        default="Calculate (25 + 63) * 4",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive agent session.",
    )

    arguments = parser.parse_args()

    if arguments.interactive:
        run_interactive_chat()
        return

    run_single_question(
        arguments.question
    )


if __name__ == "__main__":
    main()