from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)

from rag_common import (
    format_context,
    get_chat_model,
    load_vector_store,
    settings,
)


@dataclass(frozen=True)
class ChatTurn:
    """One question and answer stored in memory."""

    question: str
    answer: str


def format_history(
    history: deque[ChatTurn],
) -> str:
    """Convert chat history into prompt text."""
    if not history:
        return "(No previous conversation.)"

    lines: list[str] = []

    for turn in history:
        lines.append(
            f"User: {turn.question}"
        )
        lines.append(
            f"Assistant: {turn.answer}"
        )

    return "\n".join(lines)


def rewrite_question(
    question: str,
    history: deque[ChatTurn],
) -> str:
    """Resolve pronouns and create a standalone retrieval query."""
    if not history:
        return question

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Rewrite the latest question as a standalone "
                    "semantic-search query. Resolve pronouns such "
                    "as it, that, this policy, or those rules "
                    "using the conversation history. Do not answer "
                    "the question. Return only the rewritten query "
                    "in the user's language."
                ),
            ),
            (
                "human",
                (
                    "Conversation history:\n{history}\n\n"
                    "Latest question:\n{question}"
                ),
            ),
        ]
    )

    chain = (
        prompt
        | get_chat_model()
        | StrOutputParser()
    )

    rewritten = chain.invoke(
        {
            "history": format_history(
                history
            ),
            "question": question,
        }
    ).strip()

    return rewritten or question


def answer_question(
    question: str,
    standalone_question: str,
    history: deque[ChatTurn],
) -> str:
    """Retrieve evidence and generate a grounded answer."""
    vector_store = load_vector_store()

    documents = vector_store.similarity_search(
        standalone_question,
        k=settings.top_k,
    )

    context = format_context(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Answer only from the supplied document "
                    "context. Do not use outside knowledge or "
                    "make assumptions. If evidence is insufficient, "
                    "say that the documents do not contain enough "
                    "information. Cite evidence with labels such as "
                    "[Document 1]. Answer in the user's language."
                ),
            ),
            (
                "human",
                (
                    "Conversation history:\n{history}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Current question:\n{question}"
                ),
            ),
        ]
    )

    chain = (
        prompt
        | get_chat_model()
        | StrOutputParser()
    )

    return chain.invoke(
        {
            "history": format_history(
                history
            ),
            "context": context,
            "question": question,
        }
    ).strip()


def main() -> None:
    """Run an interactive conversational RAG loop."""
    history: deque[ChatTurn] = deque(
        maxlen=6
    )

    print("LESSON 08 — CONVERSATIONAL RAG")
    print(
        "Ask a question, then use follow-ups such as:\n"
        "  What is the mobile device policy?\n"
        "  What must employees report in it?\n\n"
        "Commands: /history, /reset, /exit\n"
    )

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {
            "/exit",
            "exit",
            "quit",
            "bye",
        }:
            print("Assistant: Goodbye!")
            break

        if question == "/reset":
            history.clear()
            print(
                "Assistant: Conversation history cleared."
            )
            continue

        if question == "/history":
            print("\n" + format_history(history) + "\n")
            continue

        standalone_question = rewrite_question(
            question,
            history,
        )

        answer = answer_question(
            question,
            standalone_question,
            history,
        )

        history.append(
            ChatTurn(
                question=question,
                answer=answer,
            )
        )

        print(
            "\nStandalone retrieval query:\n"
            f"{standalone_question}"
        )
        print(
            "\nAssistant:\n"
            f"{answer}\n"
        )

    print(
        "\nKey idea: conversation memory and document "
        "retrieval are different. History resolves the question; "
        "the vector store supplies factual evidence."
    )


if __name__ == "__main__":
    main()
