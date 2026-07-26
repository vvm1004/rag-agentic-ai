from __future__ import annotations

import argparse

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


DEFAULT_QUESTION = (
    "What is the mobile device policy?"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Answer one question with retrieved context."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
    )

    return parser


def main() -> None:
    """Retrieve context, build a prompt, and call Gemini."""
    args = build_parser().parse_args()

    vector_store = load_vector_store()
    documents = vector_store.similarity_search(
        args.question,
        k=settings.top_k,
    )

    context = format_context(documents)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Answer the user's question using the "
                    "supplied document context. Be clear and "
                    "concise."
                ),
            ),
            (
                "human",
                (
                    "Context:\n{context}\n\n"
                    "Question:\n{question}"
                ),
            ),
        ]
    )

    chain = (
        prompt
        | get_chat_model()
        | StrOutputParser()
    )

    answer = chain.invoke(
        {
            "context": context,
            "question": args.question,
        }
    )

    print("LESSON 06 — RETRIEVAL QA")
    print("Question:", args.question)

    print("\nRetrieved context")
    print("-" * 72)
    print(context)

    print("\nGenerated answer")
    print("-" * 72)
    print(answer.strip())

    print(
        "\nKey idea: RAG retrieves evidence first, then "
        "places that evidence inside the LLM prompt."
    )


if __name__ == "__main__":
    main()
