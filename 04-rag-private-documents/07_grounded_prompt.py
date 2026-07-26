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
    "Can employees eat inside company vehicles?"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask a question using a strict grounded prompt."
        )
    )

    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
    )

    return parser


def main() -> None:
    """Answer only when retrieved documents support the answer."""
    args = build_parser().parse_args()

    vector_store = load_vector_store()
    documents = vector_store.similarity_search(
        args.question,
        k=settings.top_k,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You answer questions only from the "
                    "provided private-document context.\n\n"
                    "Rules:\n"
                    "- Do not use outside knowledge.\n"
                    "- Do not make assumptions.\n"
                    "- If the context does not contain enough "
                    "information, say exactly that the documents "
                    "do not contain enough information.\n"
                    "- Cite evidence with labels such as "
                    "[Document 1].\n"
                    "- Answer in the same language as the user."
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

    context = format_context(documents)

    answer = chain.invoke(
        {
            "context": context,
            "question": args.question,
        }
    )

    print("LESSON 07 — GROUNDED PROMPT")
    print("Question:", args.question)

    print("\nAnswer")
    print("-" * 72)
    print(answer.strip())

    print("\nRetrieved evidence")
    print("-" * 72)
    print(context)

    print(
        "\nKey idea: retrieval alone does not eliminate "
        "hallucination. The prompt must tell the model how to "
        "behave when evidence is insufficient."
    )


if __name__ == "__main__":
    main()
