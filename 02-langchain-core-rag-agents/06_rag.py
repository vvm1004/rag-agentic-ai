from __future__ import annotations

import argparse

from langchain_core.documents import Document
from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)
from langchain_core.runnables import (
    RunnablePassthrough,
)

from common import get_chat_model
from vectorstore_utils import (
    build_vector_store,
)


def format_documents(
    documents: list[Document],
) -> str:
    """Convert retrieved documents into prompt context."""
    formatted_documents: list[str] = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        source = document.metadata.get(
            "source",
            "unknown",
        )

        formatted_documents.append(
            (
                f"[Document {index}]\n"
                f"Source: {source}\n"
                f"{document.page_content}"
            )
        )

    return "\n\n---\n\n".join(
        formatted_documents
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Answer questions using a local RAG pipeline."
        )
    )

    parser.add_argument(
        "--question",
        default=(
            "What is the difference "
            "between a chain and an agent?"
        ),
    )

    arguments = parser.parse_args()

    vector_store, _ = build_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 4,
        }
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Answer the question only from the supplied context. "
                    "If the context does not contain the answer, reply: "
                    "'I do not know based on the provided documents.' "
                    "Keep the answer concise."
                ),
            ),
            (
                "human",
                (
                    "Context:\n"
                    "{context}\n\n"
                    "Question:\n"
                    "{question}"
                ),
            ),
        ]
    )

    rag_chain = (
        {
            "context": (
                retriever
                | format_documents
            ),
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_chat_model(temperature=0.1)
        | StrOutputParser()
    )

    answer = rag_chain.invoke(
        arguments.question
    )

    retrieved_documents = retriever.invoke(
        arguments.question
    )

    print("=== QUESTION ===")
    print(arguments.question)

    print("\n=== RAG ANSWER ===")
    print(answer)

    print("\n=== RETRIEVED DOCUMENTS ===")

    for index, document in enumerate(
        retrieved_documents,
        start=1,
    ):
        print(f"\nDocument {index}:")
        print(
            document.page_content[:300]
        )


if __name__ == "__main__":
    main()