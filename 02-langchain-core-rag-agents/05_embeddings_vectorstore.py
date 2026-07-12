import argparse

from common import get_embeddings
from vectorstore_utils import (
    build_vector_store,
    load_chunks,
)


def display_embedding_example() -> None:
    """Generate and inspect one embedding vector."""
    chunks = load_chunks()
    embeddings = get_embeddings()

    vector = embeddings.embed_query(
        chunks[0].page_content
    )

    print("=== EMBEDDING EXAMPLE ===")
    print("First five values:", vector[:5])
    print("Vector dimensions:", len(vector))


def display_similarity_search(
    query: str,
) -> None:
    """Search the vector store for semantically similar chunks."""
    vector_store, _ = build_vector_store()

    results = (
        vector_store
        .similarity_search_with_score(
            query,
            k=3,
        )
    )

    print("\n=== SIMILARITY SEARCH ===")
    print("Query:", query)

    for index, result in enumerate(
        results,
        start=1,
    ):
        document, score = result

        print(
            f"\nResult {index} "
            f"— score: {score:.4f}"
        )

        print(
            document.page_content[:350]
        )

        print(
            "Metadata:",
            document.metadata,
        )


def display_retriever_example(
    query: str,
) -> None:
    """Use the standard retriever interface."""
    vector_store, _ = build_vector_store()

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3,
        }
    )

    documents = retriever.invoke(query)

    print("\n=== RETRIEVER RESULT ===")
    print(
        f"Retrieved {len(documents)} documents."
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print(f"\nDocument {index}:")
        print(
            document.page_content[:250]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate embeddings and semantic retrieval."
        )
    )

    parser.add_argument(
        "--query",
        default=(
            "How does LangChain retrieve "
            "relevant information?"
        ),
    )

    arguments = parser.parse_args()

    display_embedding_example()

    display_similarity_search(
        arguments.query
    )

    display_retriever_example(
        arguments.query
    )


if __name__ == "__main__":
    main()