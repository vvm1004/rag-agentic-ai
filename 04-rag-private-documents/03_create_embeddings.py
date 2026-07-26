from __future__ import annotations

from rag_common import (
    cosine_similarity,
    get_embeddings,
    settings,
    split_documents,
)


def main() -> None:
    """Embed sample chunks and compare their semantic similarity."""
    chunks = split_documents()
    embeddings = get_embeddings()

    sample_texts = [
        chunks[0].page_content,
        chunks[-1].page_content,
    ]

    vectors = embeddings.embed_documents(
        sample_texts
    )

    mobile_question = (
        "What rules apply to company mobile devices?"
    )
    incident_question = (
        "Which events must employees report?"
    )

    question_vectors = embeddings.embed_documents(
        [
            mobile_question,
            incident_question,
        ]
    )

    print("LESSON 03 — CREATE EMBEDDINGS")
    print(
        "Embedding model:",
        settings.embedding_model,
    )
    print(
        "Vector dimensions:",
        len(vectors[0]),
    )
    print(
        "First 10 values:",
        vectors[0][:10],
    )

    print("\nSemantic comparisons")
    print("-" * 72)

    print(
        "Mobile question vs first chunk:",
        round(
            cosine_similarity(
                question_vectors[0],
                vectors[0],
            ),
            4,
        ),
    )

    print(
        "Incident question vs first chunk:",
        round(
            cosine_similarity(
                question_vectors[1],
                vectors[0],
            ),
            4,
        ),
    )

    print(
        "Mobile question vs last chunk:",
        round(
            cosine_similarity(
                question_vectors[0],
                vectors[1],
            ),
            4,
        ),
    )

    print(
        "Incident question vs last chunk:",
        round(
            cosine_similarity(
                question_vectors[1],
                vectors[1],
            ),
            4,
        ),
    )

    print(
        "\nKey idea: an embedding model does not answer "
        "questions. It creates vectors used for semantic search."
    )


if __name__ == "__main__":
    main()
