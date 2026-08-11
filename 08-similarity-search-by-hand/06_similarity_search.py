from __future__ import annotations

import argparse

import numpy as np

from similarity_common import (
    DEFAULT_QUERY,
    DOCUMENTS,
    embed_documents_for_search,
    embed_query_for_search,
    normalize_rows,
    print_documents,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Embed one query with Gemini and rank the four documents "
            "using cosine similarity calculated by NumPy."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Semantic search query.",
    )

    return parser


def main() -> None:
    """Rank documents by cosine similarity to a Gemini query embedding."""
    args = build_parser().parse_args()

    document_embeddings = (
        embed_documents_for_search(
            DOCUMENTS
        )
    )

    query_embedding = (
        embed_query_for_search(
            args.query
        )
    )

    normalized_documents = normalize_rows(
        document_embeddings
    )
    normalized_query = normalize_rows(
        query_embedding
    )

    cosine_scores = (
        normalized_documents
        @ normalized_query.T
    ).reshape(-1)

    ranking = np.argsort(
        cosine_scores
    )[::-1]

    best_index = int(
        cosine_scores.argmax()
    )

    print("LESSON 06 — SIMILARITY SEARCH BY HAND")
    print("=" * 72)
    print("Query:")
    print(args.query)
    print()

    print_documents()

    print("\nCOSINE SCORES")
    print("=" * 72)

    for index, score in enumerate(
        cosine_scores
    ):
        print(
            f"Document {index}: {float(score):.6f}"
        )

    print("\nRANKING")
    print("=" * 72)

    for rank, index in enumerate(
        ranking,
        start=1,
    ):
        index = int(index)

        print(
            f"{rank}. Document {index} "
            f"(score={float(cosine_scores[index]):.6f})"
        )
        print(
            f"   {DOCUMENTS[index]}"
        )

    print("\nBEST MATCH")
    print("=" * 72)
    print(
        f"Document {best_index}:"
    )
    print(
        DOCUMENTS[best_index]
    )

    print(
        "\nKey idea: Gemini only creates the vectors. "
        "The similarity calculation, ranking, and argmax are "
        "still performed manually in this script."
    )


if __name__ == "__main__":
    main()
