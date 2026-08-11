from __future__ import annotations

import numpy as np

from similarity_common import (
    DEFAULT_QUERY,
    DOCUMENTS,
    embed_documents_for_search,
    embed_query_for_search,
    normalize_rows,
)


def _print_ranking(
    title: str,
    scores: np.ndarray,
    *,
    larger_is_better: bool,
) -> None:
    order = np.argsort(
        scores
    )

    if larger_is_better:
        order = order[::-1]

    print("\n" + title)
    print("=" * 72)

    for rank, index in enumerate(
        order,
        start=1,
    ):
        index = int(index)

        print(
            f"{rank}. doc {index} "
            f"score={float(scores[index]):.6f}"
        )
        print(
            f"   {DOCUMENTS[index]}"
        )


def main() -> None:
    """Rank the same query using three vector-comparison metrics."""
    documents = embed_documents_for_search(
        DOCUMENTS
    )

    query = embed_query_for_search(
        DEFAULT_QUERY
    )[0]

    l2_scores = np.sqrt(
        np.sum(
            (documents - query) ** 2,
            axis=1,
        )
    )

    dot_scores = (
        documents
        @ query
    )

    normalized_documents = normalize_rows(
        documents
    )

    normalized_query = normalize_rows(
        query.reshape(1, -1)
    )[0]

    cosine_scores = (
        normalized_documents
        @ normalized_query
    )

    print("LESSON 07 — COMPARE METRICS")
    print("=" * 72)
    print("Query:")
    print(DEFAULT_QUERY)

    _print_ranking(
        "L2 DISTANCE — smaller is better",
        l2_scores,
        larger_is_better=False,
    )

    _print_ranking(
        "DOT PRODUCT — larger is better",
        dot_scores,
        larger_is_better=True,
    )

    _print_ranking(
        "COSINE SIMILARITY — larger is better",
        cosine_scores,
        larger_is_better=True,
    )

    print(
        "\nKey idea: the embedding model creates vector space, "
        "while L2, dot product, and cosine define how we compare "
        "positions in that space."
    )


if __name__ == "__main__":
    main()
