from __future__ import annotations

import numpy as np

from similarity_common import (
    DOCUMENTS,
    dot_product,
    embed_texts_for_similarity,
    print_matrix,
)


def main() -> None:
    """Compare manual dot products with matrix multiplication."""
    embeddings = embed_texts_for_similarity(
        DOCUMENTS
    )
    count = embeddings.shape[0]

    dot_manual = np.empty(
        (count, count),
        dtype=np.float64,
    )

    for i in range(count):
        for j in range(count):
            dot_manual[i, j] = (
                dot_product(
                    embeddings[i],
                    embeddings[j],
                )
            )

    dot_operator = (
        embeddings
        @ embeddings.T
    )

    dot_matmul = np.matmul(
        embeddings,
        embeddings.T,
    )

    dot_np = np.dot(
        embeddings,
        embeddings.T,
    )

    print("LESSON 04 — DOT PRODUCT")
    print("=" * 72)
    print(
        "Rule: larger dot product means greater similarity "
        "for this ranking method."
    )

    print_matrix(
        dot_manual,
        title="Manual dot-product matrix",
    )

    print_matrix(
        dot_operator,
        title="embeddings @ embeddings.T",
    )

    print(
        "\nManual approximately matches @:",
        np.allclose(
            dot_manual,
            dot_operator,
            rtol=1e-5,
            atol=1e-6,
        ),
    )
    print(
        "np.matmul matches @:",
        np.allclose(
            dot_matmul,
            dot_operator,
        ),
    )
    print(
        "np.dot matches @:",
        np.allclose(
            dot_np,
            dot_operator,
        ),
    )

    print_matrix(
        -dot_operator,
        title="Dot-product distance = -similarity",
    )

    print(
        "\nKey idea: a document embedding matrix multiplied "
        "by its transpose produces an all-pairs dot-product matrix."
    )


if __name__ == "__main__":
    main()
