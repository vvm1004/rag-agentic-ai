from __future__ import annotations

import numpy as np

from similarity_common import (
    DOCUMENTS,
    embed_texts_for_similarity,
    euclidean_distance,
    print_matrix,
)


def main() -> None:
    """Calculate only unique vector pairs and mirror the matrix."""
    embeddings = embed_texts_for_similarity(
        DOCUMENTS
    )
    count = embeddings.shape[0]

    distances = np.zeros(
        (count, count),
        dtype=np.float64,
    )

    calculations = 0

    for i in range(count):
        for j in range(i + 1, count):
            distance = euclidean_distance(
                embeddings[i],
                embeddings[j],
            )

            distances[i, j] = distance
            distances[j, i] = distance
            calculations += 1

    print("LESSON 03 — OPTIMIZE THE L2 MATRIX")
    print("=" * 72)

    print_matrix(
        distances,
        title="Optimized symmetric L2 matrix",
    )

    print(
        "\nNaive matrix cells:",
        count * count,
    )
    print(
        "Unique pair calculations:",
        calculations,
    )
    print(
        "Expected unique pairs:",
        count * (count - 1) // 2,
    )

    print(
        "\nKey idea: distance(a, b) equals distance(b, a), "
        "so each pair only needs to be calculated once."
    )


if __name__ == "__main__":
    main()
