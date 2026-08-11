from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from similarity_common import (
    DOCUMENTS,
    embed_texts_for_similarity,
    euclidean_distance,
    print_matrix,
)


def main() -> None:
    """Build an all-pairs L2 distance matrix manually."""
    embeddings = embed_texts_for_similarity(
        DOCUMENTS
    )
    count = embeddings.shape[0]

    l2_manual = np.zeros(
        (count, count),
        dtype=np.float64,
    )

    for i in range(count):
        for j in range(count):
            l2_manual[i, j] = (
                euclidean_distance(
                    embeddings[i],
                    embeddings[j],
                )
            )

    print("LESSON 02 — EUCLIDEAN / L2 DISTANCE")
    print("=" * 72)
    print(
        "Rule: smaller distance means the vectors are closer."
    )

    print_matrix(
        l2_manual,
        title="Manual L2 distance matrix",
    )

    print(
        "\nDistance(document 0, document 1):",
        round(
            l2_manual[0, 1],
            6,
        ),
    )
    print(
        "Distance(document 1, document 0):",
        round(
            l2_manual[1, 0],
            6,
        ),
    )

    print(
        "\nDiagonal values:",
        np.diag(l2_manual),
    )

    l2_scipy = cdist(
        embeddings,
        embeddings,
        metric="euclidean",
    )

    print_matrix(
        l2_scipy,
        title="SciPy L2 distance matrix",
    )

    print(
        "\nManual result matches SciPy:",
        np.allclose(
            l2_manual,
            l2_scipy,
        ),
    )

    print(
        "\nKey ideas:\n"
        "- distance(a, a) = 0\n"
        "- distance(a, b) = distance(b, a)\n"
        "- smaller L2 distance means closer vectors"
    )


if __name__ == "__main__":
    main()
