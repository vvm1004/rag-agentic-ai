from __future__ import annotations

import numpy as np
import torch

from similarity_common import (
    DOCUMENTS,
    dot_product,
    embed_texts_for_similarity,
    normalize_rows,
    print_matrix,
)


def main() -> None:
    """Show that dot product of normalized vectors equals cosine similarity."""
    embeddings = embed_texts_for_similarity(
        DOCUMENTS
    )

    manual_normalized = normalize_rows(
        embeddings
    )

    torch_normalized = (
        torch.nn.functional.normalize(
            torch.from_numpy(
                embeddings
            ),
            p=2,
            dim=1,
        )
        .numpy()
    )

    manual_norms = np.sqrt(
        np.sum(
            manual_normalized ** 2,
            axis=1,
        )
    )

    count = embeddings.shape[0]

    cosine_manual = np.empty(
        (count, count),
        dtype=np.float64,
    )

    for i in range(count):
        for j in range(count):
            cosine_manual[i, j] = (
                dot_product(
                    manual_normalized[i],
                    manual_normalized[j],
                )
            )

    cosine_operator = (
        manual_normalized
        @ manual_normalized.T
    )

    cosine_distance = (
        1 - cosine_operator
    )

    print("LESSON 05 — COSINE SIMILARITY")
    print("=" * 72)

    print(
        "L2 norm of each normalized vector:"
    )
    print(manual_norms)

    print(
        "\nManual normalization matches PyTorch:",
        np.allclose(
            manual_normalized,
            torch_normalized,
            rtol=1e-5,
            atol=1e-6,
        ),
    )

    print_matrix(
        cosine_manual,
        title="Manual cosine-similarity matrix",
    )

    print_matrix(
        cosine_operator,
        title=(
            "Cosine via normalized_embeddings "
            "@ normalized_embeddings.T"
        ),
    )

    print(
        "\nManual cosine matches matrix multiplication:",
        np.allclose(
            cosine_manual,
            cosine_operator,
            rtol=1e-5,
            atol=1e-6,
        ),
    )

    print_matrix(
        cosine_distance,
        title="Cosine distance = 1 - cosine similarity",
    )

    print(
        "\nKey ideas:\n"
        "- normalized vector length is approximately 1\n"
        "- dot product of normalized vectors equals cosine similarity\n"
        "- cosine similarity closer to 1 means more similar\n"
        "- cosine distance closer to 0 means more similar"
    )


if __name__ == "__main__":
    main()
