from __future__ import annotations

import argparse

from rag_common import (
    load_vector_store,
    settings,
)


DEFAULT_QUERY = (
    "What is the company mobile device policy?"
)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Search the persisted Chroma index."
        )
    )

    parser.add_argument(
        "query",
        nargs="?",
        default=DEFAULT_QUERY,
        help="Question or semantic search query.",
    )

    return parser


def main() -> None:
    """Search Chroma and print ranked chunks."""
    args = build_parser().parse_args()
    vector_store = load_vector_store()

    results = (
        vector_store.similarity_search_with_score(
            args.query,
            k=settings.top_k,
        )
    )

    print("LESSON 05 — SIMILARITY SEARCH")
    print("Query:", args.query)
    print("Top K:", settings.top_k)
    print(
        "\nNote: Chroma returns a distance score. "
        "Smaller values generally indicate closer matches."
    )

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):
        print("\n" + "=" * 72)
        print(
            f"RESULT {rank} | distance={distance:.6f}"
        )
        print("=" * 72)
        print("Metadata:", document.metadata)
        print(document.page_content.strip())

    print(
        "\nKey idea: retrieval returns evidence. "
        "It does not yet generate a final answer."
    )


if __name__ == "__main__":
    main()
