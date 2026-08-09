"""Command-line version of the LlamaIndex Icebreaker Bot."""

from __future__ import annotations

import argparse

from modules.data_extraction import (
    ProfileDataError,
    extract_linkedin_profile,
)
from modules.data_processing import (
    create_vector_database,
    split_profile_data,
    verify_embeddings,
)
from modules.query_engine import (
    answer_user_query,
    generate_initial_facts,
)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a LlamaIndex RAG index from a professional "
            "profile and generate networking icebreakers."
        )
    )

    parser.add_argument(
        "--json",
        dest="json_path",
        help=(
            "Optional local JSON profile. If omitted, the included "
            "mock profile is used."
        ),
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional Gemini model ID. If omitted, GEMINI_MODEL "
            "from the shared .env is used."
        ),
    )

    return parser


def main() -> None:
    """Run the complete profile-processing and Q&A workflow."""
    args = build_parser().parse_args()

    try:
        profile = extract_linkedin_profile(
            args.json_path,
            mock=args.json_path is None,
        )

        nodes = split_profile_data(
            profile
        )

        print("Creating vector index...")
        index = create_vector_database(
            nodes
        )

        if not verify_embeddings(index):
            raise RuntimeError(
                "The vector index could not be verified."
            )

        print("\nGenerating initial icebreakers...\n")

        initial = generate_initial_facts(
            index,
            model_name=args.model,
        )

        print(initial.answer)
        print(
            "\n" + "=" * 72
        )
        print(
            "Ask questions about this profile."
        )
        print(
            "Commands: exit, quit, bye"
        )
        print(
            "=" * 72
        )

        while True:
            question = input("\nYou: ").strip()

            if not question:
                continue

            if question.lower() in {
                "exit",
                "quit",
                "bye",
            }:
                print("Bot: Goodbye!")
                break

            result = answer_user_query(
                index,
                question,
                model_name=args.model,
            )

            print(
                "\nBot:\n"
                f"{result.answer}"
            )

    except (
        ProfileDataError,
        RuntimeError,
    ) as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
