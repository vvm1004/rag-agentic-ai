from __future__ import annotations

from rag_common import (
    load_documents,
    print_document,
    settings,
)


def main() -> None:
    """Load the source file and inspect the LangChain Document."""
    documents = load_documents()

    print("LESSON 01 — LOAD DOCUMENT")
    print("Source path:", settings.source_path)
    print("Document objects:", len(documents))

    for index, document in enumerate(
        documents,
        start=1,
    ):
        print_document(
            document,
            number=index,
            max_characters=1200,
        )

    print(
        "\nKey idea: TextLoader converts a text file into "
        "LangChain Document objects with content and metadata."
    )


if __name__ == "__main__":
    main()
