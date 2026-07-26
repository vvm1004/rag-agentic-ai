from __future__ import annotations

from itertools import islice

from langchain_core.output_parsers import (
    StrOutputParser,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
)

from rag_common import (
    get_chat_model,
    split_documents,
)


def batched(
    values: list[str],
    size: int,
):
    """Yield fixed-size lists from a sequence."""
    iterator = iter(values)

    while True:
        group = list(
            islice(iterator, size)
        )

        if not group:
            break

        yield group


def main() -> None:
    """Summarize all chunks, then recursively combine summaries."""
    chunks = split_documents()
    model = get_chat_model()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Summarize this document chunk accurately. "
                    "Preserve important rules, exceptions, "
                    "obligations, prohibited actions, and reporting "
                    "requirements. Do not add facts."
                ),
            ),
            (
                "human",
                (
                    "Chunk {chunk_number}:\n\n"
                    "{content}"
                ),
            ),
        ]
    )

    map_chain = (
        map_prompt
        | model
        | StrOutputParser()
    )

    partial_summaries: list[str] = []

    print("LESSON 09 — SUMMARIZE ALL CHUNKS")
    print("Chunks:", len(chunks))
    print("\nMap phase")

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        summary = map_chain.invoke(
            {
                "chunk_number": index,
                "content": chunk.page_content,
            }
        ).strip()

        partial_summaries.append(summary)

        print("\n" + "-" * 72)
        print(f"Chunk {index} summary")
        print("-" * 72)
        print(summary)

    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "Combine the supplied partial summaries into "
                    "one coherent document summary. Organize the "
                    "result with short headings and bullet points. "
                    "Remove repetition, preserve important details, "
                    "and do not invent facts."
                ),
            ),
            (
                "human",
                (
                    "Partial summaries:\n\n"
                    "{summaries}"
                ),
            ),
        ]
    )

    reduce_chain = (
        reduce_prompt
        | model
        | StrOutputParser()
    )

    current_level = partial_summaries
    reduction_round = 1

    while len(current_level) > 1:
        next_level: list[str] = []

        print(
            f"\nReduce round {reduction_round}: "
            f"{len(current_level)} inputs"
        )

        for group in batched(
            current_level,
            size=6,
        ):
            combined = reduce_chain.invoke(
                {
                    "summaries": (
                        "\n\n---\n\n".join(
                            group
                        )
                    )
                }
            ).strip()

            next_level.append(combined)

        current_level = next_level
        reduction_round += 1

    final_summary = (
        current_level[0]
        if current_level
        else "No content was available."
    )

    print("\n" + "=" * 72)
    print("FINAL DOCUMENT SUMMARY")
    print("=" * 72)
    print(final_summary)

    print(
        "\nKey idea: full-document summarization processes every "
        "chunk. A normal retrieval query may return only a few "
        "similar chunks and therefore miss sections."
    )


if __name__ == "__main__":
    main()
