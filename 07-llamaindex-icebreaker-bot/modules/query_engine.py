"""Generate icebreakers and answer questions from an indexed profile."""

from __future__ import annotations

from dataclasses import dataclass

from llama_index.core import VectorStoreIndex

import icebreaker_config as config
from modules.llm_interface import (
    create_google_llm,
)


@dataclass(frozen=True)
class QueryResult:
    """Answer text plus readable retrieval evidence."""

    answer: str
    sources: str


def _retrieve_context(
    index: VectorStoreIndex,
    query: str,
) -> tuple[str, str]:
    retriever = index.as_retriever(
        similarity_top_k=(
            config.settings.similarity_top_k
        )
    )

    source_nodes = retriever.retrieve(
        query
    )

    if not source_nodes:
        return (
            "",
            "No relevant profile nodes were retrieved.",
        )

    context_parts: list[str] = []
    source_parts: list[str] = []

    for position, result in enumerate(
        source_nodes,
        start=1,
    ):
        text = result.node.get_content().strip()

        context_parts.append(
            f"[Profile Node {position}]\n{text}"
        )

        score = getattr(
            result,
            "score",
            None,
        )

        score_text = (
            f"{score:.4f}"
            if isinstance(score, (int, float))
            else "n/a"
        )

        excerpt = " ".join(
            text.split()
        )[:500]

        source_parts.append(
            (
                f"**Node {position}** "
                f"(similarity score: `{score_text}`)\n\n"
                f"> {excerpt}"
            )
        )

    return (
        "\n\n".join(context_parts),
        "\n\n".join(source_parts),
    )


def generate_initial_facts(
    index: VectorStoreIndex,
    *,
    model_name: str | None = None,
) -> QueryResult:
    """Generate three profile-grounded networking icebreakers."""
    retrieval_query = (
        "career education projects skills achievements interests "
        "professional experience"
    )

    context, sources = _retrieve_context(
        index,
        retrieval_query,
    )

    if not context:
        return QueryResult(
            answer="No relevant profile data was found.",
            sources=sources,
        )

    prompt = config.INITIAL_FACTS_TEMPLATE.format(
        context_str=context
    )

    llm = create_google_llm(
        model_name=model_name
    )
    response = llm.complete(prompt)

    return QueryResult(
        answer=response.text.strip(),
        sources=sources,
    )


def answer_user_query(
    index: VectorStoreIndex,
    user_query: str,
    *,
    model_name: str | None = None,
) -> QueryResult:
    """Answer a user question using retrieved profile context only."""
    cleaned_query = user_query.strip()

    if not cleaned_query:
        return QueryResult(
            answer="Please enter a question.",
            sources="",
        )

    context, sources = _retrieve_context(
        index,
        cleaned_query,
    )

    if not context:
        return QueryResult(
            answer=(
                "I don't know based on the available profile data."
            ),
            sources=sources,
        )

    prompt = config.USER_QUESTION_TEMPLATE.format(
        context_str=context,
        query_str=cleaned_query,
    )

    llm = create_google_llm(
        model_name=model_name
    )
    response = llm.complete(prompt)

    return QueryResult(
        answer=response.text.strip(),
        sources=sources,
    )
