"""Split profile data into nodes and create a LlamaIndex vector index."""

from __future__ import annotations

import json
from typing import Any

from llama_index.core import (
    Document,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter

import icebreaker_config as config
from modules.llm_interface import (
    create_google_embedding,
)


def split_profile_data(
    profile_data: dict[str, Any],
) -> list:
    """Convert a profile dictionary to JSON text and split it into nodes."""
    profile_json = json.dumps(
        profile_data,
        ensure_ascii=False,
        indent=2,
    )

    document = Document(
        text=profile_json,
        metadata={
            "source": "professional_profile",
        },
    )

    splitter = SentenceSplitter(
        chunk_size=config.settings.chunk_size,
        chunk_overlap=config.settings.chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(
        [document]
    )

    if not nodes:
        raise RuntimeError(
            "No nodes were created from the profile data."
        )

    return nodes


def create_vector_database(
    nodes: list,
) -> VectorStoreIndex:
    """Create an in-memory VectorStoreIndex from profile nodes."""
    if not nodes:
        raise RuntimeError(
            "Cannot create a vector index from an empty node list."
        )

    return VectorStoreIndex(
        nodes=nodes,
        embed_model=create_google_embedding(),
        show_progress=False,
    )


def verify_embeddings(
    index: VectorStoreIndex,
) -> bool:
    """Verify indexing by performing a small semantic retrieval."""
    try:
        retriever = index.as_retriever(
            similarity_top_k=1
        )
        results = retriever.retrieve(
            "professional experience education skills"
        )

        return bool(results)
    except Exception:
        return False
