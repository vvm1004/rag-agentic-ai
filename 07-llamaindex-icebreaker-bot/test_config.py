"""Small configuration test inspired by the original lab."""

from __future__ import annotations

import icebreaker_config as config


print(
    "Initial Facts Template defined:",
    bool(config.INITIAL_FACTS_TEMPLATE),
)
print(
    "User Question Template defined:",
    bool(config.USER_QUESTION_TEMPLATE),
)
print(
    "Chunk Size:",
    config.settings.chunk_size,
)
print(
    "Similarity Top K:",
    config.settings.similarity_top_k,
)
