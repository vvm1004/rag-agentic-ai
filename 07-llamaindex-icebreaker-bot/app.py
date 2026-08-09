"""Gradio web application for the LlamaIndex Icebreaker Bot."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass

import gradio as gr
from llama_index.core import VectorStoreIndex

import icebreaker_config as config
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


@dataclass
class ProfileSession:
    """One processed profile stored for the current app process."""

    index: VectorStoreIndex
    display_name: str
    model_name: str | None


_active_sessions: dict[str, ProfileSession] = {}
_sessions_lock = threading.Lock()


def _profile_display_name(
    profile: dict,
) -> str:
    value = profile.get(
        "full_name",
        "Professional profile",
    )

    return str(value)


def process_profile(
    source_choice: str,
    json_path: str | None,
    model_name: str,
) -> tuple[str | None, str, str, str]:
    """Load profile data, build an index, and generate 3 icebreakers."""
    try:
        use_mock = (
            source_choice
            == "Use included mock profile"
        )

        profile = extract_linkedin_profile(
            json_path,
            mock=use_mock,
        )

        nodes = split_profile_data(
            profile
        )

        index = create_vector_database(
            nodes
        )

        if not verify_embeddings(index):
            raise RuntimeError(
                "The vector index was created but retrieval verification failed."
            )

        selected_model = (
            model_name.strip()
            or config.settings.llm_model
        )

        initial = generate_initial_facts(
            index,
            model_name=selected_model,
        )

        session_id = str(
            uuid.uuid4()
        )

        display_name = _profile_display_name(
            profile
        )

        with _sessions_lock:
            _active_sessions[
                session_id
            ] = ProfileSession(
                index=index,
                display_name=display_name,
                model_name=selected_model,
            )

        status = (
            f"✅ Processed **{display_name}** successfully.\n\n"
            f"- Nodes created: **{len(nodes)}**\n"
            f"- Embedding model: **{config.settings.embedding_model}**\n"
            f"- LLM: **{selected_model}**\n"
            f"- Retriever top K: **{config.settings.similarity_top_k}**\n\n"
            "You can now open the **Ask Questions** tab."
        )

        return (
            session_id,
            status,
            initial.answer,
            initial.sources,
        )

    except (
        ProfileDataError,
        RuntimeError,
        ValueError,
    ) as error:
        return (
            None,
            f"❌ **Processing failed:** {error}",
            "",
            "",
        )

    except Exception as error:
        return (
            None,
            (
                "❌ **Unexpected error:** "
                f"{type(error).__name__}: {error}"
            ),
            "",
            "",
        )


def ask_profile(
    session_id: str | None,
    question: str,
) -> tuple[str, str]:
    """Answer one question using the processed profile index."""
    if not session_id:
        return (
            "Process a profile first.",
            "",
        )

    with _sessions_lock:
        session = _active_sessions.get(
            session_id
        )

    if session is None:
        return (
            (
                "This profile session is no longer available. "
                "Process the profile again."
            ),
            "",
        )

    try:
        result = answer_user_query(
            session.index,
            question,
            model_name=session.model_name,
        )

        return (
            result.answer,
            result.sources,
        )

    except Exception as error:
        return (
            (
                "Question failed: "
                f"{type(error).__name__}: {error}"
            ),
            "",
        )


def clear_session(
    session_id: str | None,
) -> tuple[
    None,
    str,
    str,
    str,
    str,
]:
    """Remove the current session and clear generated outputs."""
    if session_id:
        with _sessions_lock:
            _active_sessions.pop(
                session_id,
                None,
            )

    return (
        None,
        "No profile has been processed yet.",
        "",
        "",
        "",
    )


with gr.Blocks(
    title="AI Icebreaker Bot with LlamaIndex",
) as demo:
    session_state = gr.State(
        value=None
    )

    gr.Markdown(
        """
# AI Icebreaker Bot — LlamaIndex + Gemini

This VSCode version follows the lab's RAG idea using local mock/profile JSON
data instead of the discontinued LinkedIn API path.

```text
Profile JSON
→ LlamaIndex Document
→ SentenceSplitter
→ Nodes
→ Gemini Embeddings
→ VectorStoreIndex
→ Retriever
→ Gemini
→ Icebreakers / Q&A
```
"""
    )

    with gr.Tab("1. Process Profile"):
        source_choice = gr.Radio(
            choices=[
                "Use included mock profile",
                "Upload local JSON profile",
            ],
            value="Use included mock profile",
            label="Data source",
        )

        json_file = gr.File(
            label="Optional profile JSON",
            file_types=[".json"],
            file_count="single",
            type="filepath",
        )

        model_name = gr.Textbox(
            label="Gemini model",
            value=config.settings.llm_model,
            info=(
                "Leave the shared .env model here, or enter another "
                "Gemini model ID available to your API key."
            ),
        )

        with gr.Row():
            process_button = gr.Button(
                "Process Profile",
                variant="primary",
            )
            clear_button = gr.Button(
                "Clear Session",
            )

        process_status = gr.Markdown(
            "No profile has been processed yet."
        )

        gr.Markdown("## Three personalized icebreakers")
        initial_facts = gr.Markdown()

        with gr.Accordion(
            "Retrieved profile nodes",
            open=False,
        ):
            initial_sources = gr.Markdown()

    with gr.Tab("2. Ask Questions"):
        question = gr.Textbox(
            label="Question about the profile",
            lines=3,
            placeholder=(
                "Example: What projects could I ask this person about?"
            ),
        )

        ask_button = gr.Button(
            "Ask",
            variant="primary",
        )

        gr.Markdown("## Answer")
        answer = gr.Markdown()

        with gr.Accordion(
            "Retrieved evidence",
            open=False,
        ):
            answer_sources = gr.Markdown()

        gr.Examples(
            examples=[
                ["What is this person's current role?"],
                ["What did this person study?"],
                ["What AI projects has this person worked on?"],
                ["Suggest a networking question about their interests."],
                ["What is this person's salary?"],
            ],
            inputs=question,
        )

    gr.Markdown(
        f"""
### Local configuration

- Chat model: `{config.settings.llm_model}`
- Embedding model: `{config.settings.embedding_model}`
- Chunk size: `{config.settings.chunk_size}`
- Chunk overlap: `{config.settings.chunk_overlap}`
- Similarity top K: `{config.settings.similarity_top_k}`
- Local port: `{config.settings.server_port}`

The mock profile is fictional and included only for learning.
"""
    )

    process_button.click(
        fn=process_profile,
        inputs=[
            source_choice,
            json_file,
            model_name,
        ],
        outputs=[
            session_state,
            process_status,
            initial_facts,
            initial_sources,
        ],
    )

    ask_button.click(
        fn=ask_profile,
        inputs=[
            session_state,
            question,
        ],
        outputs=[
            answer,
            answer_sources,
        ],
    )

    question.submit(
        fn=ask_profile,
        inputs=[
            session_state,
            question,
        ],
        outputs=[
            answer,
            answer_sources,
        ],
    )

    clear_button.click(
        fn=clear_session,
        inputs=session_state,
        outputs=[
            session_state,
            process_status,
            initial_facts,
            answer,
            answer_sources,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=config.settings.server_name,
        server_port=config.settings.server_port,
        share=config.settings.share,
    )
