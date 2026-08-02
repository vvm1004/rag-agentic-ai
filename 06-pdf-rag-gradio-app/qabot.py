"""Gradio interface for PDF question answering with Gemini RAG."""

from __future__ import annotations

import gradio as gr

from config import settings
from rag_service import (
    PdfRagService,
    RagError,
)

rag_service = PdfRagService()


def index_pdf(
    file_path: str | None,
) -> tuple[str | None, str]:
    """Index the uploaded PDF and return a reusable index ID."""
    try:
        return rag_service.index_pdf(
            file_path
        )
    except RagError as error:
        return (
            None,
            f"❌ **Indexing failed:** {error}",
        )
    except Exception as error:
        return (
            None,
            (
                "❌ **Unexpected indexing error:** "
                f"{type(error).__name__}: {error}"
            ),
        )


def ask_pdf(
    index_id: str | None,
    question: str,
) -> tuple[str, str]:
    """Answer one question from the indexed PDF."""
    try:
        return rag_service.answer_question(
            index_id,
            question,
        )
    except RagError as error:
        return (
            f"❌ **Question failed:** {error}",
            "",
        )
    except Exception as error:
        return (
            (
                "❌ **Unexpected question error:** "
                f"{type(error).__name__}: {error}"
            ),
            "",
        )


def clear_application(
    index_id: str | None,
) -> tuple[
    None,
    None,
    str,
    str,
    str,
    str,
]:
    """Delete the current index and reset the interface."""
    rag_service.remove_index(
        index_id
    )

    return (
        None,
        None,
        "Upload a PDF and click **Index PDF**.",
        "",
        "",
        "",
    )


with gr.Blocks(
    title="PDF RAG Question Answering",
) as demo:
    index_state = gr.State(
        value=None
    )

    gr.Markdown(
        """
# PDF RAG Question-Answering Bot

Upload a readable PDF, build its vector index once, then ask multiple
questions about its contents.

```text
PDF → loader → chunks → Gemini embeddings → Chroma
    → retrieval → Gemini → grounded answer
```
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_file = gr.File(
                label="Upload PDF",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
            )

            with gr.Row():
                index_button = gr.Button(
                    "Index PDF",
                    variant="primary",
                )

                clear_button = gr.Button(
                    "Clear",
                    variant="secondary",
                )

            index_status = gr.Markdown(
                "Upload a PDF and click **Index PDF**."
            )

            gr.Markdown(
                f"""
### Current configuration

- Chat model: `{settings.chat_model}`
- Embedding model: `{settings.embedding_model}`
- Chunk size: `{settings.chunk_size}`
- Chunk overlap: `{settings.chunk_overlap}`
- Retrieved chunks: `{settings.top_k}`
- Maximum PDF size: `{settings.max_pdf_mb} MB`
"""
            )

        with gr.Column(scale=2):
            question = gr.Textbox(
                label="Question",
                lines=3,
                placeholder=(
                    "Ask a question about the indexed PDF..."
                ),
            )

            ask_button = gr.Button(
                "Ask PDF",
                variant="primary",
            )

            gr.Markdown("## Answer")
            answer = gr.Markdown()

            gr.Markdown("## Sources")
            sources = gr.Markdown()

            gr.Examples(
                examples=[
                    ["Summarize the main points of this PDF."],
                    ["What rules or requirements are mentioned?"],
                    ["What does the document say about exceptions?"],
                ],
                inputs=question,
            )

    gr.Markdown(
        """
### Important

- The answer is generated only from retrieved PDF text.
- Scanned image-only PDFs require OCR and are not supported here.
- The vector index is stored in application memory and disappears when the
  process stops.
- PDF chunks are sent to the Gemini API for embeddings and answer generation.
"""
    )

    index_button.click(
        fn=index_pdf,
        inputs=pdf_file,
        outputs=[
            index_state,
            index_status,
        ],
    )

    ask_button.click(
        fn=ask_pdf,
        inputs=[
            index_state,
            question,
        ],
        outputs=[
            answer,
            sources,
        ],
    )

    question.submit(
        fn=ask_pdf,
        inputs=[
            index_state,
            question,
        ],
        outputs=[
            answer,
            sources,
        ],
    )

    clear_button.click(
        fn=clear_application,
        inputs=index_state,
        outputs=[
            pdf_file,
            index_state,
            index_status,
            question,
            answer,
            sources,
        ],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=settings.server_name,
        server_port=settings.server_port,
        share=settings.share,
    )
