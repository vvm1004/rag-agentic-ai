# 06 — PDF RAG QA Bot with Gradio and Gemini

This folder contains a fully functional, interactive web application that implements **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions about them through a user-friendly Gradio web interface, using Google's Gemini AI.

## Application Architecture

Below is the lifecycle of data in this application, from uploading a PDF to receiving an AI-generated answer:

```
[User Uploads PDF]
       │
       ▼
  [PyPDFLoader] ──────▶ [Recursive Text Splitter]
                              │
                              ▼
[In-Memory Chroma DB] ◀── [Gemini Embeddings]
       │
       ▼ (User asks a question)
[Similarity Search]
       │
       ▼ (Retrieve top 4 chunks)
[Grounded Prompt]
       │
       ▼ (Combine chunks + question)
[Gemini Chat Model]
       │
       ▼
 [Final Answer & Sources Displayed in UI]
```

## Directory Structure

| File | Description |
|---|---|
| `config.py` | Manages environment variables and application settings (API keys, chunk sizes, Gradio server configs). |
| `rag_service.py` | The "Brain" of the application. Encapsulates all RAG logic (`PdfRagService` class) including PDF validation, loading, splitting, in-memory embedding, and answer generation. |
| `qabot.py` | The Gradio web interface (Front-end). Handles file uploads, user inputs, and displays the AI's responses and source evidence. |
| `check_setup.py` | A utility script to verify your environment setup before running the application. |

## Key Features

- **Gradio Web Interface**: A clean, interactive UI for uploading files and chatting, complete with example questions.
- **In-Memory Vector Database**: Unlike previous lessons that saved Chroma databases to disk, this application builds the database in your system's RAM. It uses unique `index_id`s tied to user sessions. Data disappears when the app closes, ensuring privacy.
- **Thread Safety**: Uses Python `threading.Lock()` to handle multiple users uploading PDFs or asking questions concurrently without data collision.
- **Grounded Answers with Citations**: The application uses a strict prompt to prevent AI hallucination. It forces Gemini to answer *only* from the uploaded PDF and displays the exact source excerpts used.
- **Robust Error Handling**: Wraps core logic in `try-except` blocks to display user-friendly error messages in the UI (e.g., "PDF too large") instead of crashing the server.

## Installation and Usage

### 1. Environment Configuration (`.env`)
Ensure you have a `.env` file in the root directory of your project with the following essential variables:
```env
GOOGLE_API_KEY=your_real_google_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
```

Optional Gradio settings you can configure:
```env
RAG_MAX_PDF_MB=20
GRADIO_SERVER_PORT=7860
GRADIO_SHARE=0
```

### 2. Verify Setup
Run the setup checker to ensure your API keys and dependencies are ready:
```bash
uv run python 06-pdf-rag-gradio-app/check_setup.py
```

### 3. Start the Application
Launch the Gradio web server by running the main script:
```bash
uv run python 06-pdf-rag-gradio-app/qabot.py
```

### 4. Interact with the Bot
1. Open your web browser and navigate to `http://127.0.0.1:7860`.
2. Upload a readable PDF file (scanned image-only PDFs are not supported).
3. Click the **"Index PDF"** button and wait for the success message.
4. Type your question in the text box (or click one of the examples) and click **"Ask PDF"**.
5. Read the generated answer and verify it against the provided **Sources** below it.

## Limitations
- **No OCR Support**: The application relies on `PyPDFLoader`, which extracts text layers. It cannot read text embedded inside images or scanned documents.
- **Volatile Memory**: Because the Chroma database is stored in-memory, you must re-upload and re-index your PDF every time you restart the application.
