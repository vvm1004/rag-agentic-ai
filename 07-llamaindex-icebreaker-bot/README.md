# 07 — LlamaIndex AI Icebreaker Bot with Gemini

This folder contains an interactive networking assistant application built with **LlamaIndex** and **Google Gemini**. It ingests professional profile data from local JSON files, processes and indexes the data into an in-memory vector store, generates personalized conversation starters (icebreakers), and answers detailed profile questions via a **Gradio Web Interface** or a **Command-Line Interface (CLI)**.

---

## Application Architecture

The diagram below outlines the flow of data through the LlamaIndex RAG pipeline—from raw profile JSON extraction to semantic retrieval and answer generation:

```text
[JSON Profile Data (Mock / Uploaded)]
               │
               ▼
      [LlamaIndex Document]
               │
               ▼
       [SentenceSplitter]
               │
               ▼
            [Nodes]
               │
               ▼
[In-Memory VectorStoreIndex] ◀── [GoogleGenAIEmbedding]
               │
               ▼
  [Retriever (Top K Nodes)]
               │
               ▼
[Strict Grounded Prompt Template]
               │
               ▼
     [GoogleGenAI (Gemini)]
               │
               ▼
 [Icebreakers / Q&A Answers + Retrieved Evidence]
```

---

## Directory Structure

| File / Folder | Description |
|---|---|
| `icebreaker_config.py` | Central configuration loading `.env` settings, chunk parameters, server options, and strict prompt templates. |
| `app.py` | The Gradio web interface featuring profile processing, state management, icebreaker generation, and Q&A tabs. |
| `main.py` | Terminal-based CLI version for processing profile data and running interactive Q&A sessions. |
| `check_setup.py` | Utility script to verify package dependencies and configuration settings without making API calls. |
| `data/mock_linkedin_profile.json` | Sample fictional professional profile dataset for out-of-the-box testing. |
| `modules/data_extraction.py` | Handles reading, parsing, and recursively cleaning empty values from profile JSON files. |
| `modules/data_processing.py` | Converts cleaned profile dictionaries into `Document` objects, splits them into nodes, and builds/verifies the `VectorStoreIndex`. |
| `modules/llm_interface.py` | Initializes `GoogleGenAI` LLM and `GoogleGenAIEmbedding` instances configured with Gemini API credentials. |
| `modules/query_engine.py` | Manages semantic retrieval (`similarity_top_k`), prompt construction, icebreaker generation, and grounded Q&A with similarity-scored sources. |

---

## Key Features

- **LlamaIndex RAG Framework**: Demonstrates essential LlamaIndex patterns: document creation, node parsing with `SentenceSplitter`, and vector indexing via `VectorStoreIndex`.
- **Dual Interfaces**: Choose between an interactive **Gradio Web UI** and a terminal-based **CLI tool**.
- **Automated Icebreaker Generation**: Analyzes profile nodes to construct 3 interesting, fact-grounded conversation starters paired with relevant questions.
- **Strict Anti-Hallucination Prompting**: Forces Gemini to answer exclusively from retrieved profile nodes. Returns *"I don't know based on the available profile data"* if information is absent.
- **Source Evidence & Similarity Scores**: Exposes retrieved profile nodes along with cosine similarity scores in expandable accordions for clear transparency.
- **Flexible Profile Support**: Load the built-in mock profile or upload custom structured JSON files.
- **Thread-Safe In-Memory Sessions**: Isolates multi-user vector indices in RAM using unique session UUIDs and Python `threading.Lock`.

---

## Installation and Usage

### 1. Environment Configuration (`.env`)

Ensure you have a `.env` file located in the root directory of the repository with your Google Gemini API key and desired models:

```env
GOOGLE_API_KEY=your_real_google_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2
```

### 2. Verify Setup

Run the setup checker to validate your environment, environment variables, and installed packages:

```bash
uv run python 07-llamaindex-icebreaker-bot/check_setup.py
```

### 3. Option A: Launch the Gradio Web Application

Start the web application:

```bash
uv run python 07-llamaindex-icebreaker-bot/app.py
```

1. Open your browser and navigate to `http://127.0.0.1:5002`.
2. In **Tab 1 ("1. Process Profile")**, select **"Use included mock profile"** (or upload your own JSON profile) and click **"Process Profile"**.
3. View the generated **3 Icebreakers** and inspect the **Retrieved profile nodes**.
4. Switch to **Tab 2 ("2. Ask Questions")**, enter a question (or select an example), and click **"Ask"**.

### 4. Option B: Run the Command-Line Interface (CLI)

Run the CLI directly in your terminal:

```bash
# Using the included mock profile
uv run python 07-llamaindex-icebreaker-bot/main.py

# Or specifying a custom profile JSON path and alternative Gemini model
uv run python 07-llamaindex-icebreaker-bot/main.py --json path/to/profile.json --model gemini-2.5-flash
```

Type your questions interactively in the prompt line, or type `exit`, `quit`, or `bye` to terminate the session.

---

## Custom Configuration Options

You can adjust indexing and server defaults by adding these optional variables to your `.env` file:

```env
ICEBREAKER_CHUNK_SIZE=500
ICEBREAKER_CHUNK_OVERLAP=50
ICEBREAKER_TOP_K=5
ICEBREAKER_SERVER_NAME=127.0.0.1
ICEBREAKER_SERVER_PORT=5002
ICEBREAKER_SHARE=0
```

---

## Limitations

- **No Live API Scraping**: Modern social platforms like LinkedIn restrict automated web scraping and third-party profile APIs (e.g., Proxycurl) have been discontinued. This project intentionally uses local JSON data for educational purposes.
- **Volatile Vector Index**: Profile indices are generated and stored in RAM (`VectorStoreIndex`), meaning data resets whenever the application process restarts.
