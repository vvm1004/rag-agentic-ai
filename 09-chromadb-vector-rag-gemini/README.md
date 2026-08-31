# 09 — ChromaDB Vector Search & RAG with Google Gemini

This folder provides a complete, hands-on guide to building vector search, metadata-filtered semantic retrieval, and conversational Retrieval-Augmented Generation (RAG) applications using **ChromaDB** and the **Google Gemini API** (`gemini-3.1-flash-lite` / `gemini-embedding-2-preview`).

---

## 🏗️ Architecture & Workflow

```text
                               ┌───────────────────────────┐
                               │  Raw Data / JSON Records   │
                               │    (Foods / Employees)    │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Google Gemini Embeddings  │
                               │ (gemini-embedding-2-prev) │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    ChromaDB Vector Store  │
                               │   (Cosine Distance HNSW)  │
                               └─────────────┬─────────────┘
                                             │
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
      [Lesson 01 & 02]               [Lesson 03]                    [Lesson 04]
┌──────────────────────────┐   ┌──────────────────────────┐   ┌──────────────────────────┐
│   Semantic Retrieval &   │   │ Conversational RAG Bot   │   │ Search System Benchmark  │
│    Metadata Filtering    │   │ (ChromaDB + Gemini LLM)  │   │ (Latency & Capabilities) │
│ ($gte, $in, $and compound)│   │  Context Injection Flow  │   │  Interactive vs Filtered │
└──────────────────────────┘   └──────────────────────────┘   └──────────────────────────┘
```

---

## 📁 Directory Structure

| File | Description |
|---|---|
| `shared_functions.py` (or `chroma_common.py`) | Shared utilities: environment loading, `GeminiEmbeddingFunction` for ChromaDB, dataset loading, and `GeminiModelInference` wrapper. |
| `FoodDataSet.json` | Comprehensive dataset of 20 diverse culinary dishes with nutritional details, ingredients, cuisine types, and health benefits. |
| `00_check_setup.py` | Environment diagnostic script: validates Python dependencies, API keys, ChromaDB integration, and dataset loading. |
| `01_basic_chroma_grocery.py` | Basic ChromaDB collection creation, text ingestion, and cosine distance semantic similarity search. |
| `02_employee_metadata_search.py` | Advanced ChromaDB search: exact metadata filtering (`$gte`, `$in`) and combined semantic + compound `$and` filters. |
| `03_food_rag_chatbot.py` | Enhanced conversational food recommendation chatbot powered by ChromaDB retrieval and Google Gemini generation. Includes comparison mode and help menu. |
| `04_compare_search_systems.py` | Benchmark script comparing Interactive Search, Advanced Filtered Search, and Gemini RAG Chatbot response times and trade-offs. |

---

## 🚀 Key Concepts Covered

1. **ChromaDB Custom Embedding Function**:
   - Integrating Google Gemini's `gemini-embedding-2-preview` with ChromaDB's native collection API.
   - Vector similarity in cosine distance space.

2. **Metadata Filtering**:
   - Filtering by categorical attributes (`where={"department": "Engineering"}`).
   - Numerical comparison operators (`where={"experience": {"$gte": 10}}`).
   - Set membership queries (`where={"location": {"$in": ["San Francisco", "Los Angeles"]}}`).
   - Combined semantic + compound boolean queries with `$and`.

3. **Retrieval-Augmented Generation (RAG)**:
   - Constructing structured context from vector search results.
   - Grounding Gemini prompts to prevent hallucination.
   - Conversational fallback handling and side-by-side preference comparison.

4. **Search System Trade-offs**:
   - **Interactive Similarity Search**: Fast (~0.4s), direct, but lacks natural language explanation.
   - **Advanced Filtered Search**: Combines semantic meaning with hard business rules (~0.7s).
   - **RAG Chatbot**: Rich, personalized conversational recommendations with reasoning (~2.5s).

---

## 🛠️ Installation & Setup

### 1. Environment Configuration (`.env`)

Ensure your `.env` file in the project root includes your Google API key:

```env
GOOGLE_API_KEY=your_real_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
GEMINI_EMBEDDING_DIMENSIONS=768
```

### 2. Verify Prerequisites

Run the setup validator:

```bash
uv run python 09-chromadb-vector-rag-gemini/00_check_setup.py
```

### 3. Run the Lessons Sequentially

#### Lesson 01: Basic Grocery Search
```bash
uv run python 09-chromadb-vector-rag-gemini/01_basic_chroma_grocery.py
```

#### Lesson 02: Employee Search with Metadata Filtering
```bash
uv run python 09-chromadb-vector-rag-gemini/02_employee_metadata_search.py
```

#### Lesson 03: Interactive Food Recommendation RAG Chatbot
```bash
uv run python 09-chromadb-vector-rag-gemini/03_food_rag_chatbot.py
```
*Try interactive commands in the prompt:*
- `I want a high protein meal for post-workout recovery`
- `compare` (compares two dietary preferences side-by-side)
- `help`
- `quit`

#### Lesson 04: Compare Search Systems Benchmark
```bash
uv run python 09-chromadb-vector-rag-gemini/04_compare_search_systems.py
```
