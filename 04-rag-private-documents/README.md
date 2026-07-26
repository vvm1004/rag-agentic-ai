# 04 — RAG Private Documents (Step by Step)

This folder contains a sequence of small, incremental learning scripts designed to help you understand the internal mechanics of a **RAG (Retrieval-Augmented Generation)** system.

## Learning Path

Below is the lifecycle of data in a RAG system, from a raw text file to the final AI-generated answer:

```
[Raw Document]
      │
      ▼
   [Chunks] ─────────────┐
      │                  │
      ▼                  ▼
[Embeddings]   [Full-Document Summarization]
      │               (Map-Reduce)
      ▼
[(Chroma Vector Store)]
      │
      ▼
[Semantic Retrieval]
      │
      ▼
[Retrieval QA]
      │
      ▼
[Grounded Prompt]
      │
      ▼
[Conversational RAG]
```

## Directory Structure

| File | Lesson / Functionality |
|---|---|
| `rag_common.py` | Contains shared configurations and helper functions used across all lessons. |
| `00_check_setup.py` | Validates `.env` configuration, file paths, and Chroma database status. |
| `01_load_document.py` | **Lesson 1:** Reads a `.txt` file and converts it into LangChain Document objects. |
| `02_split_document.py` | **Lesson 2:** Splits long documents into smaller, overlapping chunks. |
| `03_create_embeddings.py` | **Lesson 3:** Uses Google API to convert text into numerical vectors and compares semantic similarity using cosine distance. |
| `04_build_vector_store.py` | **Lesson 4:** Embeds all chunks and persists them into a local Chroma database. **(Must run before Lessons 5-8)** |
| `05_similarity_search.py` | **Lesson 5:** Retrieves chunks that are semantically closest to a user query (without using an LLM to answer). |
| `06_retrieval_qa.py` | **Lesson 6:** Combines the retrieved context with Gemini to generate a complete answer. |
| `07_grounded_prompt.py` | **Lesson 7:** Uses a strict prompt to prevent hallucination, forcing the AI to admit when it lacks information. |
| `08_conversational_rag.py` | **Lesson 8:** Builds a complete Conversational RAG chatbot with memory, capable of resolving follow-up questions. |
| `09_summarize_document.py` | **Lesson 9:** Uses the Map-Reduce algorithm to summarize the entire document, overcoming context window limits. |

## Installation and Usage

### 1. Environment Configuration (`.env`)
Ensure you have a `.env` file in the root directory of your project with the following variables:
```env
GOOGLE_API_KEY=your_real_google_api_key
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
```

### 2. Initialize the Database
You **MUST** run Lesson 04 to build the Chroma database before you can execute lessons 05 through 08.
```bash
uv run python 04-rag-private-documents/04_build_vector_store.py
```
*(If you wish to test with your own text file, overwrite `data/company_policies.txt` and rerun Lesson 04 to update the database).*

### 3. Run the Lessons
You can run each lesson sequentially to see the output in your terminal. For example, to run the complete conversational chatbot in Lesson 08:
```bash
uv run python 04-rag-private-documents/08_conversational_rag.py
```

## Key Concepts to Remember
- **Embeddings are not answers:** They are simply numerical representations used for Semantic Search. The AI Chat Model is what generates the textual answer.
- **RAG is not Fine-tuning:** RAG does not alter the underlying knowledge or weights of the AI model. RAG is essentially providing context: it retrieves relevant documents and forces the AI to read and answer based *only* on that context via the prompt.
- **Vector Store Memory vs. Chat Memory:** Chroma stores "Knowledge" (the documents), while Chat History stores "Your recent conversation context". They serve two different purposes in a conversational RAG system.
