# 08 — Similarity Search by Hand with Gemini

This folder contains a comprehensive, step-by-step tutorial on how Vector Search and Semantic Retrieval work under the hood. Instead of relying on a black-box Vector Database (like Pinecone or Chroma), you will use the **Google Gemini API** to generate text embeddings, and then manually compute L2 distance, dot product, and cosine similarity using pure **Python** and **NumPy**.

By stripping away the abstraction layers, you will learn the exact mathematical principles that power modern Retrieval-Augmented Generation (RAG) systems.

---

## Tutorial Architecture & Flow

The diagram below outlines the learning progression—from raw text ingestion to mathematical similarity scoring and final document ranking:

```text
[Raw Text Documents]
         │
         ▼
[Google Gemini API] ──▶ (Generates 768-dimensional embeddings)
         │
         ▼
[NumPy Vector Matrix]
         │
         ├──▶ Lesson 02: L2 Distance (Euclidean)
         ├──▶ Lesson 03: Optimized Symmetric Matrices
         ├──▶ Lesson 04: Dot Product
         ├──▶ Lesson 05: Cosine Similarity
         │
         ▼
[Similarity Search & Ranking] ──▶ (Lessons 06 & 07)
```

---

## Directory Structure

| File | Description |
|---|---|
| `similarity_common.py` | Shared configurations, API client setup, dummy text corpus (`DOCUMENTS`), and base mathematical functions. |
| `00_check_setup.py` | Validates environment variables and required package installations without calling the Gemini API. |
| `01_create_embeddings.py` | Passes text to Gemini to retrieve embeddings and inspects the output vector shapes. |
| `02_l2_distance.py` | Manually calculates Euclidean distance using nested loops and compares it against SciPy's implementation. |
| `03_l2_optimized.py` | Optimizes the symmetric L2 matrix calculation to reduce computations by roughly 50%. |
| `04_dot_product.py` | Explains dot product and introduces `NumPy` matrix multiplication for lightning-fast vectorization. |
| `05_cosine_similarity.py` | Normalizes vectors to calculate cosine similarity via dot product. |
| `06_similarity_search.py` | Simulates a real RAG retrieval step: taking a query, computing cosine similarity against the corpus, and ranking results. |
| `07_compare_metrics.py` | Demonstrates that for Gemini's length-normalized embeddings, L2, dot product, and cosine similarity all yield the exact same ranking. |

---

## Key Learning Outcomes

- **Embeddings**: Understand how Large Language Models convert semantic meaning into coordinate geometry (vectors).
- **Vectorization**: Learn why matrix multiplication (the `@` operator in Python) is vastly superior to `for` loops for calculating all-pairs similarity.
- **Normalization Magic**: Discover why, when vectors are normalized to a length of 1, the dot product mathematically equals cosine similarity—making large-scale retrieval highly efficient.
- **Demystifying RAG**: Realize that finding the "most relevant context" in RAG is fundamentally just a mathematical sort operation on cosine scores.

---

## Installation and Usage

### 1. Environment Configuration (`.env`)

Ensure you have a `.env` file located in the root directory of the repository with your Google Gemini API key:

```env
GOOGLE_API_KEY=your_real_google_api_key
GEMINI_EMBEDDING_MODEL=gemini-embedding-2-preview
GEMINI_EMBEDDING_DIMENSIONS=768
```

### 2. Install Dependencies

Install the required Python packages (such as `numpy`, `scipy`, `google-genai`):

```bash
uv add -r 08-similarity-search-by-hand/requirements.txt
```

### 3. Verify Setup

Run the setup checker to validate your environment and installed packages:

```bash
uv run python 08-similarity-search-by-hand/00_check_setup.py
```

### 4. Run the Lessons Sequentially

Execute each lesson one by one to follow the tutorial flow. Notice how each script builds on the mathematical foundations of the previous one:

```bash
uv run python 08-similarity-search-by-hand/01_create_embeddings.py
uv run python 08-similarity-search-by-hand/02_l2_distance.py
uv run python 08-similarity-search-by-hand/03_l2_optimized.py
uv run python 08-similarity-search-by-hand/04_dot_product.py
uv run python 08-similarity-search-by-hand/05_cosine_similarity.py
uv run python 08-similarity-search-by-hand/06_similarity_search.py
uv run python 08-similarity-search-by-hand/07_compare_metrics.py
```

*(Optional)* You can pass a custom query to Lesson 06 to test semantic retrieval yourself:
```bash
uv run python 08-similarity-search-by-hand/06_similarity_search.py "Tell me about software testing"
```
