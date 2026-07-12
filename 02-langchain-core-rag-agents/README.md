# 02 — LangChain Core, RAG & Agents

This module covers the core components of **LangChain**, progressing from basic chat model usage to building a complete **RAG (Retrieval-Augmented Generation)** pipeline and a tool-calling **Agent**.

## Directory Structure

### Lessons

| File | Topic |
|---|---|
| `00_check_setup.py` | **Setup Check** — verify API key and installed dependencies |
| `01_models_messages.py` | **Models & Messages** — compare temperature settings, use SystemMessage / HumanMessage / AIMessage |
| `02_prompt_templates_lcel.py` | **Prompt Templates & LCEL** — create reusable prompt templates, use the pipe (`\|`) operator |
| `03_structured_output.py` | **Structured Output** — force the AI to return structured data (JSON) using Pydantic |
| `04_documents_splitters.py` | **Document Loaders & Splitters** — load documents from multiple sources, split text into chunks |
| `05_embeddings_vectorstore.py` | **Embeddings & Vector Store** — convert text to vectors, perform semantic similarity search |
| `06_rag.py` | **RAG Pipeline** — question answering grounded in retrieved documents |
| `07_memory.py` | **Conversation Memory** — persist chat history (manually and with Checkpointer) |
| `08_multi_step_chain.py` | **Multi-step Chain** — multi-stage processing with `RunnablePassthrough.assign()` |
| `09_tools.py` | **Tools** — create and invoke tools for AI to use |
| `10_agent.py` | **Agent** — build an autonomous agent that selects tools and remembers conversations |

### Shared Modules

| File | Description |
|---|---|
| `common.py` | Initialize Gemini chat model and embedding model, extract text from responses |
| `tools_lib.py` | Tool definitions: arithmetic calculator and text formatter |
| `vectorstore_utils.py` | Load documents, split into chunks, and create a Chroma vector store |
| `data/langchain_notes.txt` | Source document used for RAG |

## Learning Path

```
01 Models & Messages
 └──▸ 02 Prompt Templates & LCEL
       └──▸ 03 Structured Output
             └──▸ 04 Document Loaders & Splitters
                   └──▸ 05 Embeddings & Vector Store
                         └──▸ 06 RAG Pipeline
                               └──▸ 07 Conversation Memory
                                     └──▸ 08 Multi-step Chain
                                           └──▸ 09 Tools
                                                 └──▸ 10 Agent
```

## Usage

```bash
# Verify environment setup
uv run python 02-langchain-core-rag-agents/00_check_setup.py

# Run each lesson in order
uv run python 02-langchain-core-rag-agents/01_models_messages.py
uv run python 02-langchain-core-rag-agents/02_prompt_templates_lcel.py
uv run python 02-langchain-core-rag-agents/03_structured_output.py
uv run python 02-langchain-core-rag-agents/04_documents_splitters.py
uv run python 02-langchain-core-rag-agents/05_embeddings_vectorstore.py
uv run python 02-langchain-core-rag-agents/06_rag.py
uv run python 02-langchain-core-rag-agents/07_memory.py
uv run python 02-langchain-core-rag-agents/08_multi_step_chain.py
uv run python 02-langchain-core-rag-agents/09_tools.py
uv run python 02-langchain-core-rag-agents/10_agent.py
```

### Optional Arguments

```bash
# Lesson 4: Load additional PDF or web page
uv run python 02-langchain-core-rag-agents/04_documents_splitters.py --pdf path/to/file.pdf
uv run python 02-langchain-core-rag-agents/04_documents_splitters.py --url "https://example.com"

# Lesson 5: Custom search query
uv run python 02-langchain-core-rag-agents/05_embeddings_vectorstore.py --query "What is LCEL?"

# Lesson 6: Custom RAG question
uv run python 02-langchain-core-rag-agents/06_rag.py --question "What is a retriever?"

# Lesson 10: Interactive agent chat
uv run python 02-langchain-core-rag-agents/10_agent.py --interactive
```

## Requirements

- Python ≥ 3.12
- A `.env` file at the project root containing `GOOGLE_API_KEY`
