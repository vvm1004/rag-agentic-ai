# 01 — Prompt Engineering & LangChain Templates

This module introduces fundamental Prompt Engineering techniques using **LangChain** and **Google Gemini** to demonstrate how to write effective prompts for large language models (LLMs).

## Directory Structure

| File | Description |
|---|---|
| `llm_gemini.py` | Shared module — initializes a Gemini chat model from environment variables |
| `01_zero_shot.py` | **Zero-shot prompting** — classify movie review sentiment without any examples |
| `02_one_shot.py` | **One-shot prompting** — explain a technical concept using 1 example |
| `03_few_shot.py` | **Few-shot prompting** — classify sentence emotions using multiple examples |
| `04_reasoning.py` | **Chain-of-Thought** — ask the AI to reason step-by-step before answering |
| `05_review_analyzer.py` | **Capstone exercise** — analyze multiple product reviews (sentiment, features, summary) |

## Prompting Techniques

### Zero-shot
Ask the AI a question directly without providing any examples. Best suited for simple, well-defined tasks.

### One-shot
Provide **1 example** so the AI understands the desired response format and style.

### Few-shot
Provide **multiple examples** (3–5) so the AI can more accurately capture classification patterns.

### Chain-of-Thought (Reasoning)
Instruct the AI to **explain its reasoning step-by-step** before giving the final answer. This improves accuracy on logic and math problems.

## Usage

```bash
# Run from the project root directory
uv run python 01-prompt-engineering-langchain-templates/01_zero_shot.py
uv run python 01-prompt-engineering-langchain-templates/02_one_shot.py
uv run python 01-prompt-engineering-langchain-templates/03_few_shot.py
uv run python 01-prompt-engineering-langchain-templates/04_reasoning.py
uv run python 01-prompt-engineering-langchain-templates/05_review_analyzer.py
```

## Requirements

- Python ≥ 3.12
- A `.env` file at the project root containing `GOOGLE_API_KEY`
