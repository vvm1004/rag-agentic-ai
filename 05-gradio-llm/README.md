# 05 — Gradio & LLM Web Interfaces

This module introduces **Gradio**, a fast and intuitive Python library for building web interfaces, and demonstrates how to seamlessly integrate it with the **Gemini API** via LangChain. 

By following these lessons, you will transition from running simple scripts in the terminal to building a fully functional, conversational AI chatbot with a modern web UI.

## Directory Structure

| File | Lesson / Description |
|---|---|
| `llm_common.py` | Shared module — initializes a Gemini chat model, loads environment variables, and provides reusable API calling functions. |
| `00_check_setup.py` | Validates your `.env` configuration and environment setup before starting the lessons. |
| `01_sum_calculator.py` | **Lesson 1:** The absolute basics of Gradio. Maps a simple Python math function to a web interface without AI. |
| `02_combine_sentences.py` | **Lesson 2:** Introduces text processing. Shows how Gradio handles text boxes and string arguments. |
| `03_common_input_types.py` | **Lesson 3:** Explores advanced UI components (Sliders, Checkboxes, Radio buttons, and Dropdowns). |
| `04_llm_gradio_interface.py` | **Lesson 4:** Combines AI with a Web UI. Creates a single-turn Q&A interface where the user types a prompt and receives a Gemini response. |
| `05_max_tokens_exercise.py` | **Lesson 5:** Demonstrates LLM output limits (`max_tokens`) and how Gradio can return **multiple outputs** simultaneously. |
| `06_bonus_chatbot.py` | **Bonus:** Uses `gr.ChatInterface` to build a complete multi-turn conversational chatbot with memory, passing conversation history back to the model. |

## Usage

You can run any of the lessons from the root of the project using `uv`. Make sure you have activated your virtual environment.

```bash
# Example: Run the simple calculator
uv run python 05-gradio-llm/01_sum_calculator.py

# Example: Run the final Chatbot
uv run python 05-gradio-llm/06_bonus_chatbot.py
```

After running a script, look for the local URL in your terminal (usually `http://127.0.0.1:7860`) and open it in your web browser. 

> **Important:** Always press `Ctrl+C` in your terminal to stop the current server before running the next lesson to prevent port conflicts (`Cannot find empty port in range: 7860-7860`).

## Requirements

- Python ≥ 3.12
- A `.env` file at the project root containing `GOOGLE_API_KEY`
- The `gradio` and `langchain-google-genai` packages installed.
