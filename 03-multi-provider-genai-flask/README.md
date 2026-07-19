# 03 — Multi-Provider GenAI Flask App

This module demonstrates how to build a robust **Flask** web application integrated with **LangChain** to analyze customer support messages. It supports dynamically switching between multiple AI providers (Google Gemini, Groq, Mistral, NVIDIA, HuggingFace, and OpenRouter) through a unified interface.

## Directory Structure

| File/Directory | Description |
|---|---|
| `app.py` | Main Flask application, containing routing, validation, and API endpoints. |
| `ai_service.py` | AI service layer. Defines the Pydantic schema for structured JSON output and constructs the LCEL chain. |
| `model_factory.py` | Factory pattern implementation to configure and instantiate LangChain ChatModels dynamically based on the user's selected provider. |
| `provider_test.py` | A quick test script to verify API connections for all configured providers. |
| `templates/` | HTML templates for the frontend web interface. |
| `static/` | CSS and JavaScript files for frontend interactivity and styling. |

## Application Architecture

```
[Web Browser / UI]
       │
       ▼ (POST /generate)
[app.py - Flask Route]
       │
       ▼ (Validate Request)
[ai_service.py - AI Logic] ──(Request ChatModel)──▶ [model_factory.py - Model Factory]
       │                                                         │
       │ ◀────────────────(Return Model Instance)────────────────┘
       ▼
(Invoke LCEL Chain)
       │
       ▼
[AI Provider API]
       │
       ▼ (JSON Output)
[ai_service.py - AI Logic]
       │
       ▼ (Pydantic Validation)
[app.py - Flask Route]
       │
       ▼ (JSON Response)
[Web Browser / UI]
```

## Key Features

- **Multi-Provider Support**: Seamlessly switch between different LLM providers using a centralized factory method without changing the core business logic.
- **Structured JSON Output**: Uses LangChain's `JsonOutputParser` and Pydantic (`AIResponse`) to force the AI to return strictly typed, predictable JSON responses.
- **Performance Optimization**: Employs Python's `@lru_cache` to cache model instances and LCEL chains, reducing overhead on subsequent API calls.
- **Robust Error Handling**: Graceful degradation and error handling for missing API keys, oversized payloads, and external provider outages.

## Usage

### 1. Configure Environment Variables
Ensure you have a `.env` file at the root of the workspace. Add the API keys for the providers you wish to use (you only need to configure the ones you want to test):
```env
GOOGLE_API_KEY="your_google_key"
GROQ_API_KEY="your_groq_key"
MISTRAL_API_KEY="your_mistral_key"
# ...
```

### 2. Run the Application
Start the Flask server from the project root:
```bash
uv run python 03-multi-provider-genai-flask/app.py
```
Open your browser and navigate to `http://127.0.0.1:5000` to interact with the application.

### 3. Verify Providers (Optional)
Run the built-in test script to check if your configured API keys are working correctly:
```bash
uv run python 03-multi-provider-genai-flask/provider_test.py
```

## Requirements

- Python ≥ 3.12
- Flask
- A `.env` file containing at least one valid API key.
