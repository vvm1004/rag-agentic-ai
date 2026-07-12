import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


def get_gemini_llm(temperature: float = 0.2):
    """
    Create a Gemini chat model for LangChain.

    Environment variables:
    - GOOGLE_API_KEY: your Gemini API key
    - GEMINI_MODEL: model name, default is gemini-2.5-flash
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        raise ValueError(
            "Missing GOOGLE_API_KEY. Please add it to your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
    )