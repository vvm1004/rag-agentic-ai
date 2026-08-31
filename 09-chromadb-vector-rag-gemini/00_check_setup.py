"""Validation script to check prerequisites and configuration for Folder 09."""

from __future__ import annotations

import importlib.util

from shared_functions import ef, get_genai_client, load_food_data, settings

REQUIRED_MODULES = {
    "chromadb": "chromadb",
    "dotenv": "python-dotenv",
    "google.genai": "google-genai",
}


def module_exists(module_name: str) -> bool:
    """Return whether a Python module can be imported."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False


def main() -> None:
    """Check configuration, dependencies, and connections."""
    print("FOLDER 09 — CHROMADB VECTOR SEARCH & RAG WITH GEMINI")
    print("=" * 72)

    print("Google API key configured:", settings.api_key_configured)
    print("Gemini LLM model:", settings.gemini_model)
    print("Gemini Embedding model:", settings.embedding_model)
    print("Configured dimensions:", settings.embedding_dimensions)

    print("\nRequired Python packages:")
    print("-" * 72)
    for mod, pkg in REQUIRED_MODULES.items():
        status = "OK" if module_exists(mod) else "MISSING"
        print(f"  • {pkg:<20}: {status}")

    # Check FoodDataSet.json
    print("\nDataset Verification:")
    print("-" * 72)
    try:
        food_items = load_food_data("./FoodDataSet.json")
        print(f"  • FoodDataSet.json loaded successfully: {len(food_items)} food records found.")
    except Exception as e:
        print(f"  • Failed to load FoodDataSet.json: {e}")

    # Test Gemini & ChromaDB integration
    print("\nAPI & Embedding Function Check:")
    print("-" * 72)
    if settings.api_key_configured:
        try:
            test_emb = ef(["Test food query"])
            print(f"  • Gemini Embedding test passed! Vector dimension: {len(test_emb[0])}")
        except Exception as err:
            print(f"  • Gemini Embedding test failed: {err}")

        try:
            client = get_genai_client()
            res = client.models.generate_content(
                model=settings.gemini_model,
                contents="Reply with 'Gemini OK'",
            )
            print(f"  • Gemini LLM test passed! Response: {res.text.strip() if res.text else 'No text'}")
        except Exception as err:
            print(f"  • Gemini LLM test failed: {err}")
    else:
        print("  • Skipped live API tests because GOOGLE_API_KEY is not configured.")

    print("=" * 72)


if __name__ == "__main__":
    main()
