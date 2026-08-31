"""Lesson 04 — Compare Food Search Systems: Interactive vs. Advanced vs. RAG Chatbot.

Demonstrates:
- Side-by-side comparison of three different retrieval paradigms:
  1. Interactive Similarity Search (pure cosine semantic matching)
  2. Advanced Filtered Search (semantic matching + exact metadata constraints)
  3. RAG Chatbot Approach (semantic retrieval + Google Gemini LLM generation)
- Timing & latency benchmarks.
- Comprehensive trade-off analysis.
"""

from __future__ import annotations

import time

from shared_functions import (
    GeminiModelInference,
    create_similarity_search_collection,
    load_food_data,
    perform_filtered_similarity_search,
    perform_similarity_search,
    populate_similarity_collection,
    settings,
)

model = GeminiModelInference(
    model_id=settings.gemini_model,
    params={"max_output_tokens": 400, "temperature": 0.4},
)


def main() -> None:
    """Compare all three search systems with the same query."""
    print("🔬 FOOD SEARCH SYSTEMS COMPARISON (Powered by Google Gemini & ChromaDB)")
    print("=" * 72)

    # Load data once for all systems
    food_items = load_food_data("./FoodDataSet.json")
    print(f"✅ Loaded {len(food_items)} food records from FoodDataSet.json\n")

    # Create distinct collections for each system
    interactive_collection = create_similarity_search_collection("comparison_interactive")
    advanced_collection = create_similarity_search_collection("comparison_advanced")
    rag_collection = create_similarity_search_collection("comparison_rag")

    # Populate all collections
    populate_similarity_collection(interactive_collection, food_items)
    populate_similarity_collection(advanced_collection, food_items)
    populate_similarity_collection(rag_collection, food_items)

    # Test query
    test_query = "chocolate dessert"

    print(f"🔍 Testing query across all 3 systems: '{test_query}'")
    print("=" * 72)

    # -------------------------------------------------------------
    # System 1: Interactive Search Style
    # -------------------------------------------------------------
    print("\n1️⃣ INTERACTIVE SEARCH APPROACH (Direct Similarity):")
    print("-" * 55)
    start_time = time.time()
    interactive_results = perform_similarity_search(interactive_collection, test_query, 3)
    interactive_time = time.time() - start_time

    for i, result in enumerate(interactive_results, 1):
        score_pct = result.get("similarity_score", 0.0) * 100
        print(f"{i}. {result.get('food_name')} ({score_pct:.1f}% match)")
        print(f"   {result.get('food_description')}")
    print(f"⏱️ Response time: {interactive_time:.3f} seconds")

    # -------------------------------------------------------------
    # System 2: Advanced Search Style
    # -------------------------------------------------------------
    print("\n2️⃣ ADVANCED SEARCH APPROACH (Metadata Filtering):")
    print("-" * 55)
    start_time = time.time()

    # Show basic search
    basic_results = perform_similarity_search(advanced_collection, test_query, 3)
    print("📋 Basic results:")
    for i, result in enumerate(basic_results, 1):
        print(
            f"   {i}. {result.get('food_name')} - {result.get('cuisine_type')} "
            f"({result.get('food_calories_per_serving')} cal)"
        )

    # Show filtered search
    filtered_results = perform_filtered_similarity_search(
        advanced_collection, test_query, cuisine_filter="American", n_results=2
    )
    print("🇺🇸 Filtered specifically for American cuisine:")
    for i, result in enumerate(filtered_results, 1):
        score_pct = result.get("similarity_score", 0.0) * 100
        print(f"   {i}. {result.get('food_name')} ({score_pct:.1f}% match)")

    advanced_time = time.time() - start_time
    print(f"⏱️ Response time: {advanced_time:.3f} seconds")

    # -------------------------------------------------------------
    # System 3: RAG Chatbot Style
    # -------------------------------------------------------------
    print("\n3️⃣ RAG CHATBOT APPROACH (ChromaDB Retrieval + Gemini LLM):")
    print("-" * 55)
    start_time = time.time()

    rag_results = perform_similarity_search(rag_collection, test_query, 3)

    if rag_results:
        top_name = rag_results[0].get("food_name", "Chocolate Dessert")
        top_score = rag_results[0].get("similarity_score", 0.0) * 100
        top_cuisine = rag_results[0].get("cuisine_type", "American")
        top_cal = rag_results[0].get("food_calories_per_serving", 400)
        second_name = (
            rag_results[1].get("food_name", "Sweet treat")
            if len(rag_results) > 1
            else "another alternative"
        )

        prompt = f"""You are a culinary expert assistant. A user searched for '{test_query}'.
Top database matches:
1. {top_name} ({top_cuisine} cuisine, {top_cal} calories, {top_score:.1f}% semantic similarity)
2. {second_name}

Provide a 2-sentence conversational, mouth-watering recommendation."""

        try:
            gen_res = model.generate(prompt=prompt)
            rag_response = gen_res["results"][0]["generated_text"].strip()
        except Exception:
            rag_response = (
                f"Perfect! I found some delicious options for you. I highly recommend {top_name} "
                f"({top_score:.0f}% match) with {top_cal} calories. You might also enjoy {second_name}!"
            )
    else:
        rag_response = "No matching dishes found in the database."

    rag_time = time.time() - start_time
    print(f"🤖 Gemini Bot: {rag_response}")
    print(f"⏱️ Response time: {rag_time:.3f} seconds")

    # -------------------------------------------------------------
    # Comparison Summary Table
    # -------------------------------------------------------------
    print("\n📊 SYSTEM COMPARISON SUMMARY:")
    print("=" * 72)
    print("Interactive Search:")
    print("  ✅ Ultra fast and simple")
    print("  ✅ Direct score and document retrieval")
    print("  ❌ No natural language explanations")

    print("\nAdvanced Search:")
    print("  ✅ Powerful categorical and numeric metadata filtering ($gte, $in)")
    print("  ✅ Combines vector similarity with hard business rules")
    print("  ❌ Requires callers to know metadata schema")

    print("\nRAG Chatbot (ChromaDB + Gemini):")
    print("  ✅ Natural language understanding & multi-turn conversational interaction")
    print("  ✅ Context-grounded, empathetic explanations")
    print("  ✅ Can reason across health benefits, taste profiles, and cooking methods")
    print("  ❌ Adds LLM generation latency (~1-2s)")

    print("\n⏱️ Performance & Latency Breakdown:")
    print("-" * 72)
    print(f"  • Interactive Search  : {interactive_time:.3f}s")
    print(f"  • Advanced Search     : {advanced_time:.3f}s")
    print(f"  • Gemini RAG Chatbot  : {rag_time:.3f}s")
    print("=" * 72)


if __name__ == "__main__":
    main()
