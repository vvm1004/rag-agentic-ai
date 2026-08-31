"""Lesson 03 — Enhanced RAG-Powered Food Recommendation Chatbot with Google Gemini.

Demonstrates:
- Loading a structured JSON food dataset (nutritional info, ingredients, taste, cuisine).
- Creating and indexing embeddings in ChromaDB with Google Gemini.
- Context injection: formatting top semantic search matches into a grounded LLM prompt.
- Natural language food recommendations and conversational comparison mode.
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared_functions import (
    GeminiModelInference,
    create_similarity_search_collection,
    load_food_data,
    perform_similarity_search,
    populate_similarity_collection,
    settings,
)

# Global variables
food_items: List[Dict[str, Any]] = []

# Initialize the Gemini LLM model
model = GeminiModelInference(
    model_id=settings.gemini_model,
    params={"max_output_tokens": 500, "temperature": 0.4},
)


def prepare_context_for_llm(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Prepare structured context from search results for LLM prompt."""
    if not search_results:
        return "No relevant food items found in the database."

    context_parts: List[str] = [
        "Based on your query, here are the most relevant food options from our database:",
        "",
    ]

    for i, result in enumerate(search_results[:3], 1):
        food_context: List[str] = [
            f"Option {i}: {result.get('food_name', 'Unknown Dish')}",
            f"  - Description: {result.get('food_description', '')}",
            f"  - Cuisine: {result.get('cuisine_type', '')}",
            f"  - Calories: {result.get('food_calories_per_serving', 0)} per serving",
        ]

        if result.get("food_ingredients"):
            ingredients = result["food_ingredients"]
            if isinstance(ingredients, list):
                food_context.append(f"  - Key ingredients: {', '.join(ingredients[:5])}")
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")

        if result.get("food_health_benefits"):
            food_context.append(f"  - Health benefits: {result['food_health_benefits']}")

        if result.get("cooking_method"):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")

        if result.get("taste_profile"):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")

        sim_score = result.get("similarity_score", 0.0)
        food_context.append(f"  - Similarity score: {sim_score * 100:.1f}%")
        food_context.append("")

        context_parts.extend(food_context)

    return "\n".join(context_parts)


def generate_fallback_response(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate fallback response when LLM is unavailable or response is truncated."""
    if not search_results:
        return (
            "I couldn't find any food items matching your request. "
            "Try describing what you're in the mood for with different words!"
        )

    top_result = search_results[0]
    response_parts = [
        f"Based on your request for '{query}', I recommend {top_result.get('food_name', 'our top match')}.",
        f"It is a {top_result.get('cuisine_type', '')} dish with approximately "
        f"{top_result.get('food_calories_per_serving', 0)} calories per serving.",
    ]

    if len(search_results) > 1:
        second_choice = search_results[1]
        response_parts.append(
            f"Another great option would be {second_choice.get('food_name', '')}."
        )

    return " ".join(response_parts)


def generate_llm_rag_response(query: str, search_results: List[Dict[str, Any]]) -> str:
    """Generate intelligent recommendation response using Google Gemini with retrieved context."""
    try:
        context = prepare_context_for_llm(query, search_results)

        prompt = f"""You are a helpful and knowledgeable culinary recommendation assistant.
A user is asking for food recommendations, and relevant options have been retrieved from our vector database.

User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, engaging response that:
1. Warmly acknowledges the user's request.
2. Recommends 1 to 3 specific food items from the retrieved options above.
3. Explains clearly why these recommendations match the user's taste, dietary preference, or health goals.
4. Highlights key details such as cuisine type, calories, health benefits, or key ingredients.
5. Uses a friendly, conversational tone while keeping the answer concise and well-formatted with bullet points.

Response:"""

        generated_response = model.generate(prompt=prompt, params=None)

        if generated_response and "results" in generated_response:
            response_text = generated_response["results"][0]["generated_text"].strip()
            if len(response_text) < 30:
                return generate_fallback_response(query, search_results)
            return response_text

        return generate_fallback_response(query, search_results)

    except Exception as e:
        print(f"⚠️ LLM Generation Note: {e}")
        return generate_fallback_response(query, search_results)


def generate_simple_comparison(
    query1: str,
    query2: str,
    results1: List[Dict[str, Any]],
    results2: List[Dict[str, Any]],
) -> str:
    """Simple comparison fallback when LLM is unavailable."""
    if not results1 and not results2:
        return "No results found for either query."
    if not results1:
        return f"Found results for '{query2}' but none for '{query1}'."
    if not results2:
        return f"Found results for '{query1}' but none for '{query2}'."

    return (
        f"For '{query1}', I recommend {results1[0].get('food_name')}. "
        f"For '{query2}', {results2[0].get('food_name')} would be perfect."
    )


def generate_llm_comparison(
    query1: str,
    query2: str,
    results1: List[Dict[str, Any]],
    results2: List[Dict[str, Any]],
) -> str:
    """Generate AI-powered side-by-side comparison between two food queries."""
    try:
        context1 = prepare_context_for_llm(query1, results1[:3])
        context2 = prepare_context_for_llm(query2, results2[:3])

        comparison_prompt = f"""You are analyzing and comparing two different food preference queries. Please provide a thoughtful culinary comparison.

Query 1: "{query1}"
Top Results for Query 1:
{context1}

Query 2: "{query2}"
Top Results for Query 2:
{context2}

Please provide a concise comparison that:
1. Highlights the key differences in flavor profile, nutrition, and cuisine between these two preferences.
2. Notes any similarities or shared characteristics.
3. Suggests which option is better suited for specific occasions or dietary goals.
4. Concludes with the single best recommendation for each query.

Comparison:"""

        generated_response = model.generate(prompt=comparison_prompt, params=None)

        if generated_response and "results" in generated_response:
            return generated_response["results"][0]["generated_text"].strip()

        return generate_simple_comparison(query1, query2, results1, results2)

    except Exception as e:
        print(f"⚠️ Comparison Generation Note: {e}")
        return generate_simple_comparison(query1, query2, results1, results2)


def show_enhanced_rag_help() -> None:
    """Display help menu for the RAG chatbot."""
    print("\n📖 ENHANCED RAG CHATBOT HELP")
    print("=" * 60)
    print("🧠 This chatbot combines ChromaDB vector retrieval with Google Gemini")
    print("   to understand your dietary cravings and suggest tailored meals.")
    print("\nHow to get great recommendations:")
    print("  • Be specific: 'healthy Italian dish under 400 calories'")
    print("  • Mention cravings: 'spicy comfort food for a rainy evening'")
    print("  • Specify diet: 'high protein meal for workout recovery'")
    print("  • Name ingredients: 'something with salmon and avocado'")
    print("\nCommands:")
    print("  • 'compare' - Compare recommendations for two queries side-by-side")
    print("  • 'help'    - Display this menu")
    print("  • 'quit'    - Exit the chatbot")
    print("=" * 60)


def handle_enhanced_rag_query(
    collection: Any,
    query: str,
    conversation_history: List[str],
) -> None:
    """Handle a user natural language query using RAG with Gemini."""
    print(f"\n🔍 Searching vector database for: '{query}'...")

    search_results = perform_similarity_search(collection, query, n_results=3)

    if not search_results:
        print("🤖 Bot: I couldn't find any food items matching your request.")
        print("      Try describing what you're in the mood for with different words!")
        return

    print(f"✅ Found {len(search_results)} relevant matches in ChromaDB")
    print(f"🧠 Generating Gemini AI response ({settings.gemini_model})...")

    ai_response = generate_llm_rag_response(query, search_results)

    print(f"\n🤖 Bot:\n{ai_response}")

    # Show detailed search matches
    print("\n📊 Search Results Details:")
    print("-" * 55)
    for i, result in enumerate(search_results[:3], 1):
        name = result.get("food_name", "Dish")
        cuisine = result.get("cuisine_type", "General")
        cal = result.get("food_calories_per_serving", 0)
        score = result.get("similarity_score", 0.0) * 100
        print(f"{i}. 🍽️  {name}")
        print(f"   📍 Cuisine: {cuisine} | 🔥 {cal} cal | 📈 Match: {score:.1f}%")


def handle_enhanced_comparison_mode(collection: Any) -> None:
    """Handle comparative analysis mode for two food preferences."""
    print("\n🔄 ENHANCED COMPARISON MODE (Powered by Gemini AI)")
    print("-" * 55)

    query1 = input("Enter first food query: ").strip()
    query2 = input("Enter second food query: ").strip()

    if not query1 or not query2:
        print("❌ Please enter both queries for comparison.")
        return

    print(f"\n🔍 Analyzing '{query1}' vs '{query2}' with ChromaDB & Gemini...")

    results1 = perform_similarity_search(collection, query1, 3)
    results2 = perform_similarity_search(collection, query2, 3)

    comparison_response = generate_llm_comparison(query1, query2, results1, results2)

    print(f"\n🤖 AI Analysis:\n{comparison_response}")

    print("\n📊 DETAILED COMPARISON TABLE")
    print("=" * 65)
    q1_label = f"Query 1: {query1[:20]}..." if len(query1) > 20 else f"Query 1: {query1}"
    q2_label = f"Query 2: {query2[:20]}..." if len(query2) > 20 else f"Query 2: {query2}"
    print(f"{q1_label:<32} | {q2_label}")
    print("-" * 65)

    max_results = max(len(results1), len(results2))
    for i in range(min(max_results, 3)):
        left = (
            f"{results1[i]['food_name']} ({results1[i]['similarity_score']*100:.0f}%)"
            if i < len(results1)
            else "---"
        )
        right = (
            f"{results2[i]['food_name']} ({results2[i]['similarity_score']*100:.0f}%)"
            if i < len(results2)
            else "---"
        )
        print(f"{left[:32]:<32} | {right[:30]}")


def enhanced_rag_food_chatbot(collection: Any) -> None:
    """Interactive conversational food recommendation assistant."""
    print("\n" + "=" * 70)
    print("🤖 ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print(f"   Powered by Google Gemini ({settings.gemini_model}) & ChromaDB")
    print("=" * 70)
    print("💬 Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help'    - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit'    - Exit the chatbot")
    print("-" * 70)

    conversation_history: List[str] = []

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                print("🤖 Bot: Please tell me what kind of food you're looking for!")
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Bon appétit and see you next time! 👋")
                break

            if user_input.lower() in ["help", "h"]:
                show_enhanced_rag_help()

            elif user_input.lower() in ["compare"]:
                handle_enhanced_comparison_mode(collection)

            else:
                handle_enhanced_rag_query(collection, user_input, conversation_history)
                conversation_history.append(user_input)

                if len(conversation_history) > 5:
                    conversation_history = conversation_history[-3:]

        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find something delicious! 👋")
            break
        except Exception as e:
            print(f"❌ Bot encountered an error: {e}")


def main() -> None:
    """Main execution flow for RAG food chatbot."""
    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
        print(f"   Powered by Google Gemini & ChromaDB ({settings.gemini_model})")
        print("=" * 65)

        global food_items
        food_items = load_food_data("./FoodDataSet.json")
        print(f"✅ Loaded {len(food_items)} food items from FoodDataSet.json")

        collection = create_similarity_search_collection(
            "enhanced_rag_food_chatbot",
            {"description": "Enhanced RAG food recommendation chatbot with Gemini"},
        )
        populate_similarity_collection(collection, food_items)
        print("✅ Vector database initialized and indexed with Gemini embeddings")

        print("🔗 Testing Gemini LLM connection...")
        test_response = model.generate(prompt="Hello, return one word 'Connected'.", params=None)
        if test_response and "results" in test_response:
            print("✅ Google Gemini LLM connection established successfully!\n")
        else:
            print("❌ Google Gemini LLM connection failed.")
            return

        enhanced_rag_food_chatbot(collection)

    except Exception as error:
        print(f"❌ Error: {error}")


if __name__ == "__main__":
    main()
