"""Lesson 01 — Basic ChromaDB Grocery Collection & Similarity Search.

Demonstrates:
- Creating an in-memory ChromaDB collection with cosine distance.
- Configuring a Gemini-powered embedding function.
- Ingesting text items with unique IDs and metadata.
- Querying the vector database and inspecting similarity distances.
"""

from __future__ import annotations

import sys
from typing import Any, cast

# Safe UTF-8 reconfiguration for Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import CollectionMetadata
from shared_functions import ef

# Create a ChromaDB in-memory client
client: ClientAPI = chromadb.Client()

# Define the name for the collection
collection_name = "my_grocery_collection"


def main() -> None:
    """Create grocery collection, insert documents, and perform similarity search."""
    try:
        print("LESSON 01 — BASIC CHROMADB GROCERY COLLECTION")
        print("=" * 72)

        # Create a collection in the Chroma database with a specified name,
        # distance metric (cosine), and Gemini embedding function.
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        collection = client.create_collection(
            name=collection_name,
            metadata=cast(
                CollectionMetadata,
                {
                    "description": "A collection for storing grocery data",
                    "hnsw:space": "cosine",
                },
            ),
            embedding_function=ef,
        )

        print(f"Collection created: {collection.name}")

        # Array of grocery-related text items
        texts = [
            "fresh red apples",
            "organic bananas",
            "ripe mangoes",
            "whole wheat bread",
            "farm-fresh eggs",
            "natural yogurt",
            "frozen vegetables",
            "grass-fed beef",
            "free-range chicken",
            "fresh salmon fillet",
            "aromatic coffee beans",
            "pure honey",
            "golden apple",
            "red fruit",
        ]

        # Create a list of unique IDs for each text item
        ids = [f"food_{index + 1}" for index, _ in enumerate(texts)]

        # Add documents and their corresponding IDs & metadata to the collection
        # ChromaDB automatically generates embeddings using the configured embedding function
        collection.add(
            documents=texts,
            metadatas=cast(Any, [{"source": "grocery_store", "category": "food"} for _ in texts]),
            ids=ids,
        )

        # Retrieve all items stored in the collection
        all_items = collection.get()
        print(f"Collection contents: {len(all_items['documents'] or [])} documents stored.")

        # Function to perform a similarity search in the collection
        def perform_similarity_search(col: chromadb.Collection, query_term: str) -> None:
            try:
                print(f"\n[Query] Searching collection for: '{query_term}'")

                # Perform a query to search for the most similar documents
                results = col.query(
                    query_texts=[query_term],
                    n_results=3,  # Retrieve top 3 results
                )

                ids_list = results.get("ids")
                if not ids_list or len(ids_list) == 0 or len(ids_list[0]) == 0:
                    print(f'No documents found similar to "{query_term}"')
                    return

                ids_res = ids_list[0]
                dist_list = results.get("distances")
                distances = dist_list[0] if dist_list is not None else [0.0] * len(ids_res)
                docs_list = results.get("documents")
                documents = docs_list[0] if docs_list is not None else [""] * len(ids_res)

                print(f'Top 3 similar documents to "{query_term}":')
                for i in range(min(3, len(ids_res))):
                    doc_id = ids_res[i]
                    distance = distances[i]
                    text = documents[i]
                    # For cosine space: distance = 1 - cosine_similarity (range 0 to 2)
                    similarity_score = max(0.0, 1.0 - distance)

                    if not text:
                        print(f' - ID: {doc_id}, Text: "Text not available", Distance: {distance:.4f}, Score: {similarity_score:.4f}')
                    else:
                        print(f' - ID: {doc_id}, Text: "{text}", Distance: {distance:.4f}, Score: {similarity_score:.4f}')
            except Exception as error:
                print(f"Error in similarity search: {error}")

        # Test similarity search with query 'apple'
        perform_similarity_search(collection, "apple")

        # Test similarity search with query 'dairy product'
        perform_similarity_search(collection, "dairy product")

    except Exception as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
