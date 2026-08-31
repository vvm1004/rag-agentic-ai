"""Shared ChromaDB and Gemini utilities for Folder 09."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast


# Ensure UTF-8 output encoding on Windows consoles safely
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import (
    CollectionMetadata,
    Embeddable,
    EmbeddingFunction,
    Embeddings,
    Metadata,
    Where,
)
from dotenv import load_dotenv
from google import genai
from google.genai import types

BASE_DIR = Path(__file__).resolve().parent

# Prefer the shared .env file in the root workspace
load_dotenv(BASE_DIR.parent / ".env")
load_dotenv(BASE_DIR / ".env", override=False)
load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Environment configuration used across folder 09."""

    google_api_key: str
    gemini_model: str
    embedding_model: str
    embedding_dimensions: int

    @property
    def api_key_configured(self) -> bool:
        normalized = self.google_api_key.strip().lower()
        return bool(
            normalized
            and not normalized.startswith("your_")
            and "replace_me" not in normalized
        )


def _read_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, received {raw!r}.") from error
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero.")
    return value


def load_settings() -> Settings:
    """Load settings from .env configuration."""
    return Settings(
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite").strip(),
        embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2-preview").strip(),
        embedding_dimensions=_read_int("GEMINI_EMBEDDING_DIMENSIONS", 768),
    )


settings = load_settings()


def require_api_key() -> None:
    """Validate that the Google API key is configured."""
    if not settings.api_key_configured:
        raise RuntimeError(
            "GOOGLE_API_KEY is missing or still contains a placeholder "
            "value in the project's .env file."
        )


def get_genai_client() -> genai.Client:
    """Create a Google GenAI client instance."""
    require_api_key()
    return genai.Client(api_key=settings.google_api_key)


class GeminiEmbeddingFunction(EmbeddingFunction[Embeddable]):
    """ChromaDB-compatible embedding function powered by Google Gemini API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        output_dimensionality: Optional[int] = None,
    ) -> None:
        self.api_key = api_key or settings.google_api_key
        self.model_name = model_name or settings.embedding_model
        self.output_dimensionality = output_dimensionality or settings.embedding_dimensions
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                require_api_key()
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def __call__(self, input: Embeddable) -> Embeddings:
        """Generate vector embeddings for input documents."""
        if not input:
            return cast(Embeddings, [])

        docs: Sequence[str] = [str(item) for item in input]

        # Format input documents for semantic search / retrieval
        contents: list[Any] = [
            [f"task: sentence similarity | query: {doc}"] for doc in docs
        ]

        config = types.EmbedContentConfig()
        if self.output_dimensionality:
            config.output_dimensionality = self.output_dimensionality

        response = self.client.models.embed_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        embeddings_list = getattr(response, "embeddings", None)
        if not embeddings_list:
            raise RuntimeError("Gemini returned no embeddings.")

        result: list[list[float]] = []
        for item in embeddings_list:
            values = getattr(item, "values", None)
            if values is None:
                raise RuntimeError("Embedding item has no vector values.")
            result.append([float(val) for val in values])

        return cast(Embeddings, result)


# Singleton embedding function instance
ef = GeminiEmbeddingFunction()

# Default in-memory ChromaDB client
client: ClientAPI = chromadb.Client()


def load_food_data(file_path: str = "./FoodDataSet.json") -> List[Dict[str, Any]]:
    """Load food data from JSON file with multi-path resolution fallback."""
    candidate_paths = [
        Path(file_path),
        BASE_DIR / file_path,
        BASE_DIR / Path(file_path).name,
        BASE_DIR / "data" / Path(file_path).name,
    ]

    target_file: Optional[Path] = None
    for p in candidate_paths:
        if p.exists() and p.is_file():
            target_file = p
            break

    if not target_file:
        raise FileNotFoundError(
            f"Could not find FoodDataSet.json in any expected locations: "
            f"{[str(p) for p in candidate_paths]}"
        )

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of food items in {target_file}, got {type(data)}.")

    return data


def create_similarity_search_collection(
    collection_name: str,
    metadata: Optional[Dict[str, Any]] = None,
    chroma_client: Optional[ClientAPI] = None,
    embedding_func: Optional[EmbeddingFunction[Embeddable]] = None,
) -> chromadb.Collection:
    """Create or reset a ChromaDB collection configured with cosine distance and Gemini embeddings."""
    db_client = chroma_client or client
    meta = metadata or {"description": "A collection for semantic similarity search"}
    # Merge distance space into metadata
    meta = {**meta, "hnsw:space": "cosine"}
    active_ef = embedding_func or ef

    try:
        # Delete existing collection if it exists to ensure fresh run
        db_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = db_client.create_collection(
        name=collection_name,
        metadata=cast(CollectionMetadata, meta),
        embedding_function=active_ef,
    )
    return collection


def populate_similarity_collection(
    collection: chromadb.Collection,
    food_items: List[Dict[str, Any]],
) -> None:
    """Populate ChromaDB collection with formatted food documents and metadata."""
    documents: List[str] = []
    ids: List[str] = []
    metadatas: List[Metadata] = []

    for index, item in enumerate(food_items):
        doc_id = str(item.get("id", f"food_{index + 1}"))
        name = str(item.get("food_name", "Unknown Food"))
        desc = str(item.get("food_description", ""))
        cuisine = str(item.get("cuisine_type", "General"))
        calories = int(item.get("food_calories_per_serving", 0))
        ingredients = item.get("food_ingredients", [])
        ingredients_str = (
            ", ".join(ingredients) if isinstance(ingredients, list) else str(ingredients)
        )
        health = str(item.get("food_health_benefits", ""))
        method = str(item.get("cooking_method", ""))
        taste = str(item.get("taste_profile", ""))

        # Construct comprehensive document text for semantic search
        doc_text = (
            f"{name} ({cuisine} cuisine). {desc} "
            f"Calories: {calories} cal. "
            f"Ingredients: {ingredients_str}. "
            f"Health benefits: {health}. "
            f"Cooking method: {method}. "
            f"Taste: {taste}."
        )

        documents.append(doc_text)
        ids.append(doc_id)
        metadatas.append({
            "food_name": name,
            "food_description": desc,
            "cuisine_type": cuisine,
            "food_calories_per_serving": calories,
            "food_ingredients": ingredients_str,
            "food_health_benefits": health,
            "cooking_method": method,
            "taste_profile": taste,
        })

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=cast(Any, metadatas),
    )


def perform_similarity_search(
    collection: chromadb.Collection,
    query_term: str,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """Perform semantic similarity search on a ChromaDB collection."""
    results = collection.query(
        query_texts=[query_term],
        n_results=n_results,
    )

    ids_list = results.get("ids")
    if not ids_list or len(ids_list) == 0 or len(ids_list[0]) == 0:
        return []

    matched_items: List[Dict[str, Any]] = []
    ids = ids_list[0]
    distances_list = results.get("distances")
    distances = distances_list[0] if distances_list is not None else [0.0] * len(ids)
    metadatas_list = results.get("metadatas")
    metadatas = metadatas_list[0] if metadatas_list is not None else [{}] * len(ids)
    documents_list = results.get("documents")
    docs = documents_list[0] if documents_list is not None else [""] * len(ids)

    for doc_id, dist, meta, doc_text in zip(ids, distances, metadatas, docs):
        # Convert cosine distance (0: identical, 2: opposite) to similarity score (0.0 to 1.0)
        similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))

        item_dict: Dict[str, Any] = dict(meta) if meta else {}
        item_dict["id"] = doc_id
        item_dict["document"] = doc_text
        item_dict["distance"] = float(dist)
        item_dict["similarity_score"] = similarity_score

        # Ensure standard keys exist
        if "food_name" not in item_dict:
            item_dict["food_name"] = doc_id
        if "food_description" not in item_dict:
            item_dict["food_description"] = doc_text
        if "cuisine_type" not in item_dict:
            item_dict["cuisine_type"] = "General"
        if "food_calories_per_serving" not in item_dict:
            item_dict["food_calories_per_serving"] = 0

        matched_items.append(item_dict)

    return matched_items


def perform_filtered_similarity_search(
    collection: chromadb.Collection,
    query_term: str,
    cuisine_filter: Optional[str] = None,
    n_results: int = 3,
) -> List[Dict[str, Any]]:
    """Perform similarity search with metadata filtering."""
    where_clause: Optional[Where] = cast(Optional[Where], {"cuisine_type": cuisine_filter} if cuisine_filter else None)

    results = collection.query(
        query_texts=[query_term],
        n_results=n_results,
        where=where_clause,
    )

    ids_list = results.get("ids")
    if not ids_list or len(ids_list) == 0 or len(ids_list[0]) == 0:
        return []

    matched_items: List[Dict[str, Any]] = []
    ids = ids_list[0]
    distances_list = results.get("distances")
    distances = distances_list[0] if distances_list is not None else [0.0] * len(ids)
    metadatas_list = results.get("metadatas")
    metadatas = metadatas_list[0] if metadatas_list is not None else [{}] * len(ids)
    documents_list = results.get("documents")
    docs = documents_list[0] if documents_list is not None else [""] * len(ids)

    for doc_id, dist, meta, doc_text in zip(ids, distances, metadatas, docs):
        similarity_score = max(0.0, min(1.0, 1.0 - float(dist)))
        item_dict: Dict[str, Any] = dict(meta) if meta else {}
        item_dict["id"] = doc_id
        item_dict["document"] = doc_text
        item_dict["distance"] = float(dist)
        item_dict["similarity_score"] = similarity_score
        matched_items.append(item_dict)

    return matched_items


class GeminiModelInference:
    """Gemini LLM interface matching ModelInference protocol."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        credentials: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None,
        space_id: Optional[str] = None,
        verify: bool = False,
    ) -> None:
        self.model_id = model_id or settings.gemini_model
        self.default_params = params or {"max_output_tokens": 500, "temperature": 0.3}
        self.client = get_genai_client()

    def generate(
        self,
        prompt: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate text using Google Gemini."""
        merged_params = dict(self.default_params)
        if params:
            merged_params.update(params)

        max_tokens = merged_params.get("max_new_tokens") or merged_params.get("max_output_tokens", 500)
        temp = merged_params.get("temperature", 0.3)

        config = types.GenerateContentConfig(
            max_output_tokens=int(max_tokens),
            temperature=float(temp),
        )

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config,
        )

        generated_text = response.text or ""
        return {
            "results": [
                {
                    "generated_text": generated_text,
                }
            ]
        }


# Default ModelInference alias for compatibility
ModelInference = GeminiModelInference
