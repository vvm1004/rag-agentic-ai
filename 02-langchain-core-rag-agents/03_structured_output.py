from typing import Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from common import get_chat_model


class MovieInfo(BaseModel):
    """Structured movie information."""

    title: str = Field(
        description="The official movie title."
    )

    director: str = Field(
        description="The name of the movie director."
    )

    year: int = Field(
        description="The movie release year."
    )

    genre: str = Field(
        description="The primary movie genre."
    )


class ReviewAnalysis(BaseModel):
    """Structured product review analysis."""

    sentiment: Literal[
        "positive",
        "neutral",
        "negative",
    ]

    features: list[str]

    summary: str


def demonstrate_json_parser() -> None:
    """Parse model-generated JSON with JsonOutputParser."""
    parser = JsonOutputParser(
        pydantic_object=MovieInfo
    )

    prompt = PromptTemplate(
        template=(
            "Provide accurate information about the movie "
            "{movie_name}.\n\n"
            "{format_instructions}\n\n"
            "Return only the requested JSON object."
        ),
        input_variables=[
            "movie_name",
        ],
        partial_variables={
            "format_instructions": (
                parser.get_format_instructions()
            ),
        },
    )

    chain = (
        prompt
        | get_chat_model(temperature=0.1)
        | parser
    )

    result = chain.invoke(
        {
            "movie_name": "The Matrix",
        }
    )

    print("=== JSON OUTPUT PARSER ===")
    print(result)

    print("\nSelected fields:")
    print("Title:", result["title"])
    print("Director:", result["director"])
    print("Year:", result["year"])
    print("Genre:", result["genre"])


def demonstrate_native_structured_output() -> None:
    """Use Gemini native JSON Schema structured output."""
    model = get_chat_model(temperature=0.1)

    structured_model = model.with_structured_output(
        schema=ReviewAnalysis,
    )

    review = (
        "The camera quality is excellent and the battery lasts "
        "all day, but the phone becomes very hot while gaming."
    )

    result = structured_model.invoke(
        (
            "Analyze the following product review. "
            "Extract its sentiment, mentioned product features, "
            "and a one-sentence summary.\n\n"
            f"Review: {review}"
        )
    )

    assert isinstance(result, ReviewAnalysis)

    print("\n=== NATIVE STRUCTURED OUTPUT ===")
    print(result)

    print("\nSelected fields:")
    print("Sentiment:", result.sentiment)
    print("Features:", result.features)
    print("Summary:", result.summary)


def main() -> None:
    demonstrate_json_parser()
    demonstrate_native_structured_output()


if __name__ == "__main__":
    main()