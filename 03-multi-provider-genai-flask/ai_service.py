"""Generate validated customer-support analysis with a selected provider."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from model_factory import (
    get_chat_model,
    get_provider_config,
)


class AIResponse(BaseModel):
    """Structured customer-support response."""

    summary: str = Field(
        min_length=1,
        description=(
            "A concise summary of the customer's message."
        ),
    )

    sentiment: int = Field(
        ge=0,
        le=100,
        description=(
            "A sentiment score from 0 to 100. "
            "0 is extremely negative, 50 is neutral, "
            "and 100 is extremely positive."
        ),
    )

    category: Literal[
        "account",
        "billing",
        "general",
        "product",
        "shipping",
        "technical",
        "other",
    ] = Field(
        description=(
            "The single category that best matches "
            "the customer inquiry."
        ),
    )

    action: str = Field(
        min_length=1,
        description=(
            "The recommended next action for a support agent."
        ),
    )

    response: str = Field(
        min_length=1,
        description=(
            "A concise, empathetic, and professional response "
            "that can be sent directly to the customer."
        ),
    )


SYSTEM_PROMPT = """
You are an AI assistant for a customer support team.

Analyze the customer's message and return only the JSON object requested by
the format instructions.

Requirements:
- Summarize the issue accurately.
- Score sentiment from 0 to 100.
- Select exactly one allowed category.
- Recommend a practical next action for a support agent.
- Write a concise customer-facing response.
- Do not invent account details, order details, policies, refund status,
  shipment status, or technical facts.
- Do not wrap the JSON in commentary outside the requested format.
""".strip()


output_parser = JsonOutputParser(
    pydantic_object=AIResponse
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "{system_prompt}\n\n{format_instructions}",
        ),
        (
            "human",
            "Customer message:\n{user_message}",
        ),
    ]
)


@lru_cache(maxsize=32)
def build_chain(provider_id: str):
    """Create and cache a provider-specific LCEL chain."""
    model = get_chat_model(provider_id)

    return (
        prompt_template
        | model
        | output_parser
    )


def generate_ai_response(
    provider_id: str,
    user_message: str,
) -> dict[str, Any]:
    """Generate and validate a structured response."""
    cleaned_message = user_message.strip()

    if not cleaned_message:
        raise ValueError(
            "The customer message cannot be empty."
        )

    provider = get_provider_config(provider_id)
    chain = build_chain(provider.id)

    parsed_result = chain.invoke(
        {
            "system_prompt": SYSTEM_PROMPT,
            "format_instructions": (
                output_parser.get_format_instructions()
            ),
            "user_message": cleaned_message,
        }
    )

    validated_result = AIResponse.model_validate(
        parsed_result
    )

    response = validated_result.model_dump()
    response["provider"] = provider.id
    response["provider_label"] = provider.label
    response["model"] = provider.model

    return response
