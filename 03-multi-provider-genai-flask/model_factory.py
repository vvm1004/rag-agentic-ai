"""Create LangChain chat models for multiple API providers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()


class ProviderConfigurationError(RuntimeError):
    """Raised when a selected provider is not configured correctly."""


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration and display metadata for one provider."""

    id: str
    label: str
    description: str
    api_key_environment_variable: str
    model_environment_variable: str
    default_model: str
    model: str
    configured: bool

    def public_dict(self) -> dict[str, str | bool]:
        """Return fields that are safe to expose through the API."""
        data = asdict(self)
        data.pop("api_key_environment_variable")
        data.pop("model_environment_variable")
        data.pop("default_model")
        return data


PROVIDER_DEFINITIONS: tuple[
    tuple[str, str, str, str, str, str],
    ...,
] = (
    (
        "gemini",
        "Google Gemini",
        "Strong default for structured analysis and general application work.",
        "GOOGLE_API_KEY",
        "GEMINI_MODEL",
        "gemini-3.5-flash",
    ),
    (
        "groq",
        "Groq",
        "Useful for comparing very fast inference with an open model.",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "qwen/qwen3-32b",
    ),
    (
        "mistral",
        "Mistral AI",
        "Direct access to Mistral chat models and structured generation.",
        "MISTRAL_API_KEY",
        "MISTRAL_MODEL",
        "mistral-small-latest",
    ),
    (
        "nvidia",
        "NVIDIA",
        "Access NVIDIA-hosted models and Nemotron through the API Catalog.",
        "NVIDIA_API_KEY",
        "NVIDIA_MODEL",
        "nvidia/nemotron-3-super-120b-a12b",
    ),
    (
        "huggingface",
        "Hugging Face",
        "Experiment with open models through Hugging Face Inference Providers.",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_MODEL",
        "openai/gpt-oss-120b",
    ),
    (
        "openrouter",
        "OpenRouter",
        "Use one OpenAI-compatible endpoint to access many model providers.",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "google/gemini-3.5-flash",
    ),
)


def _is_placeholder(value: str) -> bool:
    """Return True when a value is blank or still contains a sample secret."""
    normalized = value.strip().lower()

    return (
        not normalized
        or normalized.startswith("your_")
        or normalized.startswith("<your")
        or "replace_me" in normalized
    )


def get_provider_configs() -> list[ProviderConfig]:
    """Build provider configurations from environment variables."""
    providers: list[ProviderConfig] = []

    for (
        provider_id,
        label,
        description,
        api_key_variable,
        model_variable,
        default_model,
    ) in PROVIDER_DEFINITIONS:
        api_key = os.getenv(api_key_variable, "")
        model = os.getenv(model_variable, default_model).strip()

        providers.append(
            ProviderConfig(
                id=provider_id,
                label=label,
                description=description,
                api_key_environment_variable=api_key_variable,
                model_environment_variable=model_variable,
                default_model=default_model,
                model=model or default_model,
                configured=(
                    not _is_placeholder(api_key)
                    and bool(model or default_model)
                ),
            )
        )

    return providers


def get_provider_config(provider_id: str) -> ProviderConfig:
    """Return one provider configuration by ID."""
    normalized_id = provider_id.strip().lower()

    for provider in get_provider_configs():
        if provider.id == normalized_id:
            return provider

    supported = ", ".join(
        provider.id for provider in get_provider_configs()
    )

    raise ValueError(
        f"Unsupported provider: {provider_id}. "
        f"Supported providers: {supported}."
    )


def get_configured_providers() -> list[ProviderConfig]:
    """Return providers whose API keys are configured."""
    return [
        provider
        for provider in get_provider_configs()
        if provider.configured
    ]


def get_default_provider_id() -> str | None:
    """Return the configured default provider or the first available one."""
    configured = get_configured_providers()

    if not configured:
        return None

    requested_default = os.getenv(
        "DEFAULT_PROVIDER",
        "gemini",
    ).strip().lower()

    if any(
        provider.id == requested_default
        for provider in configured
    ):
        return requested_default

    return configured[0].id


def _require_provider(provider_id: str) -> ProviderConfig:
    """Validate that a provider has a usable API key."""
    provider = get_provider_config(provider_id)

    if not provider.configured:
        raise ProviderConfigurationError(
            f"{provider.label} is not configured. "
            f"Set {provider.api_key_environment_variable} "
            f"in the .env file."
        )

    return provider


def _get_generation_settings() -> tuple[float, int]:
    """Load common model generation settings."""
    temperature = float(
        os.getenv("LLM_TEMPERATURE", "0.1")
    )
    max_tokens = int(
        os.getenv("LLM_MAX_TOKENS", "512")
    )

    return temperature, max_tokens


@lru_cache(maxsize=32)
def get_chat_model(provider_id: str) -> BaseChatModel:
    """Create and cache the selected provider's LangChain chat model."""
    provider = _require_provider(provider_id)
    temperature, max_tokens = _get_generation_settings()

    factories: dict[
        str,
        Callable[
            [ProviderConfig, float, int],
            BaseChatModel,
        ],
    ] = {
        "gemini": _create_gemini_model,
        "groq": _create_groq_model,
        "mistral": _create_mistral_model,
        "nvidia": _create_nvidia_model,
        "huggingface": _create_huggingface_model,
        "openrouter": _create_openrouter_model,
    }

    return factories[provider.id](
        provider,
        temperature,
        max_tokens,
    )


def _create_gemini_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create a Google Gemini chat model."""
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
    )

    return ChatGoogleGenerativeAI(
        model=provider.model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=2,
    )


def _create_groq_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create a Groq chat model."""
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=provider.model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def _create_mistral_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create a Mistral API chat model."""
    from langchain_mistralai import ChatMistralAI

    return ChatMistralAI(
        model=provider.model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
    )


def _create_nvidia_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create an NVIDIA API Catalog chat model."""
    from langchain_nvidia_ai_endpoints import (
        ChatNVIDIA,
    )

    return ChatNVIDIA(
        model=provider.model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def _create_huggingface_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create a Hugging Face Inference Providers chat model."""
    from langchain_huggingface import (
        ChatHuggingFace,
        HuggingFaceEndpoint,
    )

    inference_provider = os.getenv(
        "HUGGINGFACE_PROVIDER",
        "auto",
    ).strip()

    use_sampling = temperature > 0

    endpoint_arguments: dict[str, Any] = {
        "repo_id": provider.model,
        "task": "text-generation",
        "provider": inference_provider or "auto",
        "max_new_tokens": max_tokens,
        "do_sample": use_sampling,
    }

    if use_sampling:
        endpoint_arguments["temperature"] = temperature

    endpoint = HuggingFaceEndpoint(
        **endpoint_arguments
    )

    return ChatHuggingFace(
        llm=endpoint
    )


def _create_openrouter_model(
    provider: ProviderConfig,
    temperature: float,
    max_tokens: int,
) -> BaseChatModel:
    """Create an OpenRouter model through its OpenAI-compatible API."""
    from langchain_openai import ChatOpenAI

    api_key = os.environ[
        provider.api_key_environment_variable
    ]

    default_headers: dict[str, str] = {}

    site_url = os.getenv(
        "OPENROUTER_SITE_URL",
        "",
    ).strip()

    app_name = os.getenv(
        "OPENROUTER_APP_NAME",
        "",
    ).strip()

    if site_url:
        default_headers["HTTP-Referer"] = site_url

    if app_name:
        default_headers["X-Title"] = app_name

    return ChatOpenAI(
        model=provider.model,
        api_key=api_key,
        base_url=os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=2,
        default_headers=default_headers or None,
    )
