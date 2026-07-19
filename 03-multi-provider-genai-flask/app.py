"""Flask application with user-selectable AI providers."""

from __future__ import annotations

import os
import time
from typing import Any

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)

from ai_service import generate_ai_response
from model_factory import (
    ProviderConfigurationError,
    get_default_provider_id,
    get_provider_config,
    get_provider_configs,
)

app = Flask(__name__)

MAX_MESSAGE_LENGTH = int(
    os.getenv("MAX_MESSAGE_LENGTH", "5000")
)


def error_response(
    message: str,
    status_code: int,
):
    """Return a consistent JSON error payload."""
    return jsonify(
        {
            "error": message,
        }
    ), status_code


def parse_generate_request() -> tuple[str, str]:
    """Validate the request and return provider ID and message."""
    payload: Any = request.get_json(silent=True)

    if not isinstance(payload, dict):
        raise ValueError(
            "The request body must be a JSON object."
        )

    provider_id = str(
        payload.get("provider", "")
    ).strip().lower()

    message = str(
        payload.get("message", "")
    ).strip()

    if not provider_id:
        raise ValueError(
            "The provider field is required."
        )

    provider = get_provider_config(provider_id)

    if not provider.configured:
        raise ProviderConfigurationError(
            f"{provider.label} is not configured. "
            "Add its API key to the .env file."
        )

    if not message:
        raise ValueError(
            "The message field is required."
        )

    if len(message) > MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"The message cannot exceed "
            f"{MAX_MESSAGE_LENGTH} characters."
        )

    return provider.id, message


@app.get("/")
def index():
    """Render the main provider-selection interface."""
    providers = get_provider_configs()
    default_provider = get_default_provider_id()

    return render_template(
        "index.html",
        providers=providers,
        default_provider=default_provider,
        has_configured_providers=any(
            provider.configured
            for provider in providers
        ),
        max_message_length=MAX_MESSAGE_LENGTH,
    )


@app.get("/health")
def health():
    """Return application and provider configuration status."""
    providers = get_provider_configs()

    return jsonify(
        {
            "status": "ok",
            "configured_provider_count": sum(
                provider.configured
                for provider in providers
            ),
            "providers": [
                provider.public_dict()
                for provider in providers
            ],
        }
    )


@app.get("/api/providers")
def providers():
    """Return provider metadata for the frontend."""
    return jsonify(
        {
            "default_provider": (
                get_default_provider_id()
            ),
            "providers": [
                provider.public_dict()
                for provider in get_provider_configs()
            ],
        }
    )


@app.post("/generate")
def generate():
    """Analyze a customer message using the selected provider."""
    try:
        provider_id, message = (
            parse_generate_request()
        )

    except ValueError as error:
        return error_response(
            str(error),
            400,
        )

    except ProviderConfigurationError as error:
        return error_response(
            str(error),
            503,
        )

    started_at = time.perf_counter()

    try:
        result = generate_ai_response(
            provider_id=provider_id,
            user_message=message,
        )

    except Exception:
        app.logger.exception(
            "AI generation failed for provider '%s'.",
            provider_id,
        )

        return error_response(
            (
                "The selected AI provider could not process "
                "the request. Check the server log, API key, "
                "model ID, quota, and provider availability."
            ),
            502,
        )

    duration_ms = (
        time.perf_counter() - started_at
    ) * 1000

    result["duration_ms"] = round(
        duration_ms,
        2,
    )

    return jsonify(result)


if __name__ == "__main__":
    app.run(
        host=os.getenv(
            "FLASK_HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv(
                "FLASK_PORT",
                "5000",
            )
        ),
        debug=(
            os.getenv(
                "FLASK_DEBUG",
                "0",
            ) == "1"
        ),
    )
