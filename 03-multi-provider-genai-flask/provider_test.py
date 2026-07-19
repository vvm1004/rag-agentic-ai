"""Test every configured provider from the terminal."""

from __future__ import annotations

import json

from ai_service import generate_ai_response
from model_factory import get_provider_configs

TEST_MESSAGE = """
I was charged twice for the same subscription.
I contacted support yesterday but have not received a response.
Please help me resolve the duplicate charge.
""".strip()


def main() -> None:
    """Test configured providers and skip missing API keys."""
    configured_count = 0

    for provider in get_provider_configs():
        print("\n" + "=" * 72)
        print(
            f"{provider.label} | {provider.model}"
        )
        print("=" * 72)

        if not provider.configured:
            print(
                "Skipped: API key is not configured."
            )
            continue

        configured_count += 1

        try:
            result = generate_ai_response(
                provider_id=provider.id,
                user_message=TEST_MESSAGE,
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        except Exception as error:
            print(
                "Failed: "
                f"{type(error).__name__}: {error}"
            )

    if configured_count == 0:
        print(
            "\nNo providers are configured. "
            "Copy .env.example to .env and add at least one API key."
        )


if __name__ == "__main__":
    main()
