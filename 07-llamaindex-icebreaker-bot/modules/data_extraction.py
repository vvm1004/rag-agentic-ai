"""Load and clean professional profile data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import icebreaker_config as config


class ProfileDataError(RuntimeError):
    """Raised when profile data cannot be loaded."""


def _clean_value(value: Any) -> Any:
    """Recursively remove empty values from JSON-like data."""
    if isinstance(value, dict):
        cleaned = {
            key: _clean_value(item)
            for key, item in value.items()
            if item not in (
                None,
                "",
                [],
                {},
            )
        }

        return {
            key: item
            for key, item in cleaned.items()
            if item not in (
                None,
                "",
                [],
                {},
            )
        }

    if isinstance(value, list):
        cleaned_items = [
            _clean_value(item)
            for item in value
            if item not in (
                None,
                "",
                [],
                {},
            )
        ]

        return [
            item
            for item in cleaned_items
            if item not in (
                None,
                "",
                [],
                {},
            )
        ]

    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ProfileDataError(
            f"Profile JSON was not found: {path}"
        )

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise ProfileDataError(
            f"Invalid JSON in {path.name}: {error}"
        ) from error

    if not isinstance(data, dict):
        raise ProfileDataError(
            "The profile JSON root must be an object."
        )

    cleaned = _clean_value(data)

    if not isinstance(cleaned, dict) or not cleaned:
        raise ProfileDataError(
            "The profile JSON is empty after cleaning."
        )

    return cleaned


def extract_linkedin_profile(
    json_file: str | None = None,
    *,
    mock: bool = True,
) -> dict[str, Any]:
    """Load mock profile data or a user-provided local JSON file.

    The original lab showed Proxycurl API code, but that service has been
    discontinued. This local learning version intentionally does not scrape
    LinkedIn or call a replacement profile API.
    """
    if mock:
        return _load_json(
            config.MOCK_PROFILE_PATH
        )

    if not json_file:
        raise ProfileDataError(
            "Upload a JSON profile or enable mock data."
        )

    path = Path(json_file)

    if path.suffix.lower() != ".json":
        raise ProfileDataError(
            "Only JSON profile files are supported."
        )

    return _load_json(path)
