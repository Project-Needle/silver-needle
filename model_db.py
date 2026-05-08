"""
Shared model database utilities.

Provides normalize_model_name() and lookup() for any probe that needs to
match a user-supplied model string against a JSON database (model_specs.json,
poc_tokens.json, etc.).
"""

import json
import re
from typing import Optional


def normalize_model_name(name: str) -> str:
    """
    Normalize a model identifier for database lookup:
      - Drop provider prefix: take only the segment after the last '/'
      - Replace underscores with dashes
      - Strip trailing date stamps in either form:
          YYYY-MM-DD  e.g. -2024-04-09
          YYYYMMDD    e.g. -20250619
    """
    name = name.split("/")[-1]
    name = name.replace("_", "-")
    name = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)  # YYYY-MM-DD
    name = re.sub(r"-\d{8}$", "", name)               # YYYYMMDD
    return name.lower()


def lookup(db_path: str, model_name: str) -> Optional[dict]:
    """
    Load a JSON database file and return the first entry whose 'patterns'
    list contains an exact match for model_name after normalization.

    The database file must have the shape: {"models": [{patterns: [...], ...}]}

    Returns None if the file is missing or no pattern matches.
    Substring matches are intentionally rejected (e.g. "kimi-k2" must not
    match a "kimi-k2.5" pattern).
    """
    try:
        with open(db_path, encoding="utf-8") as f:
            entries = json.load(f)["models"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None

    normalized = normalize_model_name(model_name)
    for entry in entries:
        for pattern in entry.get("patterns", []):
            if normalized == pattern.lower():
                return entry
    return None
