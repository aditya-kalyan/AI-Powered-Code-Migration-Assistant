"""Thin wrapper around the OpenAI client plus small parsing helpers shared by
every node that talks to the LLM."""

from __future__ import annotations

import json
import os
import re

from openai import OpenAI


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running the app, e.g.\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "or place it in a .env file and load it before launching app.py."
        )
    return OpenAI(api_key=api_key)


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n", "", text)
    text = re.sub(r"\n```$", "", text)
    return text.strip()


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction: strips fences, tries a direct parse, then
    falls back to grabbing the first {...} block in the text."""
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")
