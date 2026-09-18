"""Thin LLM abstraction over Groq (free tier) and Ollama (fully local).

The rest of the codebase only ever calls `get_client().complete(...)`, so the
provider is a one-line config change. Both providers are zero-cost, which the
brief requires.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

import requests

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_PROVIDER,
    LLM_TIMEOUT_S,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the model is unreachable or returns an unusable response."""


class LLMClient(ABC):
    name: str = "base"
    model: str = ""

    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str: ...

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        """Complete and parse JSON, tolerating markdown fences."""
        raw = self.complete(system, user, json_mode=True)
        return parse_json_response(raw)

    def health(self) -> dict[str, Any]:
        try:
            self.complete("Reply with the single word OK.", "ping")
            return {"provider": self.name, "model": self.model, "reachable": True}
        except Exception as exc:  # noqa: BLE001 - health check reports, never raises
            return {
                "provider": self.name,
                "model": self.model,
                "reachable": False,
                "error": str(exc)[:200],
            }


def parse_json_response(raw: str) -> dict[str, Any]:
    """Models wrap JSON in prose or ```json fences often enough to handle here."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError(f"Model did not return valid JSON. Got: {raw[:300]}")


class GroqClient(LLMClient):
    name = "groq"
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = GROQ_API_KEY, model: str = GROQ_MODEL):
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not set. Add it to .env, or set "
                "LLM_PROVIDER=ollama to run fully locally."
            )
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,  # deterministic: this is SQL, not prose
            "max_tokens": 1024,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            resp = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=LLM_TIMEOUT_S,
            )
        except requests.RequestException as exc:
            raise LLMError(f"Groq request failed: {exc}") from exc

        if resp.status_code == 429:
            raise LLMError("Groq free-tier rate limit hit. Wait a moment and retry.")
        if resp.status_code >= 400:
            raise LLMError(f"Groq error {resp.status_code}: {resp.text[:300]}")

        return resp.json()["choices"][0]["message"]["content"]


class OllamaClient(LLMClient):
    name = "ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {"temperature": 0},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate", json=payload, timeout=LLM_TIMEOUT_S
            )
        except requests.RequestException as exc:
            raise LLMError(
                f"Ollama unreachable at {self.base_url}. Is `ollama serve` running?"
            ) from exc

        if resp.status_code >= 400:
            raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
        return resp.json().get("response", "")


_CLIENT: LLMClient | None = None


def get_client(provider: str = LLM_PROVIDER) -> LLMClient:
    """Cached singleton so we do not rebuild the client on every request."""
    global _CLIENT
    if _CLIENT is None:
        if provider == "ollama":
            _CLIENT = OllamaClient()
        elif provider == "groq":
            _CLIENT = GroqClient()
        else:
            raise LLMError(f"Unknown LLM_PROVIDER '{provider}'. Use groq or ollama.")
        logger.info("LLM provider: %s (%s)", _CLIENT.name, _CLIENT.model)
    return _CLIENT
