from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..config import DEFAULT_BASE_URL, DEFAULT_MODEL, settings
from .prompting import SYSTEM_PROMPT, build_user_prompt, validate_label_response


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: int = 60
    max_retries: int = 2
    sleep_seconds: float = 0.2

    @classmethod
    def from_env(cls, model: str | None = None) -> "DeepSeekConfig":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Use --mock to test without an API key.")
        return cls(
            api_key=api_key,
            base_url=settings.deepseek_base_url,
            model=model or settings.deepseek_model,
        )


def _apply_provider_fields(body: dict[str, Any], base_url: str) -> dict[str, Any]:
    # DeepSeek V4 models enable thinking by default; reasoning tokens would eat
    # the max_tokens budget and can truncate the JSON answer. Other
    # OpenAI-compatible providers may reject this non-standard field.
    if "deepseek" in base_url.lower():
        body["thinking"] = {"type": "disabled"}
    return body


class DeepSeekLabeler:
    def __init__(self, config: DeepSeekConfig) -> None:
        self.config = config

    def label_window(self, window: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = _apply_provider_fields({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(window)},
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "temperature": 0,
            "max_tokens": 700,
        }, self.config.base_url)
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                response = requests.post(url, headers=headers, json=body, timeout=self.config.timeout_seconds)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                return validate_label_response(parsed)
            except Exception as exc:  # noqa: BLE001 - preserve API error details for output files.
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(max(0.2, self.config.sleep_seconds) * (attempt + 1))
        raise RuntimeError(f"DeepSeek labeling failed: {last_error}") from last_error


def test_connection(config: DeepSeekConfig) -> tuple[bool, str]:
    """One minimal chat/completions call to validate base URL + key + model."""
    url = config.base_url.rstrip("/") + "/chat/completions"
    body = _apply_provider_fields({
        "model": config.model,
        "messages": [{"role": "user", "content": "ping"}],
        "stream": False,
        "temperature": 0,
        "max_tokens": 4,
    }, config.base_url)
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(url, headers=headers, json=body, timeout=15)
    except requests.RequestException as exc:
        return False, f"Connection failed: {exc}"
    if response.ok:
        return True, "Connection OK"
    detail = f"HTTP {response.status_code}"
    try:
        message = response.json().get("error", {}).get("message")
        if message:
            detail = f"{detail}: {message}"
    except Exception:  # noqa: BLE001 - non-JSON error bodies are fine.
        pass
    return False, detail

