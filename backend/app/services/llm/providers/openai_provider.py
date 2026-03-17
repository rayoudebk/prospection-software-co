from __future__ import annotations

import time
from typing import Any
import requests

from app.config import get_settings
from app.services.llm.types import LLMProviderError

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - import guard
    OpenAI = None


class OpenAIProvider:
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise LLMProviderError("OPENAI_API_KEY not configured", retryable=False)
        if OpenAI is None:
            raise LLMProviderError("openai SDK unavailable", retryable=False)
        self._api_key = settings.openai_api_key
        self._client = OpenAI(api_key=settings.openai_api_key)

    @staticmethod
    def _is_deep_research_model(model: str) -> bool:
        return "deep-research" in str(model or "").strip().lower()

    @staticmethod
    def _response_value(response: Any, key: str) -> Any:
        if isinstance(response, dict):
            return response.get(key)
        return getattr(response, key, None)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = OpenAIProvider._response_value(response, "output_text")
        if output_text:
            return str(output_text).strip()

        parts: list[str] = []
        for item in OpenAIProvider._response_value(response, "output") or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for content in getattr(item, "content", []) or []:
                    if getattr(content, "type", None) == "output_text":
                        text = getattr(content, "text", None)
                        if text:
                            parts.append(str(text))
                continue
            if isinstance(item, dict):
                if item.get("type") != "message":
                    continue
                for content in item.get("content") or []:
                    if isinstance(content, dict) and content.get("type") == "output_text" and content.get("text"):
                        parts.append(str(content["text"]))
        return "\n".join(part for part in parts if part).strip()

    def _responses_create(self, *, payload: dict[str, Any], timeout_seconds: int) -> Any:
        if hasattr(self._client, "responses"):
            return self._client.responses.create(**payload, timeout=timeout_seconds)
        response = requests.post(
            "https://api.openai.com/v1/responses",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _responses_retrieve(self, *, response_id: str, timeout_seconds: int) -> Any:
        if hasattr(self._client, "responses"):
            return self._client.responses.retrieve(response_id, timeout=timeout_seconds)
        response = requests.get(
            f"https://api.openai.com/v1/responses/{response_id}",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        return response.json()

    def _generate_with_chat(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
    ) -> str:
        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=timeout_seconds,
        )
        content = response.choices[0].message.content if response.choices else ""
        return str(content or "").strip()

    def _generate_with_responses(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool,
    ) -> str:
        create_kwargs: dict[str, Any] = {
            "model": model,
            "input": prompt,
        }
        if use_web_search:
            create_kwargs["tools"] = [{"type": "web_search"}]

        if self._is_deep_research_model(model):
            create_kwargs["background"] = True

        response = self._responses_create(payload=create_kwargs, timeout_seconds=timeout_seconds)

        if self._is_deep_research_model(model):
            deadline = time.monotonic() + max(1, int(timeout_seconds))
            status = str(self._response_value(response, "status") or "").strip().lower()
            while status in {"queued", "in_progress"}:
                if time.monotonic() >= deadline:
                    raise LLMProviderError(
                        f"OpenAI deep research timed out waiting for completion: status={status}",
                        retryable=True,
                    )
                time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
                response = self._responses_retrieve(
                    response_id=str(self._response_value(response, "id") or ""),
                    timeout_seconds=min(30, max(5, int(timeout_seconds))),
                )
                status = str(self._response_value(response, "status") or "").strip().lower()
            if status and status != "completed":
                raise LLMProviderError(
                    f"OpenAI deep research ended with status={status}",
                    retryable=True,
                )

        text = self._extract_response_text(response)
        if text:
            return text
        raise LLMProviderError("OpenAI response returned no text output", retryable=True)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool = False,
    ) -> str:
        try:
            if use_web_search or self._is_deep_research_model(model):
                return self._generate_with_responses(
                    model=model,
                    prompt=prompt,
                    timeout_seconds=timeout_seconds,
                    use_web_search=use_web_search,
                )
            return self._generate_with_chat(
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            if isinstance(exc, LLMProviderError):
                raise
            raise LLMProviderError(str(exc), retryable=True) from exc
