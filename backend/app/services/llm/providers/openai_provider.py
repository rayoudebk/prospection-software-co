from __future__ import annotations

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
        self._client = OpenAI(api_key=settings.openai_api_key)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool = False,
    ) -> str:
        # Search tooling is not attached here; fallback models rely on prompt context.
        del use_web_search
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            )
            content = response.choices[0].message.content if response.choices else ""
            return str(content or "").strip()
        except Exception as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
