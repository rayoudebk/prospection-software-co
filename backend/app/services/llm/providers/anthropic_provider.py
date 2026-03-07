from __future__ import annotations

from app.config import get_settings
from app.services.llm.types import LLMProviderError

try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover - import guard
    Anthropic = None


class AnthropicProvider:
    name = "anthropic"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise LLMProviderError("ANTHROPIC_API_KEY not configured", retryable=False)
        if Anthropic is None:
            raise LLMProviderError("anthropic SDK unavailable", retryable=False)
        self._client = Anthropic(api_key=settings.anthropic_api_key)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool = False,
    ) -> str:
        del use_web_search
        try:
            response = self._client.messages.create(
                model=model,
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            )
            text_parts = []
            for block in getattr(response, "content", []) or []:
                value = getattr(block, "text", None)
                if value:
                    text_parts.append(str(value))
            return "\n".join(text_parts).strip()
        except Exception as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
