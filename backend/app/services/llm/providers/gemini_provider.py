from __future__ import annotations

from typing import Optional

from app.config import get_settings
from app.services.llm.types import LLMProviderError

try:
    from google import genai
    from google.genai import types
except Exception:  # pragma: no cover - import guard
    genai = None
    types = None


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.gemini_api_key:
            raise LLMProviderError("GEMINI_API_KEY not configured", retryable=False)
        if genai is None:
            raise LLMProviderError("google-genai SDK unavailable", retryable=False)
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool = False,
    ) -> str:
        try:
            cfg: Optional[types.GenerateContentConfig] = None
            if use_web_search and types is not None:
                google_search_tool = types.Tool(google_search=types.GoogleSearch())
                cfg = types.GenerateContentConfig(tools=[google_search_tool])
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=cfg,
            )
            return str(response.text or "").strip()
        except Exception as exc:
            raise LLMProviderError(str(exc), retryable=True) from exc
