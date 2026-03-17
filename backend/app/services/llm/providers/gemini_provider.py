from __future__ import annotations

import time
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

    @staticmethod
    def _is_deep_research_model(model: str) -> bool:
        normalized = str(model or "").strip().lower()
        return "deep-research" in normalized

    @staticmethod
    def _extract_interaction_text(interaction) -> str:
        output = getattr(interaction, "output", None)
        if output is not None:
            text = getattr(output, "text", None)
            if text:
                return str(text).strip()

        outputs = getattr(interaction, "outputs", None) or []
        parts: list[str] = []
        for item in outputs:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part).strip()

    def _generate_with_models(
        self,
        *,
        model: str,
        prompt: str,
        use_web_search: bool,
    ) -> str:
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

    def _generate_with_deep_research(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool,
    ) -> str:
        create_kwargs = {
            "input": prompt,
            "background": True,
            "store": True,
            "agent": model,
            "timeout": timeout_seconds,
        }
        if use_web_search and types is not None:
            create_kwargs["tools"] = [{"type": "google_search"}]
        interaction = self._client.interactions.create(**create_kwargs)

        deadline = time.monotonic() + max(1, int(timeout_seconds))
        status = str(getattr(interaction, "status", "") or "").strip().lower()
        while status in {"queued", "in_progress", "running"}:
            if time.monotonic() >= deadline:
                raise LLMProviderError(
                    f"Gemini deep research timed out waiting for completion: status={status}",
                    retryable=True,
                )
            time.sleep(min(2.0, max(0.1, deadline - time.monotonic())))
            interaction = self._client.interactions.get(
                getattr(interaction, "id"),
                timeout=min(30, max(5, int(timeout_seconds))),
            )
            status = str(getattr(interaction, "status", "") or "").strip().lower()
        if status and status not in {"completed", "succeeded"}:
            raise LLMProviderError(
                f"Gemini deep research ended with status={status}",
                retryable=True,
            )

        text = self._extract_interaction_text(interaction)
        if text:
            return text
        raise LLMProviderError("Gemini deep research returned no text output", retryable=True)

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        timeout_seconds: int,
        use_web_search: bool = False,
    ) -> str:
        try:
            if self._is_deep_research_model(model):
                return self._generate_with_deep_research(
                    model=model,
                    prompt=prompt,
                    timeout_seconds=timeout_seconds,
                    use_web_search=use_web_search,
                )
            return self._generate_with_models(
                model=model,
                prompt=prompt,
                use_web_search=use_web_search,
            )
        except Exception as exc:
            if isinstance(exc, LLMProviderError):
                raise
            raise LLMProviderError(str(exc), retryable=True) from exc
