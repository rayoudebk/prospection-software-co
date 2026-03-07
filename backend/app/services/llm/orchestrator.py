from __future__ import annotations

import time
from typing import Dict, List, Tuple

from app.config import get_settings
from app.services.llm.providers.anthropic_provider import AnthropicProvider
from app.services.llm.providers.gemini_provider import GeminiProvider
from app.services.llm.providers.openai_provider import OpenAIProvider
from app.services.llm.types import (
    LLMOrchestrationError,
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    ModelAttemptTrace,
    classify_retryable_error,
    now_iso,
)


class LLMOrchestrator:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._providers: Dict[str, object] = {}

    def _provider(self, name: str):
        key = str(name or "").strip().lower()
        if key in self._providers:
            return self._providers[key]
        if key == "gemini":
            instance = GeminiProvider()
        elif key == "openai":
            instance = OpenAIProvider()
        elif key == "anthropic":
            instance = AnthropicProvider()
        else:
            raise LLMProviderError(f"Unsupported LLM provider: {name}", retryable=False)
        self._providers[key] = instance
        return instance

    def _routes_for_stage(self, stage_name: str) -> List[Tuple[str, str]]:
        routes = self._settings.stage_model_routes(stage_name)
        return routes if routes else [("gemini", "gemini-2.0-flash")]

    def run_stage(self, request: LLMRequest) -> LLMResponse:
        attempts: List[ModelAttemptTrace] = []
        routes = self._routes_for_stage(request.stage.value)
        max_attempts = max(1, int(self._settings.stage_retry_max_attempts))
        backoff = max(0.0, float(self._settings.stage_retry_backoff_seconds))

        for provider_name, model in routes:
            retry_count = 0
            while retry_count < max_attempts:
                started = now_iso()
                t0 = time.perf_counter()
                try:
                    provider = self._provider(provider_name)
                    text = provider.generate(
                        model=model,
                        prompt=request.prompt,
                        timeout_seconds=max(1, int(request.timeout_seconds)),
                        use_web_search=bool(request.use_web_search),
                    )
                    ended = now_iso()
                    attempts.append(
                        ModelAttemptTrace(
                            stage=request.stage.value,
                            provider=provider_name,
                            model=model,
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                            status="success",
                            retry_count=retry_count,
                            started_at=started,
                            ended_at=ended,
                        )
                    )
                    return LLMResponse(
                        text=text,
                        provider=provider_name,
                        model=model,
                        attempts=attempts,
                    )
                except Exception as exc:
                    ended = now_iso()
                    retryable = False
                    if isinstance(exc, LLMProviderError):
                        retryable = bool(exc.retryable)
                    if not retryable:
                        retryable = classify_retryable_error(exc)
                    attempts.append(
                        ModelAttemptTrace(
                            stage=request.stage.value,
                            provider=provider_name,
                            model=model,
                            latency_ms=int((time.perf_counter() - t0) * 1000),
                            status="retryable_error" if retryable else "terminal_error",
                            retry_count=retry_count,
                            error_class=exc.__class__.__name__,
                            error_message=str(exc)[:500],
                            started_at=started,
                            ended_at=ended,
                        )
                    )
                    if retryable and retry_count < max_attempts - 1:
                        if backoff > 0:
                            time.sleep(backoff * (retry_count + 1))
                        retry_count += 1
                        continue
                    break

        raise LLMOrchestrationError(
            f"All model routes failed for stage={request.stage.value}",
            attempts=attempts,
        )
