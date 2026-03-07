from app.services.llm.orchestrator import LLMOrchestrator
from app.services.llm.types import (
    LLMOrchestrationError,
    LLMProviderError,
    LLMRequest,
    LLMStage,
)


class _FakeProvider:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, *, model: str, prompt: str, timeout_seconds: int, use_web_search: bool = False):
        if not self._responses:
            return ""
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return str(nxt)


def test_orchestrator_falls_back_to_next_provider(monkeypatch):
    orchestrator = LLMOrchestrator()
    orchestrator._settings.stage_retry_backoff_seconds = 0
    routes = [("gemini", "gemini-2.0-flash"), ("openai", "gpt-4.1-mini")]
    monkeypatch.setattr(orchestrator, "_routes_for_stage", lambda _stage: routes)
    providers = {
        "gemini": _FakeProvider([LLMProviderError("schema invalid", retryable=False)]),
        "openai": _FakeProvider(['[{"name":"Acme","website":"https://acme.test"}]']),
    }
    monkeypatch.setattr(orchestrator, "_provider", lambda name: providers[name])

    response = orchestrator.run_stage(
        LLMRequest(
            stage=LLMStage.discovery_retrieval,
            prompt="Return JSON.",
            timeout_seconds=30,
            use_web_search=True,
            expect_json=True,
        )
    )
    assert response.provider == "openai"
    assert response.model == "gpt-4.1-mini"
    assert len(response.attempts) == 2
    assert response.attempts[0].provider == "gemini"
    assert response.attempts[1].provider == "openai"


def test_orchestrator_retries_retryable_provider_errors(monkeypatch):
    orchestrator = LLMOrchestrator()
    orchestrator._settings.stage_retry_backoff_seconds = 0
    orchestrator._settings.stage_retry_max_attempts = 2
    monkeypatch.setattr(orchestrator, "_routes_for_stage", lambda _stage: [("gemini", "gemini-2.0-flash")])
    provider = _FakeProvider(
        [
            LLMProviderError("timeout", retryable=True),
            '[{"name":"Bravo","website":"https://bravo.test"}]',
        ]
    )
    monkeypatch.setattr(
        orchestrator,
        "_provider",
        lambda _name: provider,
    )

    response = orchestrator.run_stage(
        LLMRequest(
            stage=LLMStage.structured_normalization,
            prompt="Return JSON.",
            timeout_seconds=20,
            expect_json=True,
        )
    )
    assert response.provider == "gemini"
    assert len(response.attempts) == 2
    assert response.attempts[0].status == "retryable_error"
    assert response.attempts[1].status == "success"


def test_orchestrator_raises_when_all_routes_fail(monkeypatch):
    orchestrator = LLMOrchestrator()
    orchestrator._settings.stage_retry_backoff_seconds = 0
    monkeypatch.setattr(orchestrator, "_routes_for_stage", lambda _stage: [("gemini", "gemini-2.0-flash")])
    monkeypatch.setattr(
        orchestrator,
        "_provider",
        lambda _name: _FakeProvider([LLMProviderError("schema invalid", retryable=False)]),
    )

    try:
        orchestrator.run_stage(
            LLMRequest(
                stage=LLMStage.context_summary,
                prompt="Summarize.",
                timeout_seconds=20,
            )
        )
    except LLMOrchestrationError as exc:
        assert exc.attempts
        assert exc.attempts[0].status in {"terminal_error", "retryable_error"}
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected LLMOrchestrationError")
