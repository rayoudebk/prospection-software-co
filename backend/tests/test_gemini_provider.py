from types import SimpleNamespace

from app.services.llm.providers import gemini_provider as provider_module
from app.services.llm.providers.gemini_provider import GeminiProvider


class _FakeModels:
    def __init__(self) -> None:
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text='{"result":"gemini-chat"}')


class _FakeInteractions:
    def __init__(self) -> None:
        self.create_calls = []
        self.get_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="int_123", status="queued", outputs=[])

    def get(self, interaction_id, **kwargs):
        self.get_calls.append({"interaction_id": interaction_id, **kwargs})
        return SimpleNamespace(
            id=interaction_id,
            status="completed",
            outputs=[SimpleNamespace(text='{"result":"gemini-research"}')],
        )


class _FakeClient:
    def __init__(self) -> None:
        self.models = _FakeModels()
        self.interactions = _FakeInteractions()


def test_gemini_provider_uses_models_for_standard_generation(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        provider_module,
        "get_settings",
        lambda: SimpleNamespace(gemini_api_key="test-key"),
    )
    monkeypatch.setattr(provider_module, "genai", SimpleNamespace(Client=lambda api_key: fake_client))

    provider = GeminiProvider()
    text = provider.generate(
        model="gemini-2.0-flash",
        prompt="Return JSON.",
        timeout_seconds=30,
        use_web_search=False,
    )

    assert text == '{"result":"gemini-chat"}'
    assert len(fake_client.models.calls) == 1
    assert fake_client.interactions.create_calls == []


def test_gemini_provider_uses_interactions_for_deep_research(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        provider_module,
        "get_settings",
        lambda: SimpleNamespace(gemini_api_key="test-key"),
    )
    monkeypatch.setattr(provider_module, "genai", SimpleNamespace(Client=lambda api_key: fake_client))
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    provider = GeminiProvider()
    text = provider.generate(
        model="deep-research-pro-preview-12-2025",
        prompt="Return JSON.",
        timeout_seconds=30,
        use_web_search=True,
    )

    assert text == '{"result":"gemini-research"}'
    assert fake_client.models.calls == []
    assert len(fake_client.interactions.create_calls) == 1
    assert fake_client.interactions.create_calls[0]["background"] is True
    assert fake_client.interactions.create_calls[0]["agent"] == "deep-research-pro-preview-12-2025"
    assert fake_client.interactions.get_calls[0]["interaction_id"] == "int_123"
