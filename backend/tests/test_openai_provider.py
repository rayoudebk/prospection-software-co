from types import SimpleNamespace

from app.services.llm.providers import openai_provider as provider_module
from app.services.llm.providers.openai_provider import OpenAIProvider


class _FakeChatCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"result":"chat"}'))]
        )


class _FakeResponses:
    def __init__(self) -> None:
        self.create_calls = []
        self.retrieve_calls = []

    def create(self, **kwargs):
        self.create_calls.append(kwargs)
        return SimpleNamespace(id="resp_123", status="queued", output_text=None)

    def retrieve(self, response_id, **kwargs):
        self.retrieve_calls.append({"response_id": response_id, **kwargs})
        return SimpleNamespace(id=response_id, status="completed", output_text='{"result":"research"}')


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())
        self.responses = _FakeResponses()


class _FakeClientWithoutResponses:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())


def test_openai_provider_uses_chat_completions_for_standard_models(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        provider_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test-key"),
    )
    monkeypatch.setattr(provider_module, "OpenAI", lambda api_key: fake_client)

    provider = OpenAIProvider()
    text = provider.generate(
        model="gpt-4.1-mini",
        prompt="Return JSON.",
        timeout_seconds=30,
        use_web_search=False,
    )

    assert text == '{"result":"chat"}'
    assert len(fake_client.chat.completions.calls) == 1
    assert fake_client.responses.create_calls == []


def test_openai_provider_uses_background_responses_for_deep_research(monkeypatch):
    fake_client = _FakeClient()

    monkeypatch.setattr(
        provider_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test-key"),
    )
    monkeypatch.setattr(provider_module, "OpenAI", lambda api_key: fake_client)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)

    provider = OpenAIProvider()
    text = provider.generate(
        model="o4-mini-deep-research",
        prompt="Return JSON.",
        timeout_seconds=30,
        use_web_search=True,
    )

    assert text == '{"result":"research"}'
    assert fake_client.chat.completions.calls == []
    assert len(fake_client.responses.create_calls) == 1
    assert fake_client.responses.create_calls[0]["background"] is True
    assert fake_client.responses.create_calls[0]["tools"] == [{"type": "web_search"}]
    assert fake_client.responses.retrieve_calls[0]["response_id"] == "resp_123"


def test_openai_provider_falls_back_to_http_for_responses_api(monkeypatch):
    fake_client = _FakeClientWithoutResponses()

    class _FakeHttpResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    monkeypatch.setattr(
        provider_module,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test-key"),
    )
    monkeypatch.setattr(provider_module, "OpenAI", lambda api_key: fake_client)
    monkeypatch.setattr(provider_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        provider_module.requests,
        "post",
        lambda *args, **kwargs: _FakeHttpResponse({"id": "resp_123", "status": "queued"}),
    )
    monkeypatch.setattr(
        provider_module.requests,
        "get",
        lambda *args, **kwargs: _FakeHttpResponse(
            {"id": "resp_123", "status": "completed", "output_text": '{"result":"http-research"}'}
        ),
    )

    provider = OpenAIProvider()
    text = provider.generate(
        model="o4-mini-deep-research",
        prompt="Return JSON.",
        timeout_seconds=30,
        use_web_search=True,
    )

    assert text == '{"result":"http-research"}'
