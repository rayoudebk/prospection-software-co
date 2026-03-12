from app.services.crawler.connectors import chrome_devtools_mcp


def test_render_page_prefers_playwright_when_endpoint_result_is_too_thin(monkeypatch):
    monkeypatch.setattr(
        chrome_devtools_mcp,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {"chrome_mcp_timeout_seconds": 20, "chrome_mcp_endpoint": "https://renderer.internal/render"},
        )(),
    )
    monkeypatch.setattr(
        chrome_devtools_mcp,
        "_render_via_endpoint",
        lambda url, timeout_seconds, endpoint: {
            "url": url,
            "final_url": url,
            "provider": "endpoint",
            "content": "Front office titres Back office titres Plateforme Wealth Management",
            "error": None,
        },
    )
    monkeypatch.setattr(
        chrome_devtools_mcp,
        "_render_via_playwright",
        lambda url, timeout_seconds: {
            "url": url,
            "final_url": url,
            "provider": "playwright",
            "content": "Front Office detailed accordion content " * 120,
            "error": None,
        },
    )

    result = chrome_devtools_mcp.render_page_via_chrome_devtools_mcp("https://4tpm.fr/platform/front-office")

    assert result["provider"] == "playwright"
    assert "detailed accordion content" in result["content"]


def test_render_page_prefers_playwright_first_when_requested(monkeypatch):
    monkeypatch.setattr(
        chrome_devtools_mcp,
        "get_settings",
        lambda: type(
            "Settings",
            (),
            {"chrome_mcp_timeout_seconds": 20, "chrome_mcp_endpoint": "https://renderer.internal/render"},
        )(),
    )
    monkeypatch.setattr(
        chrome_devtools_mcp,
        "_render_via_playwright",
        lambda url, timeout_seconds: {
            "url": url,
            "final_url": url,
            "provider": "playwright",
            "content": "Expanded accordion content",
            "html": "<html><body><button>Expanded accordion content</button></body></html>",
            "error": None,
        },
    )
    endpoint_called = {"value": False}

    def _endpoint(url, timeout_seconds, endpoint):
        endpoint_called["value"] = True
        return {
            "url": url,
            "final_url": url,
            "provider": "endpoint",
            "content": "Thin shell",
            "html": "<html><body>Thin shell</body></html>",
            "error": None,
        }

    monkeypatch.setattr(chrome_devtools_mcp, "_render_via_endpoint", _endpoint)

    result = chrome_devtools_mcp.render_page_via_chrome_devtools_mcp(
        "https://4tpm.fr/platform/front-office",
        prefer_playwright=True,
    )

    assert result["provider"] == "playwright"
    assert "Expanded accordion content" in result["content"]
    assert endpoint_called["value"] is False
