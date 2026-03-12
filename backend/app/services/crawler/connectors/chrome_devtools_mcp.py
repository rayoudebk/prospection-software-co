from __future__ import annotations

from typing import Any, Dict, Optional
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser

from app.config import get_settings


def _normalize_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return ""
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    parsed = urlparse(normalized)
    if not parsed.netloc:
        return ""
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""
    return f"{parsed.scheme or 'https'}://{parsed.netloc}{path}{query}"


def _extract_visible_text(html: str) -> str:
    tree = HTMLParser(str(html or ""))
    if tree is None:
        return ""
    body = tree.css_first("body")
    text = body.text(separator=" ", strip=True) if body else tree.text(separator=" ", strip=True)
    return " ".join(str(text or "").split())


def _render_via_endpoint(url: str, timeout_seconds: int, endpoint: str) -> Dict[str, Any]:
    with httpx.Client(timeout=max(5, timeout_seconds + 5)) as client:
        response = client.post(
            endpoint,
            json={"url": url, "timeout_seconds": timeout_seconds, "expand_interactive": True},
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        payload = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    if not isinstance(payload, dict):
        return {"url": url, "final_url": url, "provider": "chrome_devtools_mcp_endpoint", "content": "", "error": "invalid_payload"}
    html = str(payload.get("html") or "")
    text = str(payload.get("content") or payload.get("text") or "").strip()
    if not text and html:
        text = _extract_visible_text(html)
    final_url = str(payload.get("final_url") or payload.get("url") or url).strip() or url
    return {
        "url": url,
        "final_url": final_url,
        "provider": str(payload.get("provider") or "chrome_devtools_mcp_endpoint"),
        "content": text,
        "html": html,
        "error": None if text else "empty_rendered_content",
    }


def _render_via_playwright(url: str, timeout_seconds: int) -> Dict[str, Any]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # pragma: no cover - optional dependency
        return {
            "url": url,
            "final_url": url,
            "provider": "chrome_devtools_mcp_playwright",
            "content": "",
            "error": f"playwright_unavailable:{exc}",
        }

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--disable-dev-shm-usage", "--disable-gpu", "--no-sandbox"],
            )
            page = browser.new_page()
            page.goto(url, wait_until="networkidle", timeout=max(5000, timeout_seconds * 1000))
            page.wait_for_timeout(500)
            _expand_interactive_sections(page)
            final_url = str(page.url or url)
            html = page.content()
            text: Optional[str] = None
            try:
                text = page.locator("body").inner_text(timeout=3000)
            except Exception:
                text = None
            browser.close()
        content = " ".join(str(text or _extract_visible_text(html)).split())
        return {
            "url": url,
            "final_url": final_url,
            "provider": "chrome_devtools_mcp_playwright",
            "content": content,
            "html": html,
            "error": None if content else "empty_rendered_content",
        }
    except Exception as exc:
        return {
            "url": url,
            "final_url": url,
            "provider": "chrome_devtools_mcp_playwright",
            "content": "",
            "html": "",
            "error": f"playwright_render_failed:{exc}",
        }


def _should_fallback_to_playwright(result: Dict[str, Any]) -> bool:
    content = " ".join(str(result.get("content") or "").split())
    if not content:
        return True
    if len(content) < 1200:
        return True
    weak_markers = (
        "front office titres",
        "back office titres",
        "documentation api",
        "plateforme wealth management",
    )
    if len(content) < 2200 and sum(1 for marker in weak_markers if marker in content.lower()) >= 2:
        return True
    return False


def _expand_interactive_sections(page: Any) -> None:
    """Best-effort expansion of accordions/disclosures on product-style pages."""
    try:
        page.locator("details").evaluate_all(
            """(nodes) => {
                for (const node of nodes) {
                    try { node.open = true; } catch (e) {}
                }
            }"""
        )
    except Exception:
        pass

    candidate_selector = ",".join(
        [
            "summary",
            "button[aria-expanded='false']",
            "[role='button'][aria-expanded='false']",
            "[data-state='closed'] button",
            ".accordion button",
            "[class*='accordion'] button",
            "[class*='collapse'] button",
            "[class*='disclosure'] button",
        ]
    )
    keyword_pattern = (
        "fonction|detail|feature|functionality|capability|module|workflow|"
        "portfolio|order|trade|compliance|reporting|risk|data|integration"
    )

    try:
        candidates = page.locator(candidate_selector)
        count = min(candidates.count(), 40)
    except Exception:
        return

    for idx in range(count):
        try:
            handle = candidates.nth(idx)
            if not handle.is_visible(timeout=250):
                continue
            text = " ".join(str(handle.inner_text(timeout=400) or "").split())[:160]
            aria_controls = str(handle.get_attribute("aria-controls") or "")
            class_name = str(handle.get_attribute("class") or "")
            lowered = f"{text} {aria_controls} {class_name}".lower()
            if text and not __import__("re").search(keyword_pattern, lowered):
                expanded = str(handle.get_attribute("aria-expanded") or "").lower()
                if expanded != "false":
                    continue
            handle.click(timeout=800, force=True)
            page.wait_for_timeout(120)
        except Exception:
            continue


def render_page_via_chrome_devtools_mcp(
    url: str,
    *,
    timeout_seconds: Optional[int] = None,
    max_chars: int = 120_000,
) -> Dict[str, Any]:
    """
    Render a page through a Chrome-based connector.

    Priority:
    1) configured MCP-style HTTP endpoint
    2) local Playwright Chromium fallback
    """
    settings = get_settings()
    normalized = _normalize_url(url)
    if not normalized:
        return {
            "url": url,
            "final_url": url,
            "provider": None,
            "content": "",
            "error": "invalid_url",
        }

    timeout = max(5, int(timeout_seconds or settings.chrome_mcp_timeout_seconds))
    endpoint = str(settings.chrome_mcp_endpoint or "").strip()
    result: Dict[str, Any]
    if endpoint:
        try:
            result = _render_via_endpoint(normalized, timeout, endpoint)
            if _should_fallback_to_playwright(result):
                playwright_result = _render_via_playwright(normalized, timeout)
                if len(str(playwright_result.get("content") or "")) > len(str(result.get("content") or "")) + 80:
                    result = playwright_result
        except Exception as exc:
            result = {
                "url": normalized,
                "final_url": normalized,
                "provider": "chrome_devtools_mcp_endpoint",
                "content": "",
                "error": f"endpoint_failed:{exc}",
            }
    else:
        result = _render_via_playwright(normalized, timeout)

    content = str(result.get("content") or "")
    if max_chars > 0 and len(content) > max_chars:
        result["content"] = content[:max_chars]
    result["content_length"] = len(str(result.get("content") or ""))
    return result
