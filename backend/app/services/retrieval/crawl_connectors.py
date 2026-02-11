from __future__ import annotations

from typing import Any, Dict, Optional

import httpx

from app.config import get_settings
from app.services.retrieval.cache import RetrievalCache


def fetch_page_fast(url: str) -> Dict[str, Any]:
    settings = get_settings()
    normalized = str(url or "").strip()
    if not normalized:
        return {"url": url, "content": "", "provider": None, "error": "empty_url"}

    cache = RetrievalCache()
    cached = cache.get_json("url_content", normalized.lower())
    if isinstance(cached, dict):
        return cached

    result: Dict[str, Any] = {"url": normalized, "content": "", "provider": None, "error": None}
    provider_error: Optional[str] = None

    if settings.jina_api_key:
        try:
            with httpx.Client(timeout=15) as client:
                resp = client.get(
                    f"https://r.jina.ai/http://{normalized.replace('https://', '').replace('http://', '')}",
                    headers={"Authorization": f"Bearer {settings.jina_api_key}"},
                )
                resp.raise_for_status()
                result["content"] = str(resp.text or "")[:80000]
                result["provider"] = "jina_reader"
        except Exception as exc:
            provider_error = f"jina:{exc}"

    if not result["content"] and settings.firecrawl_api_key:
        try:
            payload = {"url": normalized, "formats": ["markdown"]}
            with httpx.Client(timeout=20) as client:
                resp = client.post(
                    "https://api.firecrawl.dev/v1/scrape",
                    headers={
                        "Authorization": f"Bearer {settings.firecrawl_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            markdown = ((data.get("data") or {}).get("markdown") if isinstance(data, dict) else "") or ""
            result["content"] = str(markdown)[:80000]
            result["provider"] = "firecrawl"
        except Exception as exc:
            provider_error = f"{provider_error or ''};firecrawl:{exc}".strip(";")

    if not result["content"]:
        result["error"] = provider_error or "external_crawl_unavailable"

    cache.set_json(
        "url_content",
        normalized.lower(),
        result,
        ttl_seconds=max(60, int(settings.retrieval_url_cache_ttl_seconds)),
    )
    return result
