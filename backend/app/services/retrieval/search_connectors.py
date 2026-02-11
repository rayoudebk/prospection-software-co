from __future__ import annotations

from typing import Any, Dict, List
from urllib.parse import urlparse

import httpx

from app.config import get_settings
from app.services.retrieval.cache import RetrievalCache


def _domain_label(url: str) -> str:
    parsed = urlparse(url)
    host = str(parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return "Unknown"
    return host.split(".")[0].replace("-", " ").title()


def _to_candidate_rows(items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        rows.append(
            {
                "name": str(item.get("title") or _domain_label(url)).strip()[:300],
                "website": url,
                "official_website_url": url,
                "discovery_url": url,
                "entity_type": "company",
                "first_party_domains": [],
                "hq_country": "Unknown",
                "likely_verticals": [],
                "employee_estimate": None,
                "capability_signals": [],
                "qualification": {},
                "why_relevant": [
                    {
                        "text": str(item.get("snippet") or "Externally discovered candidate signal.")[:400],
                        "citation_url": url,
                        "dimension": "external_search_seed",
                    }
                ],
                "_origins": [
                    {
                        "origin_type": "external_search_seed",
                        "origin_url": url,
                        "source_name": source_name,
                        "source_run_id": None,
                        "metadata": {},
                    }
                ],
            }
        )
    return rows


def _search_tavily(query: str, cap: int) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.tavily_api_key:
        return []
    payload = {
        "api_key": settings.tavily_api_key,
        "query": query,
        "max_results": max(1, min(cap, 20)),
        "include_answer": False,
        "include_images": False,
    }
    with httpx.Client(timeout=15) as client:
        resp = client.post("https://api.tavily.com/search", json=payload)
        resp.raise_for_status()
        data = resp.json()
    results: List[Dict[str, Any]] = []
    for item in data.get("results", [])[:cap]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("content"),
            }
        )
    return results


def _search_serpapi(query: str, cap: int) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.serpapi_api_key:
        return []
    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "num": max(1, min(cap, 20)),
    }
    with httpx.Client(timeout=15) as client:
        resp = client.get("https://serpapi.com/search.json", params=params)
        resp.raise_for_status()
        data = resp.json()
    results: List[Dict[str, Any]] = []
    for item in data.get("organic_results", [])[:cap]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("link"),
                "snippet": item.get("snippet"),
            }
        )
    return results


def discover_candidates_from_external_search(
    query: str,
    cap: int,
) -> Dict[str, Any]:
    settings = get_settings()
    cache = RetrievalCache()
    cache_key = f"query={query}|cap={cap}"
    cached = cache.get_json("search", cache_key)
    if isinstance(cached, dict):
        return cached

    tavily_items: List[Dict[str, Any]] = []
    serpapi_items: List[Dict[str, Any]] = []
    errors: List[str] = []
    try:
        tavily_items = _search_tavily(query, cap)
    except Exception as exc:
        errors.append(f"tavily:{exc}")
    try:
        serpapi_items = _search_serpapi(query, cap)
    except Exception as exc:
        errors.append(f"serpapi:{exc}")

    output = {
        "query": query,
        "candidates": _to_candidate_rows(tavily_items, "tavily_search")
        + _to_candidate_rows(serpapi_items, "serpapi_search"),
        "errors": errors[:8],
    }
    cache.set_json(
        "search",
        cache_key,
        output,
        ttl_seconds=max(60, int(settings.retrieval_search_cache_ttl_seconds)),
    )
    return output
