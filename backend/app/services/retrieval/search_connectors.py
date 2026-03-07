from __future__ import annotations

from typing import Any, Dict, List, Optional
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
                        "source_kind": "external_search_snippet",
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


def _search_exa(
    query: str,
    cap: int,
    *,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    include_text: Optional[list[str]] = None,
    exclude_text: Optional[list[str]] = None,
) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.exa_api_key:
        return []
    payload: Dict[str, Any] = {
        "query": query,
        "numResults": max(1, min(cap, 50)),
        "useAutoprompt": False,
    }
    if include_domains:
        payload["includeDomains"] = include_domains
    if exclude_domains:
        payload["excludeDomains"] = exclude_domains
    if include_text:
        include_value = str(include_text[0]).strip()
        if include_value:
            payload["includeText"] = [include_value]
    if exclude_text:
        exclude_value = str(exclude_text[0]).strip()
        if exclude_value:
            payload["excludeText"] = [exclude_value]
    with httpx.Client(timeout=20) as client:
        resp = client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            json=payload,
        )
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
                "snippet": item.get("text") or item.get("snippet"),
            }
        )
    return results


def _search_exa_similar(url: str, cap: int) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.exa_api_key:
        return []
    payload: Dict[str, Any] = {
        "url": url,
        "numResults": max(1, min(cap, 50)),
    }
    with httpx.Client(timeout=20) as client:
        resp = client.post(
            "https://api.exa.ai/findSimilar",
            headers={"x-api-key": settings.exa_api_key, "Content-Type": "application/json"},
            json=payload,
        )
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
                "snippet": item.get("text") or item.get("snippet"),
            }
        )
    return results


def _search_brave(query: str, cap: int) -> List[Dict[str, Any]]:
    settings = get_settings()
    if not settings.brave_api_key:
        return []
    params = {
        "q": query,
        "count": max(1, min(cap, 20)),
    }
    with httpx.Client(timeout=20) as client:
        resp = client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": settings.brave_api_key},
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()
    results: List[Dict[str, Any]] = []
    web_section = data.get("web") if isinstance(data, dict) else {}
    for item in (web_section.get("results") or [])[:cap]:
        if not isinstance(item, dict):
            continue
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "snippet": item.get("description") or item.get("snippet"),
            }
        )
    return results


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


def provider_results_for_query(
    provider: str,
    *,
    query: str,
    cap: int,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
    include_text: Optional[list[str]] = None,
    exclude_text: Optional[list[str]] = None,
    seed_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    name = str(provider or "").strip().lower()
    if name == "exa":
        if seed_url:
            return _search_exa_similar(seed_url, cap)
        return _search_exa(
            query,
            cap,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_text=include_text,
            exclude_text=exclude_text,
        )
    if name == "brave":
        return _search_brave(query, cap)
    if name == "tavily":
        return _search_tavily(query, cap)
    if name == "serpapi":
        return _search_serpapi(query, cap)
    return []
