from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.config import get_settings
from app.services.retrieval.cache import RetrievalCache
from app.services.retrieval.search_connectors import provider_results_for_query
from app.services.retrieval.url_normalization import dedupe_results, normalize_domain


def _normalize_domains(values: Optional[list[str]]) -> list[str]:
    normalized: list[str] = []
    for value in values or []:
        host = normalize_domain(value)
        if not host:
            continue
        if host not in normalized:
            normalized.append(host)
    return normalized


def _apply_query_modifiers(
    query: str,
    include_terms: list[str],
    exclude_terms: list[str],
    allow_domains: list[str],
    block_domains: list[str],
) -> str:
    parts = [str(query or "").strip()]
    for term in include_terms[:6]:
        if term:
            parts.append(f"\"{term}\"")
    for term in exclude_terms[:6]:
        if term:
            parts.append(f"-\"{term}\"")
    for domain in allow_domains[:3]:
        parts.append(f"site:{domain}")
    for domain in block_domains[:3]:
        parts.append(f"-site:{domain}")
    return " ".join([part for part in parts if part]).strip()


def run_external_search_queries(
    queries: list[dict[str, Any]],
    *,
    provider_order: list[str],
    per_query_cap: int,
    total_cap: int,
    per_domain_cap: int,
    cache: Optional[RetrievalCache] = None,
) -> dict[str, Any]:
    settings = get_settings()
    cache = cache or RetrievalCache()
    errors: list[str] = []
    raw_results: list[dict[str, Any]] = []
    provider_mix: Counter[str] = Counter()
    query_counts: dict[str, int] = {}
    effective_per_query_cap = max(1, int(per_query_cap))
    effective_total_cap = max(1, int(total_cap))

    for provider in provider_order:
        provider_key = str(provider or "").strip().lower()
        if not provider_key:
            continue
        for query_meta in queries:
            query_text = str(query_meta.get("query_text") or "").strip()
            if not query_text:
                continue
            query_id = str(query_meta.get("query_id") or "").strip() or f"{provider_key}:{len(raw_results)}"
            query_type = str(query_meta.get("query_type") or "precision").strip()
            seed_url = str(query_meta.get("seed_url") or "").strip() or None
            if query_type == "seed_similar" and provider_key != "exa":
                continue
            include_terms = [str(term).strip() for term in (query_meta.get("must_include_terms") or []) if str(term).strip()]
            exclude_terms = [str(term).strip() for term in (query_meta.get("must_exclude_terms") or []) if str(term).strip()]
            allow_domains = _normalize_domains(query_meta.get("domain_allowlist"))
            block_domains = _normalize_domains(query_meta.get("domain_blocklist"))

            adjusted_query = _apply_query_modifiers(query_text, include_terms, exclude_terms, allow_domains, block_domains)
            if query_type == "seed_similar" and seed_url:
                adjusted_query = seed_url
            cache_key = (
                f"provider={provider_key}|query={adjusted_query}|seed={seed_url or ''}"
                f"|cap={effective_per_query_cap}|allow={','.join(allow_domains)}|block={','.join(block_domains)}"
                f"|include={','.join(include_terms)}|exclude={','.join(exclude_terms)}"
            )
            cached = cache.get_json("search_provider", cache_key)
            provider_items: list[dict[str, Any]] = []
            if isinstance(cached, list):
                provider_items = cached
            else:
                try:
                    provider_items = provider_results_for_query(
                        provider_key,
                        query=adjusted_query,
                        cap=effective_per_query_cap,
                        include_domains=allow_domains,
                        exclude_domains=block_domains,
                        include_text=include_terms,
                        exclude_text=exclude_terms,
                        seed_url=seed_url if query_type == "seed_similar" else None,
                    )
                except Exception as exc:
                    errors.append(f"{provider_key}:{exc}")
                    provider_items = []
                cache.set_json(
                    "search_provider",
                    cache_key,
                    provider_items,
                    ttl_seconds=max(60, int(settings.retrieval_search_cache_ttl_seconds)),
                )

            retrieved_at = datetime.utcnow().isoformat()
            for rank, item in enumerate(provider_items, start=1):
                url = str(item.get("url") or "").strip()
                if not url:
                    continue
                raw_results.append(
                    {
                        "provider": provider_key,
                        "query_id": query_id,
                        "query_type": query_type,
                        "query_text": adjusted_query,
                        "rank": rank,
                        "url": url,
                        "title": item.get("title"),
                        "snippet": item.get("snippet"),
                        "retrieved_at": retrieved_at,
                    }
                )
            query_counts[query_id] = query_counts.get(query_id, 0) + len(provider_items)

            if effective_total_cap and len(raw_results) >= effective_total_cap * 2:
                break
        if effective_total_cap and len(raw_results) >= effective_total_cap * 2:
            break

    deduped, dedupe_stats = dedupe_results(
        raw_results,
        per_domain_cap=per_domain_cap,
        total_cap=effective_total_cap,
    )
    provider_mix.update([row.get("provider") for row in deduped if row.get("provider")])

    return {
        "results": deduped,
        "errors": errors[:20],
        "provider_mix": dict(provider_mix),
        "query_counts": query_counts,
        "dedupe_stats": dedupe_stats,
    }
