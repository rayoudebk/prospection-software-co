from __future__ import annotations

import hashlib
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAMS = {
    "gclid",
    "fbclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "ref",
    "source",
    "src",
    "yclid",
    "msclkid",
}


def normalize_url(raw_url: str | None) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return ""
    scheme = (parsed.scheme or "https").lower()
    path = parsed.path or "/"
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_pairs = []
    for key, value in parse_qsl(parsed.query or "", keep_blank_values=False):
        key_lower = key.lower()
        if key_lower.startswith(TRACKING_PARAM_PREFIXES):
            continue
        if key_lower in TRACKING_PARAMS:
            continue
        query_pairs.append((key, value))
    query = urlencode(query_pairs, doseq=True)
    normalized = urlunparse((scheme, host, path, "", query, ""))
    return normalized


def normalize_domain(raw_url: str | None) -> str:
    raw = str(raw_url or "").strip()
    if not raw:
        return ""
    if not raw.startswith(("http://", "https://")):
        raw = f"https://{raw}"
    try:
        parsed = urlparse(raw)
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host or ""


def url_fingerprint(normalized_url: str) -> str:
    return hashlib.sha1(normalized_url.encode("utf-8")).hexdigest()


def domain_fingerprint(domain: str) -> str:
    return hashlib.sha1(domain.encode("utf-8")).hexdigest()


def dedupe_results(
    results: list[dict],
    *,
    per_domain_cap: int,
    total_cap: int,
) -> tuple[list[dict], dict]:
    stats = {
        "total_in": len(results),
        "total_after": 0,
        "duplicates_dropped": 0,
        "per_domain_cap_dropped": 0,
        "total_cap_dropped": 0,
    }
    deduped: list[dict] = []
    seen_urls: set[str] = set()
    domain_counts: dict[str, int] = {}
    effective_total_cap = max(0, int(total_cap))
    effective_domain_cap = max(1, int(per_domain_cap))

    for row in results:
        raw_url = str(row.get("url") or "").strip()
        normalized = normalize_url(raw_url)
        if not normalized:
            stats["duplicates_dropped"] += 1
            continue
        if normalized in seen_urls:
            stats["duplicates_dropped"] += 1
            continue
        domain = normalize_domain(normalized)
        if domain:
            current = domain_counts.get(domain, 0)
            if current >= effective_domain_cap:
                stats["per_domain_cap_dropped"] += 1
                continue
            domain_counts[domain] = current + 1

        seen_urls.add(normalized)
        row["normalized_url"] = normalized
        row["url_fingerprint"] = url_fingerprint(normalized)
        row["domain"] = domain
        row["domain_fingerprint"] = domain_fingerprint(domain) if domain else None
        deduped.append(row)
        if effective_total_cap and len(deduped) >= effective_total_cap:
            break

    if effective_total_cap and len(results) > len(deduped):
        stats["total_cap_dropped"] = max(0, len(results) - len(deduped))
    stats["total_after"] = len(deduped)
    return deduped, stats
