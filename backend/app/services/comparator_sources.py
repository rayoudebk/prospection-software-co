"""Comparator source ingestion utilities."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse
import re

import httpx
from selectolax.parser import HTMLParser


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
)
MAX_FILTERED_MENTIONS = 250
AGGREGATOR_DOMAINS = {
    "thewealthmosaic.com",
    "thewealthmosaic.co.uk",
}
EXTERNAL_HOST_BLOCKLIST = {
    "linkedin.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
    "vimeo.com",
    "wikipedia.org",
}


@dataclass
class SourceConfig:
    source_name: str
    seed_url: str
    parser: Callable[[str, str], List[Dict[str, Any]]]
    default_max_pages: int = 1


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _normalize_company_url(raw_url: Optional[str], base_url: str) -> Optional[str]:
    if not raw_url:
        return None
    raw = raw_url.strip()
    if not raw or raw.startswith(("mailto:", "tel:", "#")):
        return None
    if "{{" in raw or "}}" in raw:
        return None
    absolute = urljoin(base_url, raw)
    parsed = urlparse(absolute)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    if not host:
        return None
    return f"{parsed.scheme}://{host}{parsed.path}"


def _extract_category_tags(tree: HTMLParser) -> List[str]:
    tags: List[str] = []
    for selector in ["h1", "h2", ".breadcrumb", ".page-title", ".need-title", ".taxonomy"]:
        for node in tree.css(selector):
            text = _clean_text(node.text())
            if not text:
                continue
            for token in re.split(r"[,|/]", text):
                normalized = _clean_text(token).lower()
                if not normalized:
                    continue
                if len(normalized) < 4:
                    continue
                if normalized in {"home", "needs", "providers"}:
                    continue
                tags.append(normalized[:80])
    deduped: List[str] = []
    seen = set()
    for tag in tags:
        if tag in seen:
            continue
        seen.add(tag)
        deduped.append(tag)
    return deduped[:8]


def _extract_listing_snippets(anchor_node: Any) -> List[str]:
    snippets: List[str] = []
    current = anchor_node
    for _ in range(3):
        if not current:
            break
        text = _clean_text(current.text())
        if 24 <= len(text) <= 380:
            snippets.append(text)
        current = current.parent
    deduped: List[str] = []
    seen = set()
    for snippet in snippets:
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    return deduped[:3]


def _label_from_vendor_path(company_url: str) -> Optional[str]:
    """Infer vendor/company label from /vendors/{vendor_slug}/... path."""
    try:
        path = urlparse(company_url).path.strip("/")
        parts = path.split("/")
        if len(parts) >= 2 and parts[0] in {"vendor", "vendors"}:
            slug = parts[1].strip()
            if slug:
                return slug.replace("-", " ").replace("_", " ").title()
    except Exception:
        return None
    return None


def _extract_vendor_slugs(company_url: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Extract company and solution slugs from /vendors/{company_slug}/{solution_slug} URL."""
    if not company_url:
        return None, None
    try:
        path = urlparse(company_url).path.strip("/")
        parts = [part.strip() for part in path.split("/") if part.strip()]
        if len(parts) >= 2 and parts[0] in {"vendor", "vendors"}:
            company_slug = parts[1] or None
            solution_slug = parts[2] if len(parts) >= 3 else None
            return company_slug, (solution_slug or None)
    except Exception:
        return None, None
    return None, None


def _slug_to_label(slug: Optional[str]) -> Optional[str]:
    normalized = str(slug or "").strip()
    if not normalized:
        return None
    return normalized.replace("-", " ").replace("_", " ").title()


def _extract_website_address_from_html(profile_url: str, html: str) -> Optional[str]:
    """Extract explicit 'Website Address' value if present in profile markup."""
    candidates: list[str] = []
    patterns = [
        r"Website\s*Address.{0,260}?href=[\"']([^\"']+)[\"']",
        r"Website\s*Address.{0,180}?(https?://[^\s\"'<>]+)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, html, flags=re.IGNORECASE | re.DOTALL):
            raw = str(match.group(1) or "").strip()
            if raw:
                candidates.append(raw)
    for raw in candidates:
        normalized = _normalize_company_url(raw, profile_url)
        if not normalized:
            continue
        host = (urlparse(normalized).netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if not host or _is_aggregator_domain(host):
            continue
        if any(blocked == host or host.endswith(f".{blocked}") for blocked in EXTERNAL_HOST_BLOCKLIST):
            continue
        return normalized
    return None


def _extract_official_website_from_block(anchor_block: Any, listing_url: str) -> Optional[str]:
    """Best-effort extraction of official website URL from a listing card block."""
    listing_host = (urlparse(listing_url).netloc or "").lower()
    if listing_host.startswith("www."):
        listing_host = listing_host[4:]

    candidates: list[tuple[int, str]] = []
    for anchor in anchor_block.css("a[href]"):
        href = (anchor.attributes.get("href") or "").strip()
        if not href:
            continue
        absolute = _normalize_company_url(href, listing_url)
        if not absolute:
            continue
        parsed = urlparse(absolute)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if not host or host == listing_host or _is_aggregator_domain(host):
            continue
        if any(blocked == host or host.endswith(f".{blocked}") for blocked in EXTERNAL_HOST_BLOCKLIST):
            continue
        anchor_text = _clean_text(anchor.text()).lower()
        score = 0
        if any(token in anchor_text for token in ("website", "visit", "official", "company site")):
            score += 5
        if parsed.path in {"", "/"}:
            score += 2
        if len(anchor_text) <= 24:
            score += 1
        candidates.append((score, absolute))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def parse_wealth_mosaic_listing(html: str, listing_url: str) -> List[Dict[str, Any]]:
    """Parse Wealth Mosaic need/listing page into mention records."""
    tree = HTMLParser(html)
    if tree is None:
        return []

    category_tags = _extract_category_tags(tree)
    mentions: List[Dict[str, Any]] = []
    seen_keys = set()
    base_host = (urlparse(listing_url).netloc or "").lower()
    if base_host.startswith("www."):
        base_host = base_host[4:]
    internal_allowed_markers = (
        "/company/",
        "/companies/",
        "/provider/",
        "/providers/",
        "/solution-provider/",
        "/solution-providers/",
        "/vendor/",
        "/vendors/",
    )
    internal_blocked_markers = (
        "/login",
        "/registration",
        "/terms",
        "/privacy",
        "/suggest",
        "/become",
        "/market/",
        "/needs/",
        "/knowledge/",
        "/events/",
        "/news/",
    )
    generic_anchor_text = {
        "next",
        "previous",
        "read more",
        "see all",
        "view solution",
        "request twm support",
        "register interest",
        "suggest a solution provider",
        "privacy",
        "terms and conditions",
        "sign up or login if a wealth manager",
        "register as a solution provider",
    }

    # First-pass: parse product cards, which carry the cleanest mentions.
    for block in tree.css("div.sol-block"):
        candidate_links = []
        for anchor in block.css("a[href]"):
            href = (anchor.attributes.get("href") or "").strip()
            if not href:
                continue
            if "/vendors/" not in href and "/vendor/" not in href:
                continue
            candidate_links.append((anchor, href))
        if not candidate_links:
            continue
        # Prefer heading anchor when available.
        chosen_anchor, chosen_href = candidate_links[0]
        for anchor, href in candidate_links:
            text = _clean_text(anchor.text())
            if text and len(text) > 3:
                chosen_anchor, chosen_href = anchor, href
                break

        profile_url = _normalize_company_url(chosen_href, listing_url)
        if not profile_url:
            continue

        title_node = block.css_first("h3")
        product_title = _clean_text(title_node.text() if title_node else chosen_anchor.text())
        company_slug, solution_slug = _extract_vendor_slugs(profile_url)
        company_name = _slug_to_label(company_slug) or _label_from_vendor_path(profile_url) or product_title
        if not company_name:
            continue
        official_website_url = _extract_official_website_from_block(block, listing_url)
        entity_type = "solution" if solution_slug else "company"

        description_node = block.css_first("p")
        snippet = _clean_text(description_node.text() if description_node else "")
        snippets = []
        if product_title:
            snippets.append(f"Solution: {product_title}")
        if snippet:
            snippets.append(snippet)

        key = (company_name.lower(), profile_url.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)
        mentions.append(
            {
                "company_name": company_name[:300],
                "company_url": profile_url,
                "profile_url": profile_url,
                "official_website_url": official_website_url,
                "company_slug": company_slug,
                "solution_slug": solution_slug,
                "entity_type": entity_type,
                "listing_url": listing_url,
                "category_tags": category_tags,
                "listing_text_snippets": snippets,
                "provenance": {
                    "source_name": "wealth_mosaic",
                    "listing_url": listing_url,
                    "anchor_href": chosen_href[:1000],
                    "anchor_text": company_name[:300],
                    "extractor": "sol_block",
                    "solution_title": product_title[:300] if product_title else None,
                    "company_slug": company_slug,
                    "solution_slug": solution_slug,
                },
            }
        )

    # Second-pass fallback: broad anchor scanning.
    for anchor in tree.css("a[href]"):
        href = anchor.attributes.get("href", "")
        company_name = _clean_text(anchor.text())
        if not company_name or len(company_name) < 2:
            continue

        profile_url = _normalize_company_url(href, listing_url)
        if not profile_url:
            continue

        parsed = urlparse(profile_url)
        path = parsed.path.lower()
        host = parsed.netloc.lower()

        # Prefer provider/company profile links, skip nav/auth/legal paths.
        if host == base_host:
            if any(marker in path for marker in internal_blocked_markers):
                continue
            if not any(marker in path for marker in internal_allowed_markers):
                continue

        if company_name.lower() in generic_anchor_text:
            continue
        if re.search(r"^(subscribe|login|register|privacy|terms)", company_name, flags=re.IGNORECASE):
            continue
        if len(company_name) < 3:
            continue

        company_slug, solution_slug = _extract_vendor_slugs(profile_url)
        normalized_company_name = _slug_to_label(company_slug) or company_name
        entity_type = "solution" if solution_slug else "company"
        key = (normalized_company_name.lower(), profile_url.lower())
        if key in seen_keys:
            continue
        seen_keys.add(key)

        mentions.append(
            {
                "company_name": normalized_company_name[:300],
                "company_url": profile_url,
                "profile_url": profile_url,
                "official_website_url": None,
                "company_slug": company_slug,
                "solution_slug": solution_slug,
                "entity_type": entity_type,
                "listing_url": listing_url,
                "category_tags": category_tags,
                "listing_text_snippets": _extract_listing_snippets(anchor),
                "provenance": {
                    "source_name": "wealth_mosaic",
                    "listing_url": listing_url,
                    "anchor_href": href[:1000],
                    "anchor_text": normalized_company_name[:300],
                    "extractor": "anchor_scan",
                    "company_slug": company_slug,
                    "solution_slug": solution_slug,
                },
            }
        )

    return mentions


def _parse_generic_external_directory(
    html: str,
    listing_url: str,
    source_name: str,
    company_markers: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    tree = HTMLParser(html)
    if tree is None:
        return []
    markers = [m.lower() for m in (company_markers or [])]
    mentions: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    listing_host = (urlparse(listing_url).netloc or "").lower().replace("www.", "")

    for anchor in tree.css("a[href]"):
        href = anchor.attributes.get("href", "")
        company_url = _normalize_company_url(href, listing_url)
        if not company_url:
            continue
        parsed = urlparse(company_url)
        host = (parsed.netloc or "").lower().replace("www.", "")
        if not host or host == listing_host:
            continue
        if any(host.endswith(blocked) for blocked in EXTERNAL_HOST_BLOCKLIST):
            continue

        anchor_text = _clean_text(anchor.text())
        if markers and not any(marker in anchor_text.lower() for marker in markers):
            # Generic mode: allow long enough names when no marker match.
            if len(anchor_text.split()) < 2 and len(anchor_text) < 8:
                continue

        company_name = anchor_text or host.split(".")[0].replace("-", " ").title()
        key = (company_name.lower(), company_url.lower())
        if key in seen:
            continue
        seen.add(key)
        mentions.append(
            {
                "company_name": company_name[:300],
                "company_url": company_url,
                "profile_url": listing_url,
                "official_website_url": company_url,
                "company_slug": None,
                "solution_slug": None,
                "entity_type": "company",
                "listing_url": listing_url,
                "category_tags": [source_name.replace("_", " ")[:80]],
                "listing_text_snippets": _extract_listing_snippets(anchor),
                "provenance": {
                    "source_name": source_name,
                    "listing_url": listing_url,
                    "anchor_href": href[:1000],
                    "anchor_text": company_name[:300],
                    "extractor": "generic_external_directory",
                    "source_tier": "tier4_discovery",
                },
            }
        )
        if len(mentions) >= MAX_FILTERED_MENTIONS:
            break
    return mentions


def parse_partner_graph_listing(html: str, listing_url: str) -> List[Dict[str, Any]]:
    return _parse_generic_external_directory(
        html,
        listing_url,
        source_name="partner_graph_seed",
        company_markers=["partner", "integration", "technology"],
    )


def parse_conference_exhibitor_listing(html: str, listing_url: str) -> List[Dict[str, Any]]:
    return _parse_generic_external_directory(
        html,
        listing_url,
        source_name="conference_exhibitors_seed",
        company_markers=["exhibitor", "booth", "sponsor"],
    )


def _mention_quality_score(mention: Dict[str, Any], base_host: Optional[str]) -> int:
    score = 0
    company_name = _clean_text(str(mention.get("company_name") or ""))
    company_url = str(mention.get("company_url") or "")
    listing_snippets = mention.get("listing_text_snippets") or []
    host = (urlparse(company_url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    if base_host and host and host != base_host:
        score += 2
    if any(keyword in company_name.lower() for keyword in ("software", "platform", "technology", "systems")):
        score += 1
    if listing_snippets:
        score += 1
    if len(company_name.split()) >= 2:
        score += 1
    return score


SOURCE_REGISTRY: Dict[str, SourceConfig] = {
    "wealth_mosaic": SourceConfig(
        source_name="wealth_mosaic",
        seed_url="https://www.thewealthmosaic.com/needs/portfolio-wealth-management-systems/",
        parser=parse_wealth_mosaic_listing,
        default_max_pages=2,
    ),
    "partner_graph_seed": SourceConfig(
        source_name="partner_graph_seed",
        seed_url="https://www.avaloq.com/partners",
        parser=parse_partner_graph_listing,
        default_max_pages=1,
    ),
    "conference_exhibitors_seed": SourceConfig(
        source_name="conference_exhibitors_seed",
        seed_url="https://europe.money2020.com/exhibitors",
        parser=parse_conference_exhibitor_listing,
        default_max_pages=1,
    ),
}


def _discover_pagination_urls(html: str, listing_url: str, max_pages: int) -> List[str]:
    tree = HTMLParser(html)
    if tree is None:
        return [listing_url]

    urls = [listing_url]
    seen = {listing_url}
    for anchor in tree.css("a[href]"):
        href = anchor.attributes.get("href", "")
        if not href:
            continue
        absolute = urljoin(listing_url, href)
        if absolute in seen:
            continue
        if "page=" not in absolute and "/page/" not in absolute:
            continue
        if urlparse(absolute).netloc != urlparse(listing_url).netloc:
            continue
        seen.add(absolute)
        urls.append(absolute)
        if len(urls) >= max_pages:
            break
    return urls


def ingest_source(
    source_name: str,
    source_url: Optional[str] = None,
    max_pages: Optional[int] = None,
    timeout_seconds: int = 20,
) -> Dict[str, Any]:
    """Ingest comparator mentions for a configured source."""
    config = SOURCE_REGISTRY.get(source_name)
    if not config:
        raise ValueError(f"Unsupported comparator source: {source_name}")

    listing_url = source_url or config.seed_url
    page_limit = max_pages or config.default_max_pages
    page_limit = max(1, min(page_limit, 10))

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }

    mentions: List[Dict[str, Any]] = []
    pages_crawled = 0
    errors: List[str] = []

    with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
        try:
            seed_res = client.get(listing_url)
            seed_res.raise_for_status()
            page_urls = _discover_pagination_urls(seed_res.text, listing_url, page_limit)
        except Exception as exc:
            page_urls = [listing_url]
            errors.append(f"seed_fetch_failed:{exc}")

        for page_url in page_urls[:page_limit]:
            try:
                response = client.get(page_url)
                response.raise_for_status()
                pages_crawled += 1
                mentions.extend(config.parser(response.text, page_url))
            except Exception as exc:
                errors.append(f"page_fetch_failed:{page_url}:{exc}")

    # Deduplicate primarily by company URL (one record per vendor profile URL).
    deduped_by_url: Dict[str, Dict[str, Any]] = {}
    deduped_fallback: List[Dict[str, Any]] = []
    for mention in mentions:
        company_url = _clean_text(str(mention.get("company_url", ""))).lower()
        if company_url:
            existing = deduped_by_url.get(company_url)
            if not existing:
                deduped_by_url[company_url] = mention
                continue
            # Keep the richer mention when duplicates share the same URL.
            existing_snippet_len = len(" ".join(existing.get("listing_text_snippets", []) or []))
            candidate_snippet_len = len(" ".join(mention.get("listing_text_snippets", []) or []))
            existing_name_len = len(_clean_text(str(existing.get("company_name", ""))))
            candidate_name_len = len(_clean_text(str(mention.get("company_name", ""))))
            if (candidate_snippet_len, candidate_name_len) > (existing_snippet_len, existing_name_len):
                deduped_by_url[company_url] = mention
        else:
            deduped_fallback.append(mention)
    deduped = list(deduped_by_url.values()) + deduped_fallback

    # Secondary dedupe by normalized company name.
    seen_name: set[str] = set()
    deduped_name: List[Dict[str, Any]] = []
    for mention in deduped:
        name_key = _clean_text(str(mention.get("company_name", ""))).lower()
        url_key = _clean_text(str(mention.get("company_url", ""))).lower()
        key = name_key or url_key
        if not key:
            continue
        if key in seen_name:
            continue
        seen_name.add(key)
        deduped_name.append(mention)

    base_host = (urlparse(listing_url).netloc or "").lower()
    if base_host.startswith("www."):
        base_host = base_host[4:]
    scored = sorted(
        deduped_name,
        key=lambda mention: _mention_quality_score(mention, base_host),
        reverse=True,
    )
    filtered = [
        mention
        for mention in scored
        if _mention_quality_score(mention, base_host) >= 2
    ][:MAX_FILTERED_MENTIONS]

    return {
        "source_name": source_name,
        "source_url": listing_url,
        "pages_crawled": pages_crawled,
        "mentions": filtered,
        "errors": errors,
    }


def _is_aggregator_domain(domain: Optional[str]) -> bool:
    if not domain:
        return False
    return any(domain == agg or domain.endswith(f".{agg}") for agg in AGGREGATOR_DOMAINS)


def _pick_external_company_website(profile_url: str, html: str) -> Optional[str]:
    tree = HTMLParser(html)
    if tree is None:
        return None

    profile_domain = (urlparse(profile_url).netloc or "").lower()
    if profile_domain.startswith("www."):
        profile_domain = profile_domain[4:]

    candidates: list[tuple[int, str]] = []
    for anchor in tree.css("a[href]"):
        href = (anchor.attributes.get("href") or "").strip()
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "#")):
            continue
        absolute = _normalize_company_url(href, profile_url)
        if not absolute:
            continue
        parsed = urlparse(absolute)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if not host:
            continue
        if host == profile_domain:
            continue
        if _is_aggregator_domain(host):
            continue
        if any(blocked == host or host.endswith(f".{blocked}") for blocked in EXTERNAL_HOST_BLOCKLIST):
            continue

        anchor_text = _clean_text(anchor.text()).lower()
        score = 0
        if any(token in anchor_text for token in ("website", "visit", "company site", "official")):
            score += 4
        if any(token in parsed.path.lower() for token in ("contact", "about", "platform", "products", "solution")):
            score += 1
        if parsed.path in {"", "/"}:
            score += 1
        if len(anchor_text) <= 32:
            score += 1
        candidates.append((score, absolute))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def resolve_external_website_from_profile(
    profile_url: Optional[str],
    timeout_seconds: int = 4,
) -> Dict[str, Any]:
    """Resolve official company website from a directory profile page."""
    now = datetime.utcnow().isoformat()
    if not profile_url:
        return {
            "profile_url": profile_url,
            "official_website": None,
            "identity_confidence": "low",
            "captured_at": now,
            "error": "missing_profile_url",
        }

    normalized = profile_url.strip()
    if not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"

    domain = (urlparse(normalized).netloc or "").lower()
    if domain.startswith("www."):
        domain = domain[4:]

    if domain and not _is_aggregator_domain(domain):
        return {
            "profile_url": normalized,
            "official_website": normalized,
            "identity_confidence": "high",
            "captured_at": now,
            "error": None,
        }

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    try:
        with httpx.Client(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            response = client.get(normalized)
            response.raise_for_status()
            explicit = _extract_website_address_from_html(normalized, response.text)
            if explicit:
                return {
                    "profile_url": normalized,
                    "official_website": explicit,
                    "identity_confidence": "high",
                    "captured_at": now,
                    "error": None,
                }
            official = _pick_external_company_website(normalized, response.text)
            if official:
                return {
                    "profile_url": normalized,
                    "official_website": official,
                    "identity_confidence": "high",
                    "captured_at": now,
                    "error": None,
                }
            return {
                "profile_url": normalized,
                "official_website": None,
                "identity_confidence": "low",
                "captured_at": now,
                "error": "external_website_not_found",
            }
    except Exception as exc:
        return {
            "profile_url": normalized,
            "official_website": None,
            "identity_confidence": "low",
            "captured_at": now,
            "error": f"profile_fetch_failed:{exc}",
        }
