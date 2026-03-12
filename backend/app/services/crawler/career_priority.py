"""Shared heuristics for prioritizing relevant careers and job pages."""

from __future__ import annotations

import re

CAREER_PAGE_URL_TOKENS = (
    "/careers",
    "/career",
    "/jobs",
    "/job-",
    "/job/",
    "/apply",
    "/join-us",
    "/open-roles",
    "/open-positions",
)

# Keywords are stored in normalized space-separated form. Matching uses
# whitespace-bounded lookup after stripping punctuation from the source text.
CAREER_TARGET_KEYWORDS = (
    "product",
    "product manager",
    "product design",
    "product marketing",
    "implementation",
    "implementation consultant",
    "implementation manager",
    "customer implementation",
    "onboarding",
    "customer onboarding",
    "client services",
    "client servicing",
    "customer success",
    "professional services",
    "data",
    "data engineer",
    "data platform",
    "data operations",
    "operations",
    "workflow",
    "workflow operations",
    "sales",
    "sales engineer",
    "account executive",
    "solutions",
    "solutions engineer",
    "solutions consultant",
    "pre sales",
    "revops",
    "revenue operations",
    "sales operations",
    "go to market operations",
    "engineering",
    "engineer",
    "developer",
    "software engineer",
    "platform engineer",
    "devops",
    "site reliability",
    "sre",
)

CAREER_EXCLUDED_KEYWORDS = (
    "finance",
    "financial",
    "controller",
    "accounting",
    "accountant",
    "tax",
    "treasury",
    "fp a",
    "financial planning",
    "legal",
    "counsel",
    "paralegal",
    "recruiter",
    "recruiting",
    "talent acquisition",
    "people ops",
    "human resources",
    "hr",
    "payroll",
    "facilities",
    "workplace",
    "office manager",
)

_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")


def normalize_career_text(value: str | None) -> str:
    """Normalize text for coarse keyword matching across URLs and titles."""
    lowered = str(value or "").strip().lower()
    if not lowered:
        return ""
    return " ".join(_NORMALIZE_RE.sub(" ", lowered).split())


def is_career_page_url(url: str | None) -> bool:
    lowered = str(url or "").strip().lower()
    return any(token in lowered for token in CAREER_PAGE_URL_TOKENS)


def _keyword_hits(value: str | None, keywords: tuple[str, ...]) -> set[str]:
    normalized = normalize_career_text(value)
    if not normalized:
        return set()
    haystack = f" {normalized} "
    return {keyword for keyword in keywords if f" {keyword} " in haystack}


def career_target_keyword_hits(value: str | None) -> set[str]:
    return _keyword_hits(value, CAREER_TARGET_KEYWORDS)


def career_excluded_keyword_hits(value: str | None) -> set[str]:
    return _keyword_hits(value, CAREER_EXCLUDED_KEYWORDS)
