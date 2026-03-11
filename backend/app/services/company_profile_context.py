"""Compatibility helpers for company-profile brief and summary fields."""
from __future__ import annotations

from typing import Any


def _clean_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def get_manual_brief_text(profile: Any) -> str | None:
    return _clean_text(getattr(profile, "manual_brief_text", None))


def get_generated_context_summary(profile: Any) -> str | None:
    return _clean_text(getattr(profile, "generated_context_summary", None))
