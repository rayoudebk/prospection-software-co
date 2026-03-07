"""Data models for the unified crawler."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Evidence:
    """Evidence for a signal extraction."""
    source_url: str
    snippet: str
    selector_or_offset: str = ""


@dataclass
class Signal:
    """Extracted signal (capability, service, customer, integration)."""
    type: str  # capability, service, customer, integration
    value: str
    evidence: Evidence


@dataclass
class ContentBlock:
    """Structured content block from a page."""
    type: str  # heading, paragraph, list, table
    content: str
    level: int = 0  # for headings (1-6)


@dataclass
class CustomerEvidence:
    """Evidence of a customer from logo walls, testimonials, etc."""
    name: str
    source_url: str
    evidence_type: str  # logo_alt, aria_label, text_mention, case_study
    context: str = ""  # nearby text like "Trusted by"
    selector: str = ""


@dataclass
class PagePreview:
    """Lightweight preview of a page for scoring."""
    url: str
    title: str
    meta_description: str
    h1: str
    headings: List[str]  # h2/h3 headings
    path_depth: int
    
    @property
    def combined_text(self) -> str:
        """Combine all text fields for keyword matching."""
        parts: List[str] = []
        for value in [self.url, self.title, self.meta_description, self.h1]:
            text = str(value or "").strip()
            if text:
                parts.append(text)
        for heading in self.headings or []:
            text = str(heading or "").strip()
            if text:
                parts.append(text)
        return " ".join(parts)


@dataclass
class LLMTriageResult:
    """Result from LLM page triage."""
    url: str
    page_type: str
    contains_tags: List[str]
    priority: int  # 1-10


@dataclass
class CrawledPage:
    """Fully crawled page with structured content."""
    url: str
    title: str
    page_type: str
    blocks: List[ContentBlock] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)
    customer_evidence: List[CustomerEvidence] = field(default_factory=list)
    raw_content: str = ""  # full content, not truncated
    raw_html: str = ""  # original HTML for re-extraction if needed


@dataclass
class ContextPack:
    """Complete context pack for a company."""
    company_name: str
    website: str
    pages: List[CrawledPage] = field(default_factory=list)
    signals: List[Signal] = field(default_factory=list)  # aggregated from all pages
    customer_evidence: List[CustomerEvidence] = field(default_factory=list)  # aggregated
    raw_markdown: str = ""  # presentation layer
    product_pages_count: int = 0
    summary: str = ""
