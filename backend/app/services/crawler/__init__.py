"""Unified crawler package for context pack generation."""

from .models import (
    Evidence,
    Signal,
    ContentBlock,
    CustomerEvidence,
    PagePreview,
    CrawledPage,
    ContextPack,
    LLMTriageResult,
)
from .unified import UnifiedCrawler, crawl_multiple_urls
from .discovery import URLDiscovery
from .preview import PreviewFetcher, PageScorer, filter_and_score_urls
from .triage import LLMTriage
from .extraction import ContentExtractor, SignalExtractor
from .output import OutputGenerator

__all__ = [
    # Main entry points
    "UnifiedCrawler",
    "crawl_multiple_urls",
    
    # Phase components
    "URLDiscovery",
    "PreviewFetcher",
    "PageScorer",
    "LLMTriage",
    "ContentExtractor",
    "SignalExtractor",
    "OutputGenerator",
    
    # Data models
    "Evidence",
    "Signal",
    "ContentBlock",
    "CustomerEvidence",
    "PagePreview",
    "CrawledPage",
    "ContextPack",
    "LLMTriageResult",
    
    # Utility functions
    "filter_and_score_urls",
]
