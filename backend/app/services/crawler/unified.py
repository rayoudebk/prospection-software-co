"""Unified Crawler - orchestrates all 5 phases of crawling."""

import asyncio
from typing import Optional, Callable, List, Dict, Any
from urllib.parse import urlparse

import httpx

from .models import ContextPack, CrawledPage, PagePreview
from .discovery import URLDiscovery
from .preview import PreviewFetcher, PageScorer, filter_and_score_urls
from .triage import LLMTriage
from .extraction import ContentExtractor, SignalExtractor
from .output import OutputGenerator
from .constants import MAX_PAGES_TO_CRAWL


class UnifiedCrawler:
    """
    Unified crawler that orchestrates all 5 phases:
    
    1. Discovery - robots.txt, sitemaps, nav/footer, BFS
    2. Preview - lightweight fetch + scoring
    3. Triage - LLM classification + quota enforcement
    4. Extraction - structured content + customer evidence
    5. Output - context pack generation
    """
    
    def __init__(
        self,
        max_pages: int = MAX_PAGES_TO_CRAWL,
        timeout: int = 15,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the unified crawler.
        
        Args:
            max_pages: Maximum pages to crawl per domain
            timeout: Request timeout in seconds
            progress_callback: Optional callback for progress updates
        """
        self.max_pages = max_pages
        self.timeout = timeout
        self.progress_callback = progress_callback
    
    def _log(self, message: str) -> None:
        """Log progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    async def crawl_for_context(self, url: str) -> ContextPack:
        """
        Crawl a company website and build a context pack.
        
        Executes all 5 phases:
        1. Discover URLs from sitemaps, nav, BFS
        2. Fetch lightweight previews and score
        3. LLM triage with category quotas
        4. Deep extraction with structure preservation
        5. Generate context pack with markdown
        
        Args:
            url: Starting URL (can be domain or specific page)
        
        Returns:
            Complete ContextPack with pages, signals, and markdown
        """
        # Normalize URL
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        self._log(f"Starting crawl for {base_url}...")
        
        async with httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        ) as client:
            # Phase 1: Discovery
            self._log("Phase 1: Discovering URLs...")
            discovery = URLDiscovery(client, self.progress_callback)
            discovered_urls = await discovery.discover_urls(base_url)
            
            if not discovered_urls:
                # Fallback: at least try the starting URL
                discovered_urls = {url}
            
            self._log(f"Discovered {len(discovered_urls)} URLs")
            
            # Phase 2: Preview + Scoring
            self._log("Phase 2: Fetching previews and scoring...")
            preview_fetcher = PreviewFetcher(client, self.progress_callback)
            previews = await preview_fetcher.fetch_previews(discovered_urls)
            
            if not previews:
                # Fallback: create minimal preview from starting URL
                self._log("No previews fetched, using starting URL only")
                previews = [PagePreview(
                    url=url,
                    title="",
                    meta_description="",
                    h1="",
                    headings=[],
                    path_depth=0,
                )]
            
            # Score and sort previews
            scored_previews = filter_and_score_urls(previews)
            self._log(f"Scored {len(scored_previews)} pages")
            
            # Phase 3: LLM Triage
            self._log("Phase 3: LLM triage and quota selection...")
            triage = LLMTriage(self.progress_callback)
            selected_previews = triage.triage_previews(
                scored_previews,
                max_pages=self.max_pages
            )
            self._log(f"Selected {len(selected_previews)} pages for deep extraction")
            
            # Phase 4: Deep Extraction
            self._log("Phase 4: Deep content extraction...")
            extractor = ContentExtractor(client, self.progress_callback)
            pages = await extractor.extract_pages(selected_previews)
            
            # Extract signals from pages
            signal_extractor = SignalExtractor()
            for page in pages:
                page.signals = signal_extractor.extract_signals(page)
            
            self._log(f"Extracted content from {len(pages)} pages")
            
            # Phase 5: Output Generation
            self._log("Phase 5: Generating context pack...")
            output_gen = OutputGenerator()
            context_pack = output_gen.generate_context_pack(pages, base_url)
            
            self._log(f"Context pack complete: {len(pages)} pages, {len(context_pack.signals)} signals")
            
            return context_pack


async def crawl_multiple_urls(
    urls: List[str],
    max_pages_per_url: int = 20,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, ContextPack]:
    """
    Crawl multiple company URLs and return context packs.
    
    Args:
        urls: List of URLs to crawl
        max_pages_per_url: Max pages per domain
        progress_callback: Optional progress callback
    
    Returns:
        Dict mapping URL to ContextPack
    """
    crawler = UnifiedCrawler(
        max_pages=max_pages_per_url,
        progress_callback=progress_callback
    )
    results = {}
    
    for url in urls:
        try:
            context_pack = await crawler.crawl_for_context(url)
            results[url] = context_pack
        except Exception as e:
            if progress_callback:
                progress_callback(f"Error crawling {url}: {e}")
            continue
    
    return results
