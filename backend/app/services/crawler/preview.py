"""Phase 2: Preview Fetch + Scoring - lightweight page preview and unified scoring."""

import asyncio
import re
from typing import List, Set, Optional, Callable, Tuple, Iterable
from urllib.parse import urlparse

import httpx
from selectolax.parser import HTMLParser

from .models import PagePreview
from .constants import (
    CAPABILITY_KEYWORDS,
    SERVICE_KEYWORDS,
    PROOF_SIGNALS,
    HARD_EXCLUDE,
    SOFT_DEMOTE,
    PRIORITY_PATHS,
    MAX_PAGES_TO_PREVIEW,
    BATCH_SIZE,
    REQUEST_DELAY,
)


class PreviewFetcher:
    """Fetches lightweight previews of pages for scoring."""
    
    def __init__(
        self,
        client: httpx.AsyncClient,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.client = client
        self.progress_callback = progress_callback
    
    def _log(self, message: str) -> None:
        """Log progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)

    def _safe_node_text(self, node, strip: bool = True) -> str:
        """Safely extract text from selectolax node."""
        if node is None:
            return ""
        try:
            text = node.text(strip=strip)
        except Exception:
            return ""
        if text is None:
            return ""
        return str(text).strip() if strip else str(text)
    
    async def fetch_previews(
        self,
        urls: Iterable[str],
        max_urls: Optional[int] = None,
    ) -> List[PagePreview]:
        """
        Fetch lightweight previews for a set of URLs.
        
        Extracts only: title, meta description, h1, first 10 h2/h3 headings.
        """
        seen: set[str] = set()
        urls_list: List[str] = []
        for raw in urls:
            normalized = str(raw or "").strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            urls_list.append(normalized)

        effective_max = MAX_PAGES_TO_PREVIEW if max_urls is None else max(1, int(max_urls))
        urls_list = urls_list[:effective_max]
        previews = []
        
        self._log(f"Fetching previews for {len(urls_list)} URLs...")
        
        # Process in batches
        for i in range(0, len(urls_list), BATCH_SIZE):
            batch = urls_list[i:i + BATCH_SIZE]
            
            tasks = [self._fetch_preview(url) for url in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, PagePreview):
                    previews.append(result)
            
            self._log(f"Previewed {min(i + BATCH_SIZE, len(urls_list))}/{len(urls_list)} pages")
            
            # Small delay between batches
            if i + BATCH_SIZE < len(urls_list):
                await asyncio.sleep(REQUEST_DELAY)
        
        return previews
    
    async def _fetch_preview(self, url: str) -> Optional[PagePreview]:
        """Fetch a lightweight preview of a single page."""
        try:
            # Use a shorter timeout for previews
            response = await self.client.get(url, timeout=10.0)
            if response.status_code != 200:
                return None
            
            html = response.text
            tree = HTMLParser(html)
            
            # Extract title
            title = ""
            title_node = tree.css_first('title')
            if title_node:
                title = self._safe_node_text(title_node, strip=True)
            
            # Extract meta description
            meta_desc = ""
            meta_node = tree.css_first('meta[name="description"]')
            if meta_node:
                meta_desc = meta_node.attributes.get('content', '')
            
            # Extract H1
            h1 = ""
            h1_node = tree.css_first('h1')
            if h1_node:
                h1 = self._safe_node_text(h1_node, strip=True)
            
            # Extract first 10 h2/h3 headings
            headings = []
            for heading in tree.css('h2, h3'):
                text = self._safe_node_text(heading, strip=True)
                if text and len(text) < 200:  # Skip overly long "headings"
                    headings.append(text)
                    if len(headings) >= 10:
                        break
            
            # Calculate path depth
            parsed = urlparse(url)
            path = parsed.path.strip('/')
            path_depth = len(path.split('/')) if path else 0
            
            return PagePreview(
                url=url,
                title=title,
                meta_description=meta_desc,
                h1=h1,
                headings=headings,
                path_depth=path_depth,
            )
        
        except Exception:
            return None


class PageScorer:
    """Unified scorer for page previews."""
    
    def score_previews(
        self, 
        previews: List[PagePreview]
    ) -> List[Tuple[PagePreview, float]]:
        """
        Score all previews and return sorted by score (highest first).
        
        Scoring criteria:
        - Keyword hits in url/title/h1/headings
        - Proof signals (customers, case studies)
        - Hard penalty for auth/legal/careers
        - Soft demote for blog/press, but promote back if proof-like
        - Gentle depth penalty only if no strong positives
        """
        scored = []
        
        for preview in previews:
            score = self._score_preview(preview)
            scored.append((preview, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored
    
    def _score_preview(self, preview: PagePreview) -> float:
        """Calculate score for a single preview."""
        score = 0.0
        combined_text = preview.combined_text.lower()
        url_lower = preview.url.lower()
        
        # 1. Capability keyword hits
        for keyword in CAPABILITY_KEYWORDS:
            if keyword in combined_text:
                score += 8
        
        # 2. Service keyword hits
        for keyword in SERVICE_KEYWORDS:
            if keyword in combined_text:
                score += 6
        
        # 3. Proof signals (customers, case studies)
        proof_count = 0
        for signal in PROOF_SIGNALS:
            if signal in combined_text:
                proof_count += 1
        if proof_count > 0:
            score += 15 + (proof_count * 3)
        
        # 4. Priority path bonus
        if any(path in url_lower for path in PRIORITY_PATHS):
            score += 10
        
        # 5. Hard penalty for auth/legal/careers
        if any(excl in url_lower for excl in HARD_EXCLUDE):
            score -= 100
        
        # 6. Soft demote for blog/press/news
        is_soft_demote = any(demote in url_lower for demote in SOFT_DEMOTE)
        if is_soft_demote:
            score -= 10
            # But promote back if it has proof signals (case study in press release)
            if proof_count > 0:
                score += 20  # Net +10 for proof in blog/press
        
        # 7. Gentle depth penalty only if score is low
        if score < 15:
            score -= preview.path_depth * 1.5
        
        # 8. Bonus for rich content indicators
        if len(preview.headings) >= 5:
            score += 5  # Well-structured page
        
        if preview.h1 and len(preview.h1) > 10:
            score += 3  # Has meaningful H1
        
        if preview.meta_description and len(preview.meta_description) > 50:
            score += 2  # Has meta description
        
        return score
    
    def has_proof_signals(self, preview: PagePreview) -> bool:
        """Check if preview has customer proof signals."""
        combined_text = preview.combined_text.lower()
        return any(signal in combined_text for signal in PROOF_SIGNALS)


def filter_and_score_urls(
    previews: List[PagePreview],
    min_score: float = -50
) -> List[PagePreview]:
    """
    Filter and score previews, returning sorted list.
    
    Args:
        previews: List of page previews
        min_score: Minimum score to include (filters out very bad pages)
    
    Returns:
        Sorted list of previews (best first)
    """
    scorer = PageScorer()
    scored = scorer.score_previews(previews)
    
    # Filter by minimum score
    filtered = [(preview, score) for preview, score in scored if score >= min_score]
    
    return [preview for preview, _ in filtered]
