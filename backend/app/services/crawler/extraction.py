"""Phase 4: Deep Fetch + Structured Extraction - full content with structure preservation."""

import asyncio
import re
from typing import List, Optional, Callable, Tuple
from urllib.parse import urlparse

import httpx
import trafilatura
from selectolax.parser import HTMLParser, Node

from .models import (
    PagePreview,
    CrawledPage,
    ContentBlock,
    CustomerEvidence,
    Signal,
    Evidence,
)
from .constants import (
    LOGO_SECTION_INDICATORS,
    PAGE_TYPE_PATTERNS,
    BATCH_SIZE,
    REQUEST_DELAY,
)


class ContentExtractor:
    """Extracts structured content from pages."""
    
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

    def _safe_node_text(self, node: Optional[Node], strip: bool = True) -> str:
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

    def _safe_attr_text(self, node: Node, key: str) -> str:
        """Safely extract attribute text and normalize it."""
        if node is None:
            return ""
        raw = node.attributes.get(key, "")
        if raw is None:
            return ""
        return str(raw).strip()
    
    async def extract_pages(
        self,
        previews: List[PagePreview]
    ) -> List[CrawledPage]:
        """
        Deep fetch and extract structured content from selected pages.
        
        No truncation at extraction time - full content preserved.
        """
        pages = []
        
        self._log(f"Extracting content from {len(previews)} pages...")
        
        # Process in batches
        for i in range(0, len(previews), BATCH_SIZE):
            batch = previews[i:i + BATCH_SIZE]
            
            tasks = [self._extract_page(preview) for preview in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, CrawledPage):
                    pages.append(result)
            
            self._log(f"Extracted {min(i + BATCH_SIZE, len(previews))}/{len(previews)} pages")
            
            # Small delay between batches
            if i + BATCH_SIZE < len(previews):
                await asyncio.sleep(REQUEST_DELAY)
        
        return pages
    
    async def _extract_page(self, preview: PagePreview) -> Optional[CrawledPage]:
        """Extract structured content from a single page."""
        try:
            response = await self.client.get(preview.url, timeout=15.0)
            if response.status_code != 200:
                return None
            
            html = response.text
            tree = HTMLParser(html)
            
            # Determine page type from URL
            page_type = self._classify_page_type(preview.url)
            
            # Extract structured content blocks
            blocks = self._extract_content_blocks(tree)
            
            # Extract customer evidence (logos, mentions)
            customer_evidence = self._extract_customer_evidence(tree, html, preview.url)
            
            # Extract raw text using trafilatura (for full content)
            raw_content = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                include_tables=True,
                no_fallback=False,
            ) or ""
            
            # Extract title
            title = preview.title
            if not title:
                title_node = tree.css_first('title')
                if title_node:
                    title = self._safe_node_text(title_node, strip=True)
            
            return CrawledPage(
                url=preview.url,
                title=title,
                page_type=page_type,
                blocks=blocks,
                signals=[],  # Signals extracted in Phase 5
                customer_evidence=customer_evidence,
                raw_content=raw_content,
                raw_html=html,
            )
        
        except Exception as e:
            self._log(f"Error extracting {preview.url}: {e}")
            return None
    
    def _classify_page_type(self, url: str) -> str:
        """Classify page type based on URL patterns."""
        url_lower = url.lower()
        
        for page_type, patterns in PAGE_TYPE_PATTERNS.items():
            if any(pattern in url_lower for pattern in patterns):
                return page_type
        
        return "other"
    
    def _extract_content_blocks(self, tree: HTMLParser) -> List[ContentBlock]:
        """Extract structured content blocks from HTML."""
        blocks = []
        
        # Remove script, style, nav, footer, header for main content extraction
        for tag in ['script', 'style', 'nav', 'footer', 'header', 'aside']:
            for node in tree.css(tag):
                node.decompose()
        
        # Find main content area
        main_content = tree.css_first('main, article, [role="main"], .content, #content, .main-content')
        if main_content is None:
            main_content = tree.body
        
        if main_content is None:
            return blocks
        
        # Extract headings, paragraphs, lists, tables
        self._extract_blocks_recursive(main_content, blocks)
        
        return blocks
    
    def _extract_blocks_recursive(
        self,
        node: Node,
        blocks: List[ContentBlock],
        depth: int = 0
    ) -> None:
        """Recursively extract content blocks from a node."""
        if depth > 10:  # Prevent infinite recursion
            return
        
        tag = node.tag if hasattr(node, 'tag') else None
        
        if tag is None:
            return
        
        # Headings
        if tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            text = self._safe_node_text(node, strip=True)
            if text and len(text) > 2:
                level = int(tag[1])
                blocks.append(ContentBlock(
                    type="heading",
                    content=text,
                    level=level,
                ))
        
        # Paragraphs
        elif tag == 'p':
            text = self._safe_node_text(node, strip=True)
            if text and len(text) > 20:  # Skip very short paragraphs
                blocks.append(ContentBlock(
                    type="paragraph",
                    content=text,
                    level=0,
                ))
        
        # Lists
        elif tag in ['ul', 'ol']:
            items = []
            for li in node.css('li'):
                item_text = self._safe_node_text(li, strip=True)
                if item_text:
                    items.append(f"- {item_text}")
            
            if items:
                blocks.append(ContentBlock(
                    type="list",
                    content="\n".join(items),
                    level=0,
                ))
            return  # Don't recurse into list items
        
        # Tables
        elif tag == 'table':
            table_text = self._extract_table_text(node)
            if table_text:
                blocks.append(ContentBlock(
                    type="table",
                    content=table_text,
                    level=0,
                ))
            return  # Don't recurse into table cells
        
        # Recurse into children
        for child in node.iter():
            if child != node:
                self._extract_blocks_recursive(child, blocks, depth + 1)
    
    def _extract_table_text(self, table_node: Node) -> str:
        """Extract text representation of a table."""
        rows = []
        
        for tr in table_node.css('tr'):
            cells = []
            for td in tr.css('td, th'):
                cell_text = self._safe_node_text(td, strip=True)
                cells.append(cell_text)
            
            if cells:
                rows.append(" | ".join(cells))
        
        return "\n".join(rows)
    
    def _extract_customer_evidence(
        self,
        tree: HTMLParser,
        html: str,
        source_url: str
    ) -> List[CustomerEvidence]:
        """Extract customer evidence from logo walls, testimonials, etc."""
        evidence = []
        html_lower = html.lower()
        
        # Find sections that likely contain customer logos
        logo_sections = self._find_logo_sections(tree, html_lower)
        
        for section in logo_sections:
            # Extract from img alt attributes
            for img in section.css('img[alt]'):
                alt = self._safe_attr_text(img, "alt")
                if alt and len(alt) > 2 and len(alt) < 100:
                    # Filter out generic alt texts
                    if not self._is_generic_alt(alt):
                        evidence.append(CustomerEvidence(
                            name=alt,
                            source_url=source_url,
                            evidence_type="logo_alt",
                            context=self._get_section_context(section),
                            selector=self._get_selector(img),
                        ))
            
            # Extract from SVG aria-labels
            for svg in section.css('svg[aria-label]'):
                label = self._safe_attr_text(svg, "aria-label")
                if label and len(label) > 2 and len(label) < 100:
                    evidence.append(CustomerEvidence(
                        name=label,
                        source_url=source_url,
                        evidence_type="aria_label",
                        context=self._get_section_context(section),
                        selector=self._get_selector(svg),
                    ))
            
            # Extract from elements with aria-label
            for elem in section.css('[aria-label]'):
                label = self._safe_attr_text(elem, "aria-label")
                if label and len(label) > 2 and len(label) < 100:
                    if not self._is_generic_alt(label):
                        evidence.append(CustomerEvidence(
                            name=label,
                            source_url=source_url,
                            evidence_type="aria_label",
                            context=self._get_section_context(section),
                            selector=self._get_selector(elem),
                        ))
        
        # Also look for explicit customer mentions in text
        text_evidence = self._extract_customer_text_mentions(tree, source_url)
        evidence.extend(text_evidence)
        
        # Dedupe by name
        seen_names = set()
        unique_evidence = []
        for e in evidence:
            name_lower = e.name.lower()
            if name_lower not in seen_names:
                seen_names.add(name_lower)
                unique_evidence.append(e)
        
        return unique_evidence
    
    def _find_logo_sections(self, tree: HTMLParser, html_lower: str) -> List[Node]:
        """Find sections that likely contain customer logos."""
        sections = []
        
        # Look for sections with logo-related class names or IDs
        selectors = [
            '[class*="logo"]',
            '[class*="customer"]',
            '[class*="client"]',
            '[class*="partner"]',
            '[class*="trusted"]',
            '[class*="testimonial"]',
            '[id*="logo"]',
            '[id*="customer"]',
            '[id*="client"]',
        ]
        
        for selector in selectors:
            try:
                for node in tree.css(selector):
                    sections.append(node)
            except Exception:
                continue
        
        # Also look for sections containing indicator text
        for indicator in LOGO_SECTION_INDICATORS:
            if indicator in html_lower:
                # Try to find the containing section
                for section in tree.css('section, div'):
                    section_text = self._safe_node_text(section, strip=True).lower()
                    if indicator in section_text[:500]:
                        sections.append(section)
                        break
        
        return sections
    
    def _is_generic_alt(self, alt: str) -> bool:
        """Check if alt text is generic/not a company name."""
        generic_patterns = [
            'logo', 'icon', 'image', 'photo', 'picture', 'img',
            'arrow', 'button', 'close', 'menu', 'search', 'loading',
            'placeholder', 'default', 'avatar', 'user', 'profile',
        ]
        
        alt_lower = alt.lower()
        
        # Too short
        if len(alt) < 3:
            return True
        
        # Just a generic word
        if alt_lower in generic_patterns:
            return True
        
        # Starts with generic patterns
        if any(alt_lower.startswith(p) for p in ['the ', 'a ', 'an ']):
            return True
        
        return False
    
    def _get_section_context(self, section: Node) -> str:
        """Get contextual text from a section (e.g., 'Trusted by')."""
        # Look for nearby heading or text that indicates context
        for heading in section.css('h1, h2, h3, h4, h5, h6, p'):
            text = self._safe_node_text(heading, strip=True)
            if text and any(ind in text.lower() for ind in LOGO_SECTION_INDICATORS):
                return text[:100]
        
        return ""
    
    def _get_selector(self, node: Node) -> str:
        """Generate a simple CSS selector for a node."""
        tag = node.tag if hasattr(node, 'tag') else 'unknown'
        
        # Try to get a unique identifier
        id_attr = node.attributes.get('id', '')
        if id_attr:
            return f"#{id_attr}"
        
        class_attr = node.attributes.get('class', '')
        if class_attr:
            classes = class_attr.split()[:2]  # First 2 classes
            return f"{tag}.{'.'.join(classes)}"
        
        return tag
    
    def _extract_customer_text_mentions(
        self,
        tree: HTMLParser,
        source_url: str
    ) -> List[CustomerEvidence]:
        """Extract customer mentions from case study or testimonial text."""
        evidence = []
        
        # Look for case study patterns
        case_study_patterns = [
            r'(?:case study|success story)[:\s]+([A-Z][A-Za-z\s&]+)',
            r'(?:customer|client)[:\s]+([A-Z][A-Za-z\s&]+)',
            r'"([^"]+)"[,\s]+(?:CEO|CTO|VP|Director|Manager|Head)',
        ]
        
        text = self._safe_node_text(tree.body, strip=False) if tree.body else ""
        
        for pattern in case_study_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name and len(name) > 3 and len(name) < 50:
                    evidence.append(CustomerEvidence(
                        name=name,
                        source_url=source_url,
                        evidence_type="text_mention",
                        context=match.group(0)[:100],
                        selector="",
                    ))
        
        return evidence


class SignalExtractor:
    """Extracts signals (capabilities, services, etc.) from content."""
    
    CAPABILITY_PATTERNS = [
        r'(?:we offer|we provide|our platform|features include)[:\s]+([^.]+)',
        r'(?:capabilities|features)[:\s]+([^.]+)',
    ]
    
    def extract_signals(self, page: CrawledPage) -> List[Signal]:
        """Extract signals from a crawled page."""
        signals = []
        
        # Extract from content blocks
        for block in page.blocks:
            if block.type == "list":
                # Lists often contain feature/capability items
                items = block.content.split('\n')
                for item in items:
                    item = item.lstrip('- ').strip()
                    if item and len(item) > 10 and len(item) < 200:
                        signals.append(Signal(
                            type="capability",
                            value=item,
                            evidence=Evidence(
                                source_url=page.url,
                                snippet=item,
                                selector_or_offset="",
                            ),
                        ))
        
        # Extract from customer evidence
        for ce in page.customer_evidence:
            signals.append(Signal(
                type="customer",
                value=ce.name,
                evidence=Evidence(
                    source_url=ce.source_url,
                    snippet=ce.context,
                    selector_or_offset=ce.selector,
                ),
            ))
        
        return signals
