"""Phase 4: Deep Fetch + Structured Extraction - full content with structure preservation."""

import asyncio
import re
from typing import List, Optional, Callable, Tuple
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from selectolax.parser import HTMLParser, Node

from .connectors.chrome_devtools_mcp import render_page_via_chrome_devtools_mcp
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
        self._bundle_text_cache: dict[str, str] = {}
    
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

    def _decode_js_string(self, value: str) -> str:
        text = str(value or "")
        if not text:
            return ""
        if "\\u" in text or "\\x" in text:
            try:
                text = bytes(text, "utf-8").decode("unicode_escape")
            except Exception:
                pass
        return " ".join(text.split()).strip()

    def _extract_meta_description(self, tree: HTMLParser) -> str:
        for selector in (
            'meta[name="description"]',
            'meta[property="og:description"]',
            'meta[name="twitter:description"]',
        ):
            node = tree.css_first(selector)
            if not node:
                continue
            content = self._safe_attr_text(node, "content")
            if content:
                return content
        return ""

    def _looks_like_customer_entity(self, name: str, evidence_type: str) -> bool:
        text = str(name or "").strip()
        if not text or not any(ch.isalpha() for ch in text):
            return False
        if re.search(r"[.!?]", text):
            return False

        lowered = text.lower()
        words = [word for word in re.split(r"\s+", text) if word]
        noisy_terms = (
            "award",
            "provider",
            "software",
            "solution",
            "solutions",
            "consultancy",
            "consulting",
            "architecture",
            "onboarding",
            "cross device",
            "omnichannel",
            "intelligent",
            "checklist",
            "logo",
            "colour",
            "color",
            "png",
            "jpg",
            "jpeg",
            "svg",
        )
        if any(token in lowered for token in noisy_terms):
            return False
        if any(lowered.startswith(prefix) for prefix in ("how ", "why ", "our ", "with ", "secure ")):
            return False

        if evidence_type in {"logo_alt", "aria_label"}:
            if len(words) > 6:
                return False
            if not any(ch.isupper() or ch.isdigit() for ch in text):
                return False
        elif len(words) > 10:
            return False
        return True

    def _needs_render_fallback(
        self,
        *,
        blocks: List[ContentBlock],
        customer_evidence: List[CustomerEvidence],
        raw_content: str,
        title: str,
    ) -> bool:
        normalized_text = " ".join(str(raw_content or "").split())
        if customer_evidence or len(blocks) >= 3:
            return False
        if len(normalized_text) >= 240:
            return False
        if len((title or "").strip()) > 40 and len(normalized_text) >= 80:
            return False
        return True

    def _should_attempt_render_enrichment(
        self,
        *,
        preview: PagePreview,
        page_type: str,
        raw_content: str,
    ) -> bool:
        url_lower = str(preview.url or "").lower()
        if page_type not in {"product", "solutions", "docs", "services"}:
            return False
        if not any(token in url_lower for token in ("/platform/", "/solutions/", "/technology", "/documentation", "/docs/", "/api/")):
            return False
        normalized = " ".join(str(raw_content or "").split()).lower()
        if len(normalized) < 240:
            return True
        interactive_markers = (
            "fonctionnalites detaillees",
            "functionalities",
            "detailed features",
            "accordion",
            "expand",
        )
        return any(marker in normalized for marker in interactive_markers)

    def _bundle_candidate_urls(self, tree: HTMLParser, source_url: str) -> List[str]:
        parsed = urlparse(source_url)
        domain = parsed.netloc
        candidates: list[tuple[int, str]] = []
        seen: set[str] = set()
        for script in tree.css("script[src]"):
            src = self._safe_attr_text(script, "src")
            if not src:
                continue
            absolute = urljoin(source_url, src)
            bundle = urlparse(absolute)
            if bundle.netloc != domain:
                continue
            path = (bundle.path or "").lower()
            if not path.endswith(".js"):
                continue
            key = absolute.lower()
            if key in seen:
                continue
            seen.add(key)
            score = 0
            if "/assets/" in path:
                score += 12
            if any(token in path for token in ("/index-", "/main-", "/app-", "/index.", "/main.", "/app.")):
                score += 20
            if "vendor" in path or "polyfill" in path:
                score -= 8
            score -= min(len(path), 120) // 24
            candidates.append((score, absolute))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [url for _, url in candidates[:3]]

    async def _fetch_bundle_text(self, bundle_url: str) -> str:
        cached = self._bundle_text_cache.get(bundle_url)
        if cached is not None:
            return cached
        try:
            response = await self.client.get(bundle_url, timeout=20.0)
            if response.status_code != 200:
                self._bundle_text_cache[bundle_url] = ""
                return ""
            text = str(response.text or "")
            self._bundle_text_cache[bundle_url] = text
            return text
        except Exception:
            self._bundle_text_cache[bundle_url] = ""
            return ""

    def _looks_like_route_label(self, label: str) -> bool:
        text = str(label or "").strip()
        if not text:
            return False
        if len(text) < 3 or len(text) > 80:
            return False
        if any(ch in text for ch in "{}[]<>"):
            return False
        if text.lower() in {"solutions", "plateforme", "technologie & services", "à propos"}:
            return False
        return True

    async def _extract_spa_bundle_artifacts(
        self,
        tree: HTMLParser,
        source_url: str,
    ) -> tuple[list[CustomerEvidence], list[Signal], list[str]]:
        bundle_urls = self._bundle_candidate_urls(tree, source_url)
        if not bundle_urls:
            return [], [], []

        customers: list[CustomerEvidence] = []
        signals: list[Signal] = []
        text_fragments: list[str] = []
        seen_customer_names: set[str] = set()
        seen_signal_keys: set[tuple[str, str]] = set()
        seen_fragments: set[str] = set()

        def add_fragment(value: str) -> None:
            normalized = " ".join(str(value or "").split()).strip()
            if not normalized or normalized.lower() in seen_fragments:
                return
            seen_fragments.add(normalized.lower())
            text_fragments.append(normalized)

        for bundle_url in bundle_urls:
            bundle_text = await self._fetch_bundle_text(bundle_url)
            if not bundle_text:
                continue

            for match in re.finditer(r'\{name:"([^"]{2,120})",(?:logo|src):"([^"]{1,240})"', bundle_text):
                raw_name, raw_asset = match.groups()
                name = self._decode_js_string(raw_name)
                asset = self._decode_js_string(raw_asset)
                asset_lower = asset.lower()
                if not self._looks_like_customer_entity(name, "bundle_logo_manifest"):
                    continue
                if "/customer_logo/" in asset_lower:
                    key = name.lower()
                    if key in seen_customer_names:
                        continue
                    seen_customer_names.add(key)
                    customers.append(
                        CustomerEvidence(
                            name=name,
                            source_url=source_url,
                            evidence_type="bundle_logo_manifest",
                            context="Named account from first-party SPA customer logo manifest",
                            selector=bundle_url,
                        )
                    )
                    add_fragment(name)
                elif any(token in asset_lower for token in ("/partenaire/", "/partner", "/integration", "/ecosystem")):
                    key = ("integration", name.lower())
                    if key in seen_signal_keys:
                        continue
                    seen_signal_keys.add(key)
                    signals.append(
                        Signal(
                            type="integration",
                            value=name,
                            evidence=Evidence(
                                source_url=source_url,
                                snippet="Named partner from first-party SPA partner logo manifest",
                                selector_or_offset=bundle_url,
                            ),
                        )
                    )
                    add_fragment(name)

            for route_match in re.finditer(r'to:"(/[^"]+)"', bundle_text):
                route_path = route_match.group(1)
                if not route_path.startswith(("/solutions/", "/platform/", "/technology", "/technology-and-services/", "/infrastructure")):
                    continue
                window = bundle_text[route_match.end() : route_match.end() + 450]
                label_match = re.search(r'children:"([^"]{2,120})"', window)
                if not label_match:
                    continue
                label = self._decode_js_string(label_match.group(1))
                if not self._looks_like_route_label(label):
                    continue
                signal_type = "workflow"
                if route_path.startswith("/solutions/"):
                    signal_type = "customer_archetype"
                elif route_path.startswith(("/technology", "/technology-and-services/")):
                    signal_type = "service"
                key = (signal_type, label.lower())
                if key in seen_signal_keys:
                    continue
                seen_signal_keys.add(key)
                signals.append(
                    Signal(
                        type=signal_type,
                        value=label,
                        evidence=Evidence(
                            source_url=source_url,
                            snippet=f"SPA route label for {route_path}",
                            selector_or_offset=bundle_url,
                        ),
                    )
                )
                add_fragment(label)

        return customers, signals, text_fragments

    def _rendered_blocks(self, preview: PagePreview, rendered_text: str) -> List[ContentBlock]:
        blocks: List[ContentBlock] = []
        heading = str(preview.h1 or preview.title or "").strip()
        if heading:
            blocks.append(ContentBlock(type="heading", content=heading[:180], level=1))
        for paragraph in re.split(r"\n{2,}", rendered_text):
            normalized = " ".join(paragraph.split())
            if len(normalized) < 40:
                continue
            blocks.append(ContentBlock(type="paragraph", content=normalized[:1200], level=0))
            if len(blocks) >= 5:
                break
        return blocks

    def _looks_like_interactive_capability_label(self, value: str) -> bool:
        text = " ".join(str(value or "").split()).strip()
        if len(text) < 16 or len(text) > 180:
            return False
        lowered = text.lower()
        if lowered in {
            "solutions",
            "plateforme",
            "technologie & services",
            "à propos",
            "demander une présentation",
        }:
            return False
        if any(token in lowered for token in ("linkedin", "mentions légales", "contact@")):
            return False
        keywords = (
            "pms",
            "oms",
            "trading",
            "trade",
            "portfolio",
            "portefeuille",
            "ordre",
            "reporting",
            "performance",
            "compliance",
            "contrôle",
            "monitoring",
            "routage",
            "connectivité",
            "modélisation",
            "allocation",
            "bourse",
            "gestion",
            "post-trade",
            "pré-trade",
        )
        return any(token in lowered for token in keywords)

    def _rendered_html_to_text(self, rendered_html: str) -> str:
        extracted = trafilatura.extract(
            rendered_html,
            include_links=False,
            include_images=False,
            include_tables=True,
            no_fallback=False,
        ) or ""
        normalized = " ".join(str(extracted or "").split())
        if normalized:
            return normalized
        tree = HTMLParser(rendered_html)
        if tree is None:
            return ""
        body = tree.body or tree.css_first("main")
        text = body.text(separator=" ", strip=True) if body else tree.text(separator=" ", strip=True)
        return " ".join(str(text or "").split())

    def _extract_rendered_dom_blocks(
        self,
        *,
        preview: PagePreview,
        page_type: str,
        rendered_html: str,
    ) -> List[ContentBlock]:
        tree = HTMLParser(str(rendered_html or ""))
        if tree is None:
            return []
        blocks = self._extract_content_blocks(tree)

        url_lower = str(preview.url or "").lower()
        if page_type in {"product", "solutions"} and any(token in url_lower for token in ("/platform/", "/solutions/")):
            main = tree.css_first("main") or tree.body or tree
            interactive_items: list[str] = []
            seen_items: set[str] = set()
            for button in main.css("button"):
                text = " ".join(self._safe_node_text(button, strip=True).split())
                if not self._looks_like_interactive_capability_label(text):
                    continue
                key = text.lower()
                if key in seen_items:
                    continue
                seen_items.add(key)
                interactive_items.append(f"- {text}")
            if interactive_items:
                blocks.append(
                    ContentBlock(
                        type="list",
                        content="\n".join(interactive_items[:24]),
                        level=0,
                    )
                )
        return blocks
    
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
        total_batches = max(1, (len(previews) + BATCH_SIZE - 1) // BATCH_SIZE)
        for i in range(0, len(previews), BATCH_SIZE):
            batch = previews[i:i + BATCH_SIZE]
            batch_index = (i // BATCH_SIZE) + 1
            if batch:
                self._log(f"Extraction batch {batch_index}/{total_batches}: {batch[0].url}")
            
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

            bundle_customers, bundle_signals, bundle_fragments = await self._extract_spa_bundle_artifacts(tree, preview.url)
            
            # Extract structured content blocks
            blocks = self._extract_content_blocks(tree)
            
            # Extract customer evidence (logos, mentions)
            customer_evidence = self._extract_customer_evidence(tree, html, preview.url)
            if bundle_customers:
                customer_evidence.extend(bundle_customers)
            
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
            meta_description = self._extract_meta_description(tree)
            content_fragments = [fragment for fragment in [meta_description, *bundle_fragments] if fragment]
            if content_fragments:
                existing_content = " ".join(str(raw_content or "").split())
                additions = [
                    fragment
                    for fragment in content_fragments
                    if fragment.lower() not in existing_content.lower()
                ]
                if additions:
                    raw_content = "\n".join([*additions, raw_content]).strip()
                    if not blocks:
                        blocks = self._rendered_blocks(preview, "\n\n".join(additions))

            interactive_enrichment = self._should_attempt_render_enrichment(
                preview=preview,
                page_type=page_type,
                raw_content=raw_content,
            )
            should_render = self._needs_render_fallback(
                blocks=blocks,
                customer_evidence=customer_evidence,
                raw_content=raw_content,
                title=title,
            ) or interactive_enrichment

            if should_render:
                rendered = await asyncio.to_thread(
                    render_page_via_chrome_devtools_mcp,
                    preview.url,
                    timeout_seconds=20,
                    prefer_playwright=interactive_enrichment,
                )
                rendered_html = str(rendered.get("html") or "")
                rendered_text = " ".join(str(rendered.get("content") or "").split())
                rendered_blocks = (
                    self._extract_rendered_dom_blocks(
                        preview=preview,
                        page_type=page_type,
                        rendered_html=rendered_html,
                    )
                    if rendered_html
                    else []
                )
                if rendered_html and not rendered_text:
                    rendered_text = self._rendered_html_to_text(rendered_html)
                render_error = str(rendered.get("error") or "").strip()
                if render_error:
                    self._log(
                        f"Rendered extraction issue for {preview.url}: "
                        f"provider={rendered.get('provider')} error={render_error}"
                    )
                min_gain = 40 if interactive_enrichment else 120
                rendered_block_gain = len(rendered_blocks) > len(blocks) + 2
                if len(rendered_text) > len(" ".join(raw_content.split())) + min_gain or rendered_block_gain:
                    raw_content = rendered_text
                    blocks = rendered_blocks or self._rendered_blocks(preview, rendered_text)
            
            return CrawledPage(
                url=preview.url,
                title=title,
                page_type=page_type,
                blocks=blocks,
                signals=bundle_signals,
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
                        if not self._looks_like_customer_entity(alt, "logo_alt"):
                            continue
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
                    if not self._looks_like_customer_entity(label, "aria_label"):
                        continue
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
                        if not self._looks_like_customer_entity(label, "aria_label"):
                            continue
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
                    if not self._looks_like_customer_entity(name, "text_mention"):
                        continue
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
