"""Phase 1: URL Discovery - robots.txt, sitemaps, nav/footer extraction, BFS."""

import asyncio
import gzip
import re
from typing import Set, List, Optional, Callable
from urllib.parse import urlparse, urljoin, parse_qs, urlencode
import xml.etree.ElementTree as ET

import httpx
from selectolax.parser import HTMLParser

from .constants import (
    SITEMAP_PATHS,
    STRIP_PARAMS,
    PRIORITY_PATHS,
    HUB_PATTERNS,
    HARD_EXCLUDE,
    MAX_URLS_FROM_SITEMAP,
    MAX_URLS_FROM_HOMEPAGE,
    MAX_BFS_DEPTH,
)


class URLDiscovery:
    """Discovers URLs from a domain using multiple strategies."""
    
    def __init__(
        self,
        client: httpx.AsyncClient,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.client = client
        self.progress_callback = progress_callback
        self._visited_sitemaps: Set[str] = set()
    
    def _log(self, message: str) -> None:
        """Log progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    async def discover_urls(self, base_url: str) -> Set[str]:
        """
        Discover all candidate URLs from a domain.
        
        Combines:
        1. robots.txt Sitemap: lines
        2. Recursive sitemap parsing
        3. Nav/footer link extraction
        4. BFS from hub pages
        """
        urls = set()
        
        # Always include the base URL
        urls.add(base_url)
        
        # 1. Parse robots.txt for sitemap URLs
        self._log("Checking robots.txt for sitemaps...")
        robots_sitemaps = await self._parse_robots_sitemaps(base_url)
        
        # 2. Parse sitemaps (from robots.txt + common paths)
        self._log("Parsing sitemaps...")
        sitemap_urls = await self._parse_all_sitemaps(base_url, robots_sitemaps)
        urls.update(sitemap_urls)
        self._log(f"Found {len(sitemap_urls)} URLs from sitemaps")
        
        # 3. Extract nav/footer links from homepage
        self._log("Extracting navigation links...")
        nav_urls = await self._extract_nav_footer_links(base_url)
        urls.update(nav_urls)
        self._log(f"Found {len(nav_urls)} URLs from navigation")
        
        # 4. BFS from hub pages (if we don't have enough URLs)
        if len(urls) < 50:
            self._log("Expanding from hub pages...")
            hub_urls = await self._bfs_from_hubs(base_url, urls)
            urls.update(hub_urls)
            self._log(f"Found {len(hub_urls)} additional URLs from BFS")
        
        # Normalize and dedupe
        normalized = self._normalize_and_dedupe(urls, base_url)
        self._log(f"Total unique URLs discovered: {len(normalized)}")
        
        return normalized
    
    async def _parse_robots_sitemaps(self, base_url: str) -> List[str]:
        """Parse robots.txt for Sitemap: directives."""
        sitemaps = []
        
        try:
            response = await self.client.get(f"{base_url}/robots.txt")
            if response.status_code == 200:
                for line in response.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_url = line.split(":", 1)[1].strip()
                        if sitemap_url:
                            sitemaps.append(sitemap_url)
        except Exception:
            pass
        
        return sitemaps
    
    async def _parse_all_sitemaps(
        self, 
        base_url: str, 
        robots_sitemaps: List[str]
    ) -> Set[str]:
        """Parse all sitemaps: from robots.txt and common paths."""
        urls = set()
        
        # Combine robots.txt sitemaps with common paths
        sitemap_urls_to_try = set(robots_sitemaps)
        for path in SITEMAP_PATHS:
            sitemap_urls_to_try.add(f"{base_url}{path}")
        
        # Parse each sitemap
        for sitemap_url in sitemap_urls_to_try:
            if sitemap_url in self._visited_sitemaps:
                continue
            
            sitemap_urls = await self._parse_sitemap_recursive(sitemap_url)
            urls.update(sitemap_urls)
            
            if len(urls) >= MAX_URLS_FROM_SITEMAP:
                break
        
        return urls
    
    async def _parse_sitemap_recursive(self, sitemap_url: str) -> Set[str]:
        """
        Parse a sitemap, handling both urlset and sitemapindex.
        Recursively processes sitemap indexes.
        """
        if sitemap_url in self._visited_sitemaps:
            return set()
        
        self._visited_sitemaps.add(sitemap_url)
        urls = set()
        
        try:
            response = await self.client.get(sitemap_url)
            if response.status_code != 200:
                return urls
            
            # Handle gzip compressed sitemaps
            content = response.content
            if sitemap_url.endswith('.gz') or response.headers.get('content-encoding') == 'gzip':
                try:
                    content = gzip.decompress(content)
                except Exception:
                    pass
            
            # Parse as text
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
            
            # Try to parse XML
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                # Fallback to regex extraction
                return self._extract_urls_regex(content)
            
            # Determine if this is a sitemap index or urlset
            root_tag = root.tag.lower()
            
            if 'sitemapindex' in root_tag:
                # This is a sitemap index - recurse into child sitemaps
                child_sitemaps = []
                for elem in root.iter():
                    if elem.tag.lower().endswith('loc'):
                        child_url = elem.text.strip() if elem.text else ""
                        if child_url and child_url not in self._visited_sitemaps:
                            child_sitemaps.append(child_url)
                
                # Recursively parse child sitemaps (limit to avoid explosion)
                for child_url in child_sitemaps[:20]:
                    child_urls = await self._parse_sitemap_recursive(child_url)
                    urls.update(child_urls)
                    if len(urls) >= MAX_URLS_FROM_SITEMAP:
                        break
            else:
                # This is a regular urlset - extract page URLs
                for elem in root.iter():
                    if elem.tag.lower().endswith('loc'):
                        page_url = elem.text.strip() if elem.text else ""
                        if page_url and not page_url.endswith('.xml') and not page_url.endswith('.xml.gz'):
                            urls.add(page_url)
                            if len(urls) >= MAX_URLS_FROM_SITEMAP:
                                break
        
        except Exception:
            pass
        
        return urls
    
    def _extract_urls_regex(self, content: str) -> Set[str]:
        """Fallback regex extraction of URLs from sitemap content."""
        urls = set()
        url_pattern = re.compile(r'<loc>\s*(https?://[^<\s]+)\s*</loc>', re.IGNORECASE)
        
        for match in url_pattern.finditer(content):
            url = match.group(1).strip()
            if url and not url.endswith('.xml') and not url.endswith('.xml.gz'):
                urls.add(url)
                if len(urls) >= MAX_URLS_FROM_SITEMAP:
                    break
        
        return urls
    
    async def _extract_nav_footer_links(self, base_url: str) -> Set[str]:
        """Extract links from nav and footer sections using DOM parsing."""
        urls = set()
        parsed_base = urlparse(base_url)
        
        try:
            response = await self.client.get(base_url)
            if response.status_code != 200:
                return urls
            
            tree = HTMLParser(response.text)
            
            # Extract links from nav elements
            for nav in tree.css('nav'):
                for link in nav.css('a[href]'):
                    href = link.attributes.get('href', '')
                    full_url = self._resolve_url(href, parsed_base)
                    if full_url:
                        urls.add(full_url)
            
            # Extract links from footer
            for footer in tree.css('footer'):
                for link in footer.css('a[href]'):
                    href = link.attributes.get('href', '')
                    full_url = self._resolve_url(href, parsed_base)
                    if full_url:
                        urls.add(full_url)
            
            # Also extract links from header (often contains main nav)
            for header in tree.css('header'):
                for link in header.css('a[href]'):
                    href = link.attributes.get('href', '')
                    full_url = self._resolve_url(href, parsed_base)
                    if full_url:
                        urls.add(full_url)
            
            # Extract from common menu classes
            menu_selectors = [
                '.menu a[href]', '.nav a[href]', '.navigation a[href]',
                '[class*="menu"] a[href]', '[class*="nav"] a[href]',
                '[role="navigation"] a[href]', '[role="menu"] a[href]',
            ]
            
            for selector in menu_selectors:
                try:
                    for link in tree.css(selector):
                        href = link.attributes.get('href', '')
                        full_url = self._resolve_url(href, parsed_base)
                        if full_url:
                            urls.add(full_url)
                except Exception:
                    continue
        
        except Exception:
            pass
        
        return urls
    
    async def _bfs_from_hubs(self, base_url: str, known_urls: Set[str]) -> Set[str]:
        """BFS expansion from high-value hub pages."""
        new_urls = set()
        parsed_base = urlparse(base_url)
        
        # Find hub pages from known URLs
        hub_urls = []
        for url in known_urls:
            url_lower = url.lower()
            if any(pattern in url_lower for pattern in HUB_PATTERNS):
                hub_urls.append(url)
        
        # Limit hubs to process
        hub_urls = hub_urls[:10]
        
        # BFS from each hub
        for depth in range(MAX_BFS_DEPTH):
            if not hub_urls:
                break
            
            next_level = []
            
            for hub_url in hub_urls[:5]:  # Process 5 hubs per level
                try:
                    response = await self.client.get(hub_url)
                    if response.status_code != 200:
                        continue
                    
                    tree = HTMLParser(response.text)
                    
                    # Extract all links from the page
                    for link in tree.css('a[href]'):
                        href = link.attributes.get('href', '')
                        full_url = self._resolve_url(href, parsed_base)
                        
                        if full_url and full_url not in known_urls and full_url not in new_urls:
                            # Check if it's a priority path
                            if self._is_priority_url(full_url):
                                new_urls.add(full_url)
                                next_level.append(full_url)
                    
                    if len(new_urls) >= MAX_URLS_FROM_HOMEPAGE:
                        break
                
                except Exception:
                    continue
                
                # Small delay between requests
                await asyncio.sleep(0.2)
            
            hub_urls = next_level
        
        return new_urls
    
    def _resolve_url(self, href: str, parsed_base) -> Optional[str]:
        """Resolve a href to a full URL, filtering out non-page links."""
        if not href:
            return None
        
        href = href.strip()
        
        # Skip non-page links
        if href.startswith(('#', 'javascript:', 'mailto:', 'tel:', 'data:')):
            return None
        
        # Convert relative to absolute
        if href.startswith('/'):
            full_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
        elif href.startswith(('http://', 'https://')):
            # Only include same domain
            parsed_href = urlparse(href)
            if parsed_href.netloc != parsed_base.netloc:
                return None
            full_url = href
        else:
            # Relative path
            full_url = f"{parsed_base.scheme}://{parsed_base.netloc}/{href.lstrip('/')}"
        
        # Skip excluded paths
        if any(excl in full_url.lower() for excl in HARD_EXCLUDE):
            return None
        
        # Skip file extensions that aren't pages
        skip_extensions = ['.pdf', '.jpg', '.jpeg', '.png', '.gif', '.svg', '.css', '.js', '.zip', '.mp4', '.webp']
        if any(full_url.lower().endswith(ext) for ext in skip_extensions):
            return None
        
        return full_url
    
    def _is_priority_url(self, url: str) -> bool:
        """Check if URL matches priority patterns."""
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in PRIORITY_PATHS)
    
    def _normalize_and_dedupe(self, urls: Set[str], base_url: str) -> Set[str]:
        """Normalize URLs and remove duplicates."""
        normalized = set()
        parsed_base = urlparse(base_url)
        
        for url in urls:
            try:
                parsed = urlparse(url)
                
                # Must be same domain
                if parsed.netloc != parsed_base.netloc:
                    continue
                
                # Remove fragment
                url_no_fragment = url.split('#')[0]
                
                # Parse and filter query params
                if '?' in url_no_fragment:
                    base_part, query = url_no_fragment.split('?', 1)
                    params = parse_qs(query, keep_blank_values=True)
                    
                    # Remove tracking params
                    filtered_params = {
                        k: v for k, v in params.items() 
                        if k.lower() not in STRIP_PARAMS
                    }
                    
                    if filtered_params:
                        clean_query = urlencode(filtered_params, doseq=True)
                        url_no_fragment = f"{base_part}?{clean_query}"
                    else:
                        url_no_fragment = base_part
                
                # Normalize trailing slashes (keep for paths, remove for files)
                if not any(url_no_fragment.endswith(ext) for ext in ['.html', '.htm', '.php', '.asp', '.aspx']):
                    url_no_fragment = url_no_fragment.rstrip('/')
                
                normalized.add(url_no_fragment)
            
            except Exception:
                continue
        
        return normalized
