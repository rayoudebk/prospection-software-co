"""Phase 5: Output Generation - markdown and structured context pack assembly."""

from typing import List, Dict, Optional
from urllib.parse import urlparse
from collections import defaultdict

from .models import (
    CrawledPage,
    ContextPack,
    Signal,
    CustomerEvidence,
    ContentBlock,
)


class OutputGenerator:
    """Generates final context pack output with markdown view."""
    
    # Max chars per page in markdown output (truncation at render, not crawl)
    MAX_CONTENT_PER_PAGE_RENDER = 2000
    MAX_TOTAL_MARKDOWN = 50000
    
    def generate_context_pack(
        self,
        pages: List[CrawledPage],
        base_url: str,
    ) -> ContextPack:
        """
        Assemble final context pack from crawled pages.
        
        Aggregates signals, generates markdown view.
        """
        # Extract company name
        company_name = self._extract_company_name(pages, base_url)
        
        # Aggregate all signals from pages
        all_signals = []
        all_customer_evidence = []
        
        for page in pages:
            all_signals.extend(page.signals)
            all_customer_evidence.extend(page.customer_evidence)
        
        # Dedupe signals
        unique_signals = self._dedupe_signals(all_signals)
        unique_customers = self._dedupe_customer_evidence(all_customer_evidence)
        
        # Generate markdown
        raw_markdown = self._generate_markdown(
            company_name,
            base_url,
            pages,
            unique_signals,
            unique_customers,
        )
        
        # Count product pages
        product_types = {"product", "solutions", "features", "platform"}
        product_pages_count = sum(1 for p in pages if p.page_type in product_types)
        
        # Generate summary
        summary = self._generate_summary(pages)
        
        return ContextPack(
            company_name=company_name,
            website=base_url,
            pages=pages,
            signals=unique_signals,
            customer_evidence=unique_customers,
            raw_markdown=raw_markdown,
            product_pages_count=product_pages_count,
            summary=summary,
        )
    
    def _extract_company_name(self, pages: List[CrawledPage], base_url: str) -> str:
        """Extract company name from pages or domain."""
        # Try about pages first
        for page in pages:
            if page.page_type == "about" or "/about" in page.url.lower():
                if page.title:
                    # Often "Company Name | About" or "About - Company Name"
                    parts = page.title.replace(" - ", "|").split("|")
                    if len(parts) >= 2:
                        # Take the shorter part (usually company name)
                        name = min(parts, key=len).strip()
                        if len(name) > 2 and len(name) < 50:
                            return name
        
        # Try homepage
        for page in pages:
            path = urlparse(page.url).path
            if path in ["", "/", "/home", "/index.html"]:
                if page.title:
                    parts = page.title.replace(" - ", "|").split("|")
                    if parts:
                        return parts[0].strip()
        
        # Fallback to domain
        parsed = urlparse(base_url)
        domain = parsed.netloc.replace("www.", "")
        name = domain.split(".")[0]
        return name.title()
    
    def _dedupe_signals(self, signals: List[Signal]) -> List[Signal]:
        """Deduplicate signals by value."""
        seen = set()
        unique = []
        
        for signal in signals:
            key = (signal.type, signal.value.lower())
            if key not in seen:
                seen.add(key)
                unique.append(signal)
        
        return unique
    
    def _dedupe_customer_evidence(
        self,
        evidence: List[CustomerEvidence]
    ) -> List[CustomerEvidence]:
        """Deduplicate customer evidence by name."""
        seen = set()
        unique = []
        
        for e in evidence:
            key = e.name.lower()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        
        return unique
    
    def _generate_summary(self, pages: List[CrawledPage]) -> str:
        """Generate brief summary from pages."""
        # Try about page content
        for page in pages:
            if page.page_type == "about":
                if page.raw_content:
                    return page.raw_content[:500]
        
        # Try homepage
        for page in pages:
            path = urlparse(page.url).path
            if path in ["", "/", "/home"]:
                if page.raw_content:
                    return page.raw_content[:500]
        
        # Fallback to first page with content
        for page in pages:
            if page.raw_content:
                return page.raw_content[:500]
        
        return ""
    
    def _generate_markdown(
        self,
        company_name: str,
        website: str,
        pages: List[CrawledPage],
        signals: List[Signal],
        customer_evidence: List[CustomerEvidence],
    ) -> str:
        """Generate structured markdown from crawled data."""
        sections = []
        total_chars = 0
        
        # Header
        sections.append(f"# {company_name}\n")
        sections.append(f"**Website:** {website}\n")
        total_chars += len(sections[-1]) + len(sections[-2])
        
        # Group pages by type
        by_type: Dict[str, List[CrawledPage]] = defaultdict(list)
        for page in pages:
            by_type[page.page_type].append(page)
        
        # About section
        if "about" in by_type:
            section = self._format_section(
                "About",
                by_type["about"],
                max_pages=2,
                max_content=800,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Products section
        if "product" in by_type:
            section = self._format_section(
                "Products & Platform",
                by_type["product"],
                max_pages=5,
                max_content=1000,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Solutions section
        if "solutions" in by_type:
            section = self._format_section(
                "Solutions & Use Cases",
                by_type["solutions"],
                max_pages=4,
                max_content=800,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Features section
        if "features" in by_type:
            section = self._format_section(
                "Features & Capabilities",
                by_type["features"],
                max_pages=3,
                max_content=600,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Integrations section
        if "integrations" in by_type:
            section = self._format_section(
                "Integrations & Partners",
                by_type["integrations"],
                max_pages=3,
                max_content=500,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Services section
        if "services" in by_type:
            section = self._format_section(
                "Services & Implementation",
                by_type["services"],
                max_pages=2,
                max_content=500,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Customers section (combine with customer evidence)
        customer_section = self._format_customers_section(
            by_type.get("customers", []),
            customer_evidence,
        )
        if customer_section:
            sections.append(customer_section)
            total_chars += len(customer_section)
        
        # Pricing section
        if "pricing" in by_type:
            section = self._format_section(
                "Pricing",
                by_type["pricing"],
                max_pages=1,
                max_content=400,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Security section
        if "security" in by_type:
            section = self._format_section(
                "Security & Compliance",
                by_type["security"],
                max_pages=1,
                max_content=400,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Docs/Resources section
        if "docs" in by_type:
            section = self._format_section(
                "Documentation & Resources",
                by_type["docs"],
                max_pages=2,
                max_content=400,
            )
            if section:
                sections.append(section)
                total_chars += len(section)
        
        # Signals summary
        if signals:
            signals_section = self._format_signals_section(signals)
            if signals_section:
                sections.append(signals_section)
        
        return "\n".join(sections)
    
    def _format_section(
        self,
        title: str,
        pages: List[CrawledPage],
        max_pages: int = 3,
        max_content: int = 800,
    ) -> str:
        """Format a section of pages."""
        if not pages:
            return ""
        
        lines = [f"\n## {title}\n"]
        
        for page in pages[:max_pages]:
            lines.append(f"\n### {page.title}\n")
            lines.append(f"*Source: {page.url}*\n")
            
            # Use content blocks if available, otherwise raw content
            if page.blocks:
                content = self._format_blocks(page.blocks, max_content)
            elif page.raw_content:
                content = page.raw_content[:max_content]
                if len(page.raw_content) > max_content:
                    content += "..."
            else:
                content = ""
            
            if content:
                lines.append(f"\n{content}\n")
        
        return "\n".join(lines)
    
    def _format_blocks(
        self,
        blocks: List[ContentBlock],
        max_chars: int
    ) -> str:
        """Format content blocks into markdown."""
        lines = []
        total_chars = 0
        
        for block in blocks:
            if total_chars >= max_chars:
                lines.append("...")
                break
            
            if block.type == "heading":
                prefix = "#" * (block.level + 1)  # Shift down since we're in a subsection
                line = f"{prefix} {block.content}"
            elif block.type == "list":
                line = block.content
            elif block.type == "table":
                line = f"```\n{block.content}\n```"
            else:
                line = block.content
            
            lines.append(line)
            total_chars += len(line)
        
        return "\n\n".join(lines)
    
    def _format_customers_section(
        self,
        customer_pages: List[CrawledPage],
        customer_evidence: List[CustomerEvidence],
    ) -> str:
        """Format customers section combining pages and evidence."""
        lines = ["\n## Customers & Case Studies\n"]
        has_content = False
        
        # Add customer evidence
        if customer_evidence:
            has_content = True
            lines.append("\n### Identified Customers\n")
            
            # Group by evidence type
            logo_customers = [e for e in customer_evidence if e.evidence_type in ("logo_alt", "aria_label")]
            text_customers = [e for e in customer_evidence if e.evidence_type == "text_mention"]
            
            if logo_customers:
                lines.append("**From logo sections:**\n")
                for e in logo_customers[:20]:  # Limit to 20
                    lines.append(f"- {e.name}")
                lines.append("")
            
            if text_customers:
                lines.append("**From case studies/mentions:**\n")
                for e in text_customers[:10]:
                    lines.append(f"- {e.name}")
                lines.append("")
        
        # Add customer pages
        for page in customer_pages[:3]:
            has_content = True
            lines.append(f"\n### {page.title}\n")
            lines.append(f"*Source: {page.url}*\n")
            
            if page.raw_content:
                content = page.raw_content[:500]
                if len(page.raw_content) > 500:
                    content += "..."
                lines.append(f"\n{content}\n")
        
        if has_content:
            return "\n".join(lines)
        return ""
    
    def _format_signals_section(self, signals: List[Signal]) -> str:
        """Format extracted signals summary."""
        lines = ["\n## Extracted Signals\n"]
        
        # Group by type
        by_type: Dict[str, List[Signal]] = defaultdict(list)
        for signal in signals:
            by_type[signal.type].append(signal)
        
        if "capability" in by_type:
            lines.append("\n### Capabilities\n")
            for s in by_type["capability"][:15]:
                lines.append(f"- {s.value}")
        
        if "service" in by_type:
            lines.append("\n### Services\n")
            for s in by_type["service"][:10]:
                lines.append(f"- {s.value}")
        
        if "integration" in by_type:
            lines.append("\n### Integrations\n")
            for s in by_type["integration"][:10]:
                lines.append(f"- {s.value}")
        
        return "\n".join(lines) if len(lines) > 1 else ""
