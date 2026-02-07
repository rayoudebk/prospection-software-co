"""Phase 3: LLM-Assisted Triage - page classification and quota-based selection."""

import json
from typing import List, Dict, Optional, Callable
from collections import defaultdict

from google import genai

from app.config import get_settings
from .models import PagePreview, LLMTriageResult
from .constants import COVERAGE_QUOTAS, PAGE_TYPE_PATTERNS


class LLMTriage:
    """LLM-assisted page triage using previews (not just URLs)."""
    
    def __init__(self, progress_callback: Optional[Callable[[str], None]] = None):
        self.progress_callback = progress_callback
        self._client = None
    
    def _log(self, message: str) -> None:
        """Log progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            settings = get_settings()
            if settings.gemini_api_key:
                self._client = genai.Client(api_key=settings.gemini_api_key)
        return self._client
    
    def triage_previews(
        self,
        previews: List[PagePreview],
        max_pages: int = 30
    ) -> List[PagePreview]:
        """
        Triage previews using LLM and enforce category quotas.
        
        Args:
            previews: Scored and sorted previews (best first)
            max_pages: Maximum total pages to select
        
        Returns:
            Selected previews respecting category quotas
        """
        self._log("Triaging pages with LLM...")
        
        client = self._get_client()
        
        if client is None:
            # Fallback to heuristic selection
            self._log("No Gemini API key, using heuristic triage...")
            return self._heuristic_triage(previews, max_pages)
        
        # Get LLM classifications
        triage_results = self._llm_classify_pages(previews[:100])  # Limit to top 100
        
        if not triage_results:
            # Fallback to heuristic if LLM fails
            self._log("LLM triage failed, using heuristic...")
            return self._heuristic_triage(previews, max_pages)
        
        # Apply quota-based selection
        selected = self._apply_quotas(triage_results, max_pages)
        
        # Map back to previews
        selected_urls = {r.url for r in selected}
        result = [p for p in previews if p.url in selected_urls]
        
        self._log(f"Selected {len(result)} pages after quota enforcement")
        return result
    
    def _llm_classify_pages(
        self,
        previews: List[PagePreview]
    ) -> List[LLMTriageResult]:
        """Use LLM to classify pages based on previews."""
        client = self._get_client()
        if not client:
            return []
        
        # Format previews for prompt
        preview_lines = []
        for i, p in enumerate(previews):
            headings_str = ", ".join(p.headings[:5]) if p.headings else "none"
            preview_lines.append(
                f"{i+1}. URL: {p.url}\n"
                f"   Title: {p.title}\n"
                f"   H1: {p.h1}\n"
                f"   Meta: {p.meta_description[:150] if p.meta_description else 'none'}\n"
                f"   Headings: {headings_str}"
            )
        
        previews_text = "\n\n".join(preview_lines)
        
        prompt = f"""You are analyzing web pages for M&A research to understand a company's products, services, and customers.

## Pages to Classify

{previews_text}

## Task
For each page, determine:
1. page_type: one of [product, solutions, features, integrations, customers, services, pricing, security, docs, about, other]
2. contains_tags: which topics it likely covers from [capabilities, services, customers, integrations, pricing, security, docs]
3. priority: 1-10 (10 = highest value for understanding the company)

## Priority Guidelines
- 10: Core product/platform pages with feature lists
- 9: Customer case studies with named clients
- 8: Solutions/use-cases showing value proposition
- 7: Integrations/API/ecosystem pages
- 6: Services/implementation pages
- 5: About/company overview
- 4: Pricing pages
- 3: Security/compliance pages
- 2: Documentation/resources
- 1: Generic or low-value pages

## Output
Return ONLY a JSON array:
```json
[
  {{"url": "...", "page_type": "product", "contains_tags": ["capabilities", "integrations"], "priority": 10}},
  ...
]
```

Include ALL {len(previews)} pages in your response."""

        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
            )
            
            response_text = response.text.strip()
            
            # Extract JSON array
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            
            if start_idx == -1 or end_idx <= start_idx:
                return []
            
            results_json = json.loads(response_text[start_idx:end_idx])
            
            # Convert to LLMTriageResult objects
            results = []
            for item in results_json:
                try:
                    results.append(LLMTriageResult(
                        url=item.get("url", ""),
                        page_type=item.get("page_type", "other"),
                        contains_tags=item.get("contains_tags", []),
                        priority=item.get("priority", 5),
                    ))
                except Exception:
                    continue
            
            return results
        
        except Exception as e:
            self._log(f"LLM classification error: {e}")
            return []
    
    def _apply_quotas(
        self,
        triage_results: List[LLMTriageResult],
        max_pages: int
    ) -> List[LLMTriageResult]:
        """Apply category quotas to select balanced page set."""
        # Group by page type
        by_type: Dict[str, List[LLMTriageResult]] = defaultdict(list)
        for result in triage_results:
            by_type[result.page_type].append(result)
        
        # Sort each group by priority
        for page_type in by_type:
            by_type[page_type].sort(key=lambda x: x.priority, reverse=True)
        
        selected = []
        remaining_budget = max_pages
        
        # First pass: ensure minimums from quotas
        for page_type, (min_count, max_count) in COVERAGE_QUOTAS.items():
            if page_type in by_type and min_count > 0:
                to_take = min(min_count, len(by_type[page_type]), remaining_budget)
                selected.extend(by_type[page_type][:to_take])
                by_type[page_type] = by_type[page_type][to_take:]
                remaining_budget -= to_take
        
        # Second pass: fill up to maximums by priority
        if remaining_budget > 0:
            # Collect all remaining with their priorities
            remaining = []
            for page_type, results in by_type.items():
                max_count = COVERAGE_QUOTAS.get(page_type, (0, 5))[1]
                already_selected = sum(1 for s in selected if s.page_type == page_type)
                can_add = max_count - already_selected
                remaining.extend(results[:can_add])
            
            # Sort by priority and take remaining budget
            remaining.sort(key=lambda x: x.priority, reverse=True)
            selected.extend(remaining[:remaining_budget])
        
        return selected
    
    def _heuristic_triage(
        self,
        previews: List[PagePreview],
        max_pages: int
    ) -> List[PagePreview]:
        """Fallback heuristic triage without LLM."""
        # Classify using URL patterns
        by_type: Dict[str, List[PagePreview]] = defaultdict(list)
        
        for preview in previews:
            page_type = self._classify_by_url(preview.url)
            by_type[page_type].append(preview)
        
        selected = []
        remaining_budget = max_pages
        
        # Apply quotas
        for page_type, (min_count, max_count) in COVERAGE_QUOTAS.items():
            if page_type in by_type:
                to_take = min(max_count, len(by_type[page_type]), remaining_budget)
                selected.extend(by_type[page_type][:to_take])
                remaining_budget -= to_take
                
                if remaining_budget <= 0:
                    break
        
        # Fill remaining with "other" or unclassified
        if remaining_budget > 0 and "other" in by_type:
            selected.extend(by_type["other"][:remaining_budget])
        
        return selected
    
    def _classify_by_url(self, url: str) -> str:
        """Classify page type based on URL patterns."""
        url_lower = url.lower()
        
        for page_type, patterns in PAGE_TYPE_PATTERNS.items():
            if any(pattern in url_lower for pattern in patterns):
                return page_type
        
        return "other"
