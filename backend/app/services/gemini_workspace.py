"""Gemini client methods for workspace-based workflow."""
import json
import re
from typing import List, Dict, Any, Optional

from app.config import get_settings

# Initialize using new google-genai SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


class GeminiWorkspaceClient:
    """Gemini client with focused methods for workspace workflow."""
    
    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = "gemini-2.0-flash"
    
    def run_discovery_universe(
        self,
        context_pack: str,
        taxonomy_bricks: List[Dict[str, str]],
        geo_scope: Dict[str, Any],
        vertical_focus: List[str],
        comparator_mentions: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover candidate universe using web search.
        
        Returns strict JSON array of candidates with citations.
        """
        brick_names = [b.get("name", "") for b in taxonomy_bricks]
        priority_bricks = brick_names[:5]
        
        region = geo_scope.get("region", "EU+UK")
        include_countries = geo_scope.get("include_countries", [])
        exclude_countries = geo_scope.get("exclude_countries", [])
        
        geo_description = region
        if include_countries:
            geo_description += f" (focus: {', '.join(include_countries)})"
        if exclude_countries:
            geo_description += f" (excluding: {', '.join(exclude_countries)})"
        
        verticals = ", ".join(vertical_focus) if vertical_focus else "asset managers, wealth managers, banks, insurers"
        comparator_mentions = comparator_mentions or []

        seed_lines: List[str] = []
        for mention in comparator_mentions[:40]:
            name = str(mention.get("company_name") or "").strip()
            url = str(mention.get("company_url") or "").strip()
            tags = [
                str(tag).strip()
                for tag in (mention.get("category_tags") or [])
                if isinstance(tag, str) and str(tag).strip()
            ]
            snippet = ""
            snippets = mention.get("listing_text_snippets") or []
            if isinstance(snippets, list) and snippets:
                snippet = str(snippets[0])[:160]
            if not name:
                continue
            line = f"- {name}"
            if url:
                line += f" ({url})"
            if tags:
                line += f" | tags: {', '.join(tags[:3])}"
            if snippet:
                line += f" | snippet: {snippet}"
            seed_lines.append(line)
        seed_section = "\n".join(seed_lines) if seed_lines else "- No directory seeds provided."
        
        prompt = f"""You are an M&A research analyst. Your task is to use Google Search to discover potential acquisition targets.

## BUYER CONTEXT
{context_pack[:4000] if context_pack else "The buyer is a financial technology company seeking to expand capabilities."}

## TARGET PROFILE
- **Geography:** {geo_description}
- **Verticals:** {verticals}
- **Capability bricks of interest:** {', '.join(priority_bricks)}

## COMPARATOR DIRECTORY SEEDS
Use these as starting comparators, then expand to close peers/adjacencies with independent evidence:
{seed_section}

## SEARCH TASK
Use Google Search to find 15-20 software companies that:
1. Operate in wealth management, asset management, securities processing, or investment technology
2. Are based in or primarily serve: {region}
3. Are small/mid-market (NOT mega vendors like Bloomberg, Refinitiv, SS&C, FIS, Broadridge, Fiserv, Temenos)
4. Have products/modules that map to one or more of these bricks: {', '.join(brick_names)}
5. Primarily sell to B2B financial institutions (asset managers, wealth managers, private banks, fund admins, insurers)
6. Prefer enterprise software model (contact-sales/quote-led acceptable); avoid low-ticket self-serve apps

## SEARCH QUERIES TO TRY
- "wealth management software companies {region}"
- "portfolio management system vendors Europe"
- "investment management platform companies"
- "asset management technology startups"
- "fund administration software vendors"
- "financial planning software Europe"
- Company-specific searches for names you discover

## OUTPUT FORMAT
Return ONLY a valid JSON array. Each object must have these exact fields:
```json
[
  {{
    "name": "Company Name",
    "website": "https://company-website.com",
    "hq_country": "UK",
    "likely_verticals": ["asset_manager", "wealth_manager"],
    "employee_estimate": 45,
    "capability_signals": ["Portfolio management", "Client reporting"],
    "qualification": {{
      "go_to_market": "b2b_enterprise|b2b_mixed|b2c|unknown",
      "software_heaviness": 4,
      "pricing_model": "enterprise_quote|public_tiered|usage|unknown",
      "public_price_floor_usd_month": null,
      "target_customer": "asset_managers|wealth_managers|banks|fund_admins|retail_investors|mixed|unknown"
    }},
    "why_relevant": [
      {{"text": "Capability match evidence", "citation_url": "https://source-url.com", "dimension": "capability"}},
      {{"text": "B2B ICP evidence", "citation_url": "https://source-url.com", "dimension": "icp"}},
      {{"text": "Pricing/GTM evidence (or explicit no public pricing)", "citation_url": "https://source-url.com", "dimension": "pricing_gtm"}}
    ]
  }}
]
```

## CRITICAL REQUIREMENTS
1. Every factual claim MUST have a citation_url
2. Do NOT include companies mentioned in the buyer context
3. website MUST be a real, working URL
4. Return ONLY the JSON array, no other text
5. At least 2 why_relevant items per company
6. Prefer first-party or official/public source links over aggregator redirects
7. For UK/FR companies, include at least one filing or registry citation (Companies House / INPI / Infogreffe / Pappers) whenever available"""

        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            response_text = response.text or "[]"

            parsed_payload: Any
            try:
                parsed_payload = json.loads(response_text)
            except json.JSONDecodeError:
                def extract_json_objects(blob: str) -> List[Dict[str, Any]]:
                    objects: List[Dict[str, Any]] = []
                    depth = 0
                    start_idx = None
                    in_string = False
                    escape = False
                    for idx, ch in enumerate(blob):
                        if in_string:
                            if escape:
                                escape = False
                            elif ch == "\\":
                                escape = True
                            elif ch == "\"":
                                in_string = False
                            continue
                        if ch == "\"":
                            in_string = True
                            continue
                        if ch == "{":
                            if depth == 0:
                                start_idx = idx
                            depth += 1
                        elif ch == "}":
                            if depth > 0:
                                depth -= 1
                                if depth == 0 and start_idx is not None:
                                    raw_obj = blob[start_idx:idx + 1]
                                    try:
                                        parsed_obj = json.loads(raw_obj)
                                    except json.JSONDecodeError:
                                        continue
                                    if isinstance(parsed_obj, dict):
                                        objects.append(parsed_obj)
                    return objects

                cleaned = re.sub(r"[\x00-\x1f]+", " ", response_text)
                start_idx = cleaned.find("[")
                end_idx = cleaned.rfind("]") + 1
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        parsed_payload = json.loads(cleaned[start_idx:end_idx])
                    except json.JSONDecodeError:
                        parsed_payload = None
                else:
                    parsed_payload = None

                if parsed_payload is None:
                    recovered_objects = extract_json_objects(cleaned[:40000])
                    if recovered_objects:
                        parsed_payload = recovered_objects

                if parsed_payload is None:
                    repair_prompt = f"""Repair the following malformed JSON into a valid JSON array.

Rules:
- Output ONLY valid JSON array
- Keep original fields when possible
- If an object is broken, keep partial object with available keys
- Do not add explanations

Malformed payload:
{cleaned[:20000]}
"""
                    repair_response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=repair_prompt,
                    )
                    repaired_text = (repair_response.text or "[]").strip()
                    start_idx = repaired_text.find("[")
                    end_idx = repaired_text.rfind("]") + 1
                    if start_idx == -1 or end_idx <= start_idx:
                        recovered_objects = extract_json_objects(repaired_text[:40000])
                        if recovered_objects:
                            parsed_payload = recovered_objects
                        else:
                            raise ValueError("No JSON array found in repaired response")
                    else:
                        try:
                            parsed_payload = json.loads(repaired_text[start_idx:end_idx])
                        except json.JSONDecodeError:
                            recovered_objects = extract_json_objects(repaired_text[:40000])
                            if recovered_objects:
                                parsed_payload = recovered_objects
                            else:
                                raise

            if isinstance(parsed_payload, dict):
                candidates = []
                for key in ("candidates", "items", "companies", "vendors", "targets", "results", "data"):
                    if isinstance(parsed_payload.get(key), list):
                        candidates = parsed_payload.get(key, [])
                        break
                if not candidates:
                    for value in parsed_payload.values():
                        if isinstance(value, list):
                            candidates = value
                            break
                if not candidates and parsed_payload.get("name") and parsed_payload.get("website"):
                    candidates = [parsed_payload]
            elif isinstance(parsed_payload, list):
                candidates = parsed_payload
            else:
                candidates = []
            
            # Validate structure
            validated = []
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                if not c.get("name") or not c.get("website"):
                    continue
                validated.append({
                    "name": c.get("name", ""),
                    "website": c.get("website", ""),
                    "hq_country": c.get("hq_country", "Unknown"),
                    "likely_verticals": c.get("likely_verticals", []),
                    "employee_estimate": c.get("employee_estimate"),
                    "capability_signals": c.get("capability_signals", []),
                    "qualification": c.get("qualification", {}),
                    "why_relevant": c.get("why_relevant", []),
                })
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error in discovery: {e}")
            return []
        except Exception as e:
            print(f"Gemini API error in discovery: {e}")
            raise
    
    def run_enrich_modules(self, vendor_url: str, taxonomy_bricks: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Enrich vendor with module/capability mapping to bricks.
        
        Returns strict JSON with modules mapped to bricks.
        """
        brick_names = [b.get("name", "") for b in taxonomy_bricks]
        brick_list = "\n".join([f"- {b.get('id', '')}: {b.get('name', '')}" for b in taxonomy_bricks])
        
        prompt = f"""Research the company at {vendor_url} and identify their software products/modules.

## BRICK TAXONOMY
Map each product/module to one of these bricks:
{brick_list}

## TASK
1. Search for {vendor_url} products, solutions, platform, modules
2. Identify their distinct software capabilities
3. Map each to a brick from the taxonomy
4. Include evidence URLs for each claim

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "modules": [
    {{
      "name": "Product/Module Name",
      "brick_id": "brick-uuid-here",
      "brick_name": "PMS",
      "description": "What this module does",
      "evidence_urls": ["https://vendor-url.com/products/module"]
    }}
  ]
}}
```

Return ONLY the JSON object."""

        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            response_text = response.text
            
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                return {"modules": []}
            
            result = json.loads(response_text[start_idx:end_idx])
            return result
            
        except json.JSONDecodeError:
            return {"modules": []}
        except Exception as e:
            print(f"Gemini API error in enrich_modules: {e}")
            raise
    
    def run_enrich_customers(self, vendor_url: str) -> Dict[str, Any]:
        """
        Enrich vendor with customer information.
        
        Returns strict JSON with customer list.
        """
        prompt = f"""Research the company at {vendor_url} and find their customers/clients.

## TASK
1. Search for {vendor_url} customers, clients, case studies, testimonials
2. Identify named customers with evidence
3. Note the context (case study, logo, press release, etc.)

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "customers": [
    {{
      "name": "Customer Company Name",
      "context": "case_study|logo|testimonial|press_release",
      "evidence_url": "https://source-url.com"
    }}
  ]
}}
```

Return ONLY the JSON object."""

        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            response_text = response.text
            
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                return {"customers": []}
            
            result = json.loads(response_text[start_idx:end_idx])
            return result
            
        except json.JSONDecodeError:
            return {"customers": []}
        except Exception as e:
            print(f"Gemini API error in enrich_customers: {e}")
            raise
    
    def run_enrich_hiring(self, vendor_url: str) -> Dict[str, Any]:
        """
        Enrich vendor with hiring/team information.
        
        Returns strict JSON with hiring insights.
        """
        prompt = f"""Research the company at {vendor_url} and find hiring/team information.

## TASK
1. Search for {vendor_url} careers, jobs, team, employees
2. Look for job postings, LinkedIn company page, team pages
3. Identify engineering vs sales/ops mix signals
4. Note approximate team size if discoverable

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "hiring": {{
    "postings": [
      {{
        "title": "Job Title",
        "location": "City, Country",
        "category": "engineering|product|sales|operations|other",
        "evidence_url": "https://source-url.com"
      }}
    ],
    "mix_summary": {{
      "engineering_heavy": true,
      "team_size_estimate": "10-50",
      "notes": "Observations about team composition"
    }}
  }}
}}
```

Return ONLY the JSON object."""

        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            response_text = response.text
            
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                return {"hiring": {"postings": [], "mix_summary": {}}}
            
            result = json.loads(response_text[start_idx:end_idx])
            return result
            
        except json.JSONDecodeError:
            return {"hiring": {"postings": [], "mix_summary": {}}}
        except Exception as e:
            print(f"Gemini API error in enrich_hiring: {e}")
            raise
    
    def run_enrich_full(
        self,
        vendor_url: str,
        vendor_name: str,
        taxonomy_bricks: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Full enrichment in a single call.
        
        Returns complete dossier JSON.
        """
        brick_list = "\n".join([f"- {b.get('id', '')}: {b.get('name', '')}" for b in taxonomy_bricks])
        
        prompt = f"""Research the company "{vendor_name}" at {vendor_url} thoroughly.

## BRICK TAXONOMY
{brick_list}

## TASKS
1. **Modules**: Find their products/modules and map to bricks
2. **Customers**: Find named customers from case studies, logos, press
3. **Hiring**: Find job postings and team composition signals
4. **Integrations**: Find technology integrations and partnerships

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{{
  "modules": [
    {{
      "name": "Module Name",
      "brick_id": "uuid",
      "brick_name": "PMS",
      "description": "What it does",
      "evidence_urls": ["https://..."]
    }}
  ],
  "customers": [
    {{
      "name": "Customer Name",
      "context": "case_study",
      "evidence_url": "https://..."
    }}
  ],
  "hiring": {{
    "postings": [
      {{
        "title": "Job Title",
        "location": "Location",
        "category": "engineering",
        "evidence_url": "https://..."
      }}
    ],
    "mix_summary": {{
      "engineering_heavy": true,
      "team_size_estimate": "10-50",
      "notes": "..."
    }}
  }},
  "integrations": [
    {{
      "name": "Integration Name",
      "type": "data_provider|custodian|broker|other",
      "evidence_url": "https://..."
    }}
  ]
}}
```

CRITICAL: Return ONLY the JSON object, no other text. Every claim needs evidence_url."""

        try:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            response_text = response.text
            
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                return {
                    "modules": [],
                    "customers": [],
                    "hiring": {"postings": [], "mix_summary": {}},
                    "integrations": []
                }
            
            result = json.loads(response_text[start_idx:end_idx])
            
            # Ensure all keys exist
            if "modules" not in result:
                result["modules"] = []
            if "customers" not in result:
                result["customers"] = []
            if "hiring" not in result:
                result["hiring"] = {"postings": [], "mix_summary": {}}
            if "integrations" not in result:
                result["integrations"] = []
            
            return result
            
        except json.JSONDecodeError:
            return {
                "modules": [],
                "customers": [],
                "hiring": {"postings": [], "mix_summary": {}},
                "integrations": []
            }
        except Exception as e:
            print(f"Gemini API error in enrich_full: {e}")
            raise
    
    def summarize_context_pack(self, raw_markdown: str, buyer_url: str) -> str:
        """
        Generate AI summary of crawled context pack.
        
        Returns enhanced markdown with structured insights.
        """
        prompt = f"""Analyze this context pack about a company and create a structured summary.

## RAW CONTEXT
{raw_markdown[:8000]}

## TASK
Create a structured summary covering:
1. **Company Overview**: What they do, value proposition
2. **Products & Capabilities**: Key modules and features
3. **Target Customers**: Who they serve (ICP)
4. **Technology Signals**: Integrations, tech stack hints
5. **Differentiators**: What makes them unique

Keep it concise but fact-rich. Preserve source citations.

Return as clean markdown."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            
            return response.text
            
        except Exception as e:
            print(f"Gemini API error in summarize: {e}")
            return raw_markdown
