"""Gemini API client with google_search grounding."""
import json
from typing import List, Dict, Any

from app.config import get_settings

# #region agent log
_LOG_PATH = "/app/debug.log"  # Inside Docker container mount
def _debug_log(hyp, loc, msg, data=None):
    import time; open(_LOG_PATH, "a").write(json.dumps({"hypothesisId": hyp, "location": loc, "message": msg, "data": data, "timestamp": int(time.time()*1000)}) + "\n")
# #endregion

# Initialize using new google-genai SDK
# #region agent log
try:
    from google import genai
    from google.genai import types
    _debug_log("H18", "gemini_client.py:module", "google-genai SDK imported", {"version": getattr(genai, "__version__", "unknown")})
except ImportError as e:
    _debug_log("H18", "gemini_client.py:module", "google-genai import FAILED", {"error": str(e)})
    genai = None
    types = None
# #endregion


class GeminiClient:
    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        
        # #region agent log
        _debug_log("H19", "gemini_client.py:init", "Creating client", {"has_genai": genai is not None})
        # #endregion
        
        # New SDK uses Client pattern
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model_name = "gemini-2.0-flash"

    def generate_landscape(
        self,
        context_pack: str,
        region_scope: str,
        intent: str,
        exclusions: dict,
    ) -> List[Dict[str, Any]]:
        """
        Generate acquisition landscape using Gemini with web search.
        
        Returns list of candidate companies with evidence.
        """
        bpo_toggle = exclusions.get("bpo_toggle", True)
        exclude_keywords = exclusions.get("keywords", [])
        
        # Extract company names from context to exclude them
        context_preview = (context_pack[:2000] if context_pack else "").lower()
        
        prompt = f"""You are an M&A research analyst. Your job is to USE GOOGLE SEARCH to discover potential acquisition targets.

## YOUR TASK
Search the web for 10 fintech/wealthtech software companies in {region_scope} that could be acquisition targets.

## SEARCH QUERIES TO EXECUTE
You MUST use Google Search with queries like:
- "wealth management software companies {region_scope}"
- "portfolio management system vendors Europe"  
- "investment management platform startups"
- "asset management technology companies"
- "securities trading software firms"
- "financial planning software {region_scope}"
- "robo advisor platforms Europe"
- "fund administration software vendors"

## REFERENCE CONTEXT (DO NOT RETURN THESE COMPANIES)
The buyer already knows about these companies from their context pack - DO NOT include any of them:
- 4TPM, JUMP Technology, Temenos, or any company whose website/details appear in the context below.

Context summary for understanding target profile:
{context_pack[:3000] if context_pack else "Focus on wealth management and securities processing software."}

## CRITERIA FOR TARGET COMPANIES
1. Must be a SOFTWARE company (SaaS, platform, technology product)
2. Must operate in wealth management, asset management, securities, or financial infrastructure
3. Must be based in or serve: {region_scope}
4. Should be small/mid-market (suitable for acquisition - NOT Bloomberg, Refinitiv, SS&C, FIS, Broadridge, Fiserv, Temenos)
5. {"Avoid BPO-heavy companies" if bpo_toggle else "No BPO restriction"}
6. Intent: {intent}

## OUTPUT FORMAT
Return ONLY a JSON array with 10 companies you FOUND VIA SEARCH:
```json
[
  {{
    "name": "Actual Company Name from Search",
    "website": "https://their-real-website.com",
    "country": "Country",
    "why_fit": "2-3 sentences explaining what they do and why they fit the acquisition criteria.",
    "software_signals": ["Signal 1", "Signal 2"],
    "bpo_signals": ["Signal if any"],
    "similarities": "How this company is similar to the reference companies (target customers, technology, capabilities).",
    "watchouts": "Key risks or differences (business model, size, integration complexity, market position).",
    "evidence_links": ["https://source-url-1", "https://source-url-2"],
    "fit_score": 75
  }}
]
```

IMPORTANT: 
- Use Google Search to find REAL companies with REAL websites
- Do NOT return 4TPM, JUMP Technology, Temenos, or any company from the context
- Return exactly 10 different companies
- Return ONLY the JSON array, no other text"""

        try:
            # New SDK: use types.Tool with GoogleSearch
            # #region agent log
            _debug_log("H20", "gemini_client.py:generate_landscape", "Creating google_search tool")
            # #endregion
            
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            
            # #region agent log
            _debug_log("H20", "gemini_client.py:generate_landscape", "Tool created", {"tool": str(google_search_tool)[:300]})
            # #endregion
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                ),
            )
            
            # Extract JSON from response
            response_text = response.text
            
            # #region agent log
            _debug_log("H20", "gemini_client.py:generate_landscape", "API call succeeded", {
                "response_length": len(response_text),
                "response_preview": response_text[:500] if response_text else "empty",
                "has_grounding": hasattr(response, 'candidates') and len(response.candidates) > 0,
            })
            # #endregion
            
            # Find JSON array in response
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            
            if start_idx == -1 or end_idx == 0:
                # Try to parse whole response
                candidates = json.loads(response_text)
            else:
                candidates = json.loads(response_text[start_idx:end_idx])
            
            # #region agent log
            _debug_log("H20", "gemini_client.py:generate_landscape", "Parsed candidates", {
                "count": len(candidates),
                "names": [c.get("name", "?") for c in candidates[:5]]
            })
            # #endregion
            
            return candidates
            
        except json.JSONDecodeError as e:
            # Return empty list on parse error
            print(f"JSON parse error: {e}")
            print(f"Response was: {response.text[:500]}")
            return []
        except Exception as e:
            # #region agent log
            _debug_log("H20", "gemini_client.py:generate_landscape", "API call FAILED", {"error": str(e), "error_type": type(e).__name__})
            # #endregion
            print(f"Gemini API error: {e}")
            raise

    def generate_deep_profile(
        self,
        target_name: str,
        target_website: str | None,
        context_pack: str,
    ) -> Dict[str, Any]:
        """
        Generate deep profile for a specific target company.
        
        Returns markdown profile with evidence links.
        """
        prompt = f"""You are an M&A research analyst conducting deep due diligence on a potential acquisition target.

## Target Company
- Name: {target_name}
- Website: {target_website or "Unknown"}

## Buyer Context
{context_pack if context_pack else "The buyer is a financial technology company seeking to expand capabilities in securities/wealth processing."}

## Task
Research this company thoroughly and produce a comprehensive profile covering:

1. **Overview** - What they do, their core value proposition
2. **Products & Capabilities** - Specific modules, features, technology
3. **Target Customers (ICP)** - Who they sell to, any named clients from case studies
4. **Business Model** - SaaS vs license vs services, pricing signals
5. **Geography & Regulatory** - Where they operate, any compliance certifications
6. **Technology Signals** - Stack, integrations, API availability
7. **Team & Organization** - Size signals, engineering vs ops balance
8. **Risks & Red Flags** - Any concerns from public information
9. **Diligence Questions** - Key questions to validate assumptions
10. **Strategic Fit Hypothesis** - Why the buyer might acquire them (label as hypothesis)

## Requirements
- Only include facts that have sources
- Clearly label hypotheses and assumptions
- Include source URLs for key claims
- Be specific, not generic

## Output Format
Return a JSON object with:
```json
{{
  "markdown": "# Full profile in markdown format...",
  "evidence_links": ["https://source1...", "https://source2..."],
  "key_facts": ["Fact 1", "Fact 2"],
  "open_questions": ["Question 1", "Question 2"]
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
            
            # Find JSON object in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx == -1 or end_idx == 0:
                # Return raw text as markdown if no JSON
                return {
                    "markdown": response_text,
                    "evidence_links": [],
                    "key_facts": [],
                    "open_questions": [],
                }
            
            profile_data = json.loads(response_text[start_idx:end_idx])
            return profile_data
            
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return {
                "markdown": response.text,
                "evidence_links": [],
            }
        except Exception as e:
            print(f"Gemini API error: {e}")
            raise
