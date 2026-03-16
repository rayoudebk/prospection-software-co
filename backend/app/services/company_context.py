"""Company-context bootstrap, normalization, and scope-derived discovery helpers."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from app.config import get_settings
from app.models.company_context import CompanyContextPack
from app.models.workspace import CompanyProfile
from app.services.llm.orchestrator import LLMOrchestrator
from app.services.llm.types import LLMOrchestrationError, LLMRequest, LLMStage
from app.services.reporting import normalize_domain
from app.services.retrieval.url_normalization import normalize_url

CLAIM_USER_STATUSES = {"system", "confirmed", "edited", "removed"}
EXPANSION_ITEM_STATUSES = {
    "source_grounded",
    "corroborated_expansion",
    "hypothesis",
    "user_kept",
    "user_removed",
    "user_deprioritized",
}
EXPANSION_ITEM_TYPES = {
    "adjacent_capability",
    "adjacent_customer_segment",
    "named_account_anchor",
    "geography_expansion",
}

DEFAULT_OPEN_QUESTIONS = [
    "Which customer segment is most strategic for target discovery?",
    "What revenue model matters most for this sourcing pass?",
    "What size window should the system optimize for?",
]

BUYER_EVIDENCE_MIN_SCORE = 3
MARKET_MAP_REASONING_PROMPT_VERSION = "v2"
MARKET_MAP_REASONING_OUTPUT_SCHEMA = {
    "source_summary": "string",
    "customer_node_ids": ["taxonomy_id"],
    "workflow_node_ids": ["taxonomy_id"],
    "capability_node_ids": ["taxonomy_id"],
    "delivery_or_integration_node_ids": ["taxonomy_id"],
    "active_lens_ids": ["lens_id"],
    "adjacency_hypotheses": [
        {
            "text": "string",
            "supporting_node_ids": ["taxonomy_id"],
            "confidence": 0.72,
        }
    ],
    "confidence_gaps": ["string"],
    "open_questions": ["string"],
}
EXPANSION_RESEARCH_PROMPT_VERSION = "v1"
EXPANSION_RESEARCH_OUTPUT_SCHEMA = {
    "adjacent_capabilities": [
        {
            "label": "Voting rights / proxy voting",
            "why_it_matters": "string",
            "evidence_urls": ["https://example.com"],
            "source_entity_names": ["Comparator A", "BNP Paribas"],
            "market_importance": "high|medium|low",
            "operational_centrality": "core|meaningful|peripheral",
            "workflow_criticality": "high|medium|low",
            "daily_operator_usage": "high|medium|low",
            "switching_cost_intensity": "high|medium|low",
            "confidence": 0.62,
        }
    ],
    "adjacent_customer_segments": [],
    "named_account_anchors": [],
    "geography_expansions": [],
}
EXPANSION_BRIEF_OUTPUT_SCHEMA = {
    "reasoning_status": "success|degraded|not_run",
    "reasoning_warning": "string|null",
    "adjacent_capabilities": [
        {
            "id": "expansion_xxx",
            "label": "Voting rights / proxy voting",
            "expansion_type": "adjacent_capability",
            "status": "source_grounded|corroborated_expansion|hypothesis|user_kept|user_removed|user_deprioritized",
            "confidence": 0.62,
            "why_it_matters": "string",
            "evidence_urls": ["https://example.com"],
            "supporting_node_ids": ["taxonomy_xxx"],
            "source_entity_names": ["Comparator A", "BNP Paribas"],
            "market_importance": "high|medium|low",
            "operational_centrality": "core|meaningful|peripheral",
            "workflow_criticality": "high|medium|low",
            "daily_operator_usage": "high|medium|low",
            "switching_cost_intensity": "high|medium|low",
            "priority_tier": "core_adjacent|meaningful_adjacent|edge_case",
        }
    ],
    "adjacent_customer_segments": [],
    "named_account_anchors": [],
    "geography_expansions": [],
}

CUSTOMER_KEYWORDS = (
    "asset manager",
    "wealth manager",
    "private equity",
    "fund administrator",
    "bank",
    "insurer",
    "advisor",
    "enterprise",
    "mid-market",
    "SMB",
    "fund manager",
    "portfolio manager",
    "operations team",
    "finance team",
    "compliance team",
    "healthcare",
    "healthcare provider",
    "hospital",
    "doctor",
    "physician",
    "clinic",
    "pharmacy",
    "medical practice",
)

BUSINESS_MODEL_PATTERNS = (
    ("saas", "SaaS / subscription software"),
    ("subscription", "Subscription-based software"),
    ("recurring", "Recurring revenue model"),
    ("license", "License-based software"),
    ("licence", "License-based software"),
    ("implementation", "Implementation and onboarding services"),
    ("managed service", "Managed services offering"),
    ("professional services", "Professional services revenue"),
    ("consulting", "Consulting-led services"),
    ("multi-year", "Multi-year enterprise contracts"),
    ("annual contract", "Annual contract model"),
)

DEPLOYMENT_PATTERNS = (
    ("cloud", "Cloud-delivered product"),
    ("api", "API / integration-led deployment"),
    ("integrat", "Integration-heavy deployment"),
    ("on-prem", "On-premise deployment option"),
    ("hosted", "Hosted deployment option"),
    ("implementation", "High-touch implementation required"),
)

NEGATIVE_CONSTRAINT_PATTERNS = (
    ("no saas", "Exclude SaaS-first software companies"),
    ("non-saas", "Exclude SaaS-first software companies"),
    ("not saas", "Exclude SaaS-first software companies"),
    ("no services", "Exclude services-led businesses"),
    ("no consulting", "Exclude consulting-led businesses"),
)

TAXONOMY_LAYERS = {"customer_archetype", "workflow", "capability", "delivery_or_integration"}
TAXONOMY_SCOPE_STATUSES = {"in_scope", "out_of_scope", "removed"}
LENS_TYPES = {
    "same_customer_different_product",
    "same_product_different_customer",
    "different_product_different_customer_within_market_box",
}
RELATION_TYPES = {"buys_capability", "supports_workflow", "adjacent_to"}
JOB_PAGE_KEYWORDS = (
    "product",
    "workflow",
    "customer",
    "client",
    "integration",
    "implementation",
    "operations",
    "compliance",
    "portfolio",
    "staffing",
    "shift",
    "planning",
    "scheduling",
    "reporting",
    "data",
    "ai",
    "api",
    "platform",
)
CAPABILITY_KEYWORDS = (
    "pms/oms",
    "pms",
    "oms",
    "straight-through processing",
    "stp",
    "portfolio management system",
    "order management system",
    "wealth management platform",
    "client lifecycle management",
    "clm",
    "onboarding platform",
    "staffing platform",
    "workforce management platform",
    "replacement planning",
    "shift replacement",
)
DELIVERY_OR_INTEGRATION_KEYWORDS = (
    "apis rest",
    "rest api",
    "documentation api",
    "api documentation",
    "integration",
    "integrations",
    "partner ecosystem",
    "modular platform",
    "infrastructure",
    "cloud",
    "hosted",
)
DELIVERY_OR_INTEGRATION_CANONICALS = (
    ("apis rest", "REST API"),
    ("rest api", "REST API"),
    ("documentation api", "API documentation"),
    ("api documentation", "API documentation"),
    ("partner ecosystem", "Partner ecosystem"),
    ("modular platform", "Modular platform"),
    ("cloud", "Cloud delivery"),
    ("hosted", "Hosted deployment"),
    ("infrastructure", "Infrastructure"),
)
GENERIC_CAPABILITY_HEADINGS = {
    "capacités clés",
    "fonctionnalités détaillées",
    "bénéfices par audience",
    "benefits by audience",
    "capabilities",
    "key capabilities",
    "functionalities",
    "detailed functionalities",
}
GENERIC_SUPPORT_HEADINGS = {
    "architecture & intégration",
    "architecture & integration",
    "investisseurs",
    "gérants",
    "entités tiers",
    "brokers",
    "marchés",
    "back office",
    "systèmes d'information",
    "systems of record",
    "pour les gérants et équipes de trading",
    "pour les équipes it",
}
SENTENCE_LIKE_CAPABILITY_PREFIXES = (
    "le client ",
    "la suite ",
    "ce que ",
    "pour les ",
    "pour la ",
    "pour le ",
    "our ",
    "we ",
    "clients can ",
    "this ",
    "these ",
)
CAPABILITY_DEMOTION_PREFIXES = (
    "dépôt ",
    "assignation ",
    "cantonnement ",
    "interface dédiée ",
    "plan de déploiement ",
    "pour les ",
    "pour la ",
    "pour le ",
)
TRAILING_CONNECTOR_WORDS = {
    "and",
    "or",
    "ou",
    "et",
    "for",
    "to",
    "de",
    "des",
    "du",
    "la",
    "le",
    "les",
    "au",
    "aux",
    "with",
    "your",
    "my",
    "our",
    "votre",
    "vos",
    "notre",
    "nos",
}
OPEN_QUESTION_BLOCKLIST_TERMS = (
    "roadmap",
    "growth",
    "go-to-market",
    "go to market",
    "commercial expansion",
    "strategic focus",
    "strategic priority",
    "prioritized for growth",
    "prioritised for growth",
)
CAPABILITY_QUALITY_KEYWORDS = (
    "pms",
    "oms",
    "stp",
    "portfolio",
    "portefeuille",
    "portefeuilles",
    "génération",
    "order",
    "ordre",
    "trade",
    "pré-trade",
    "post-trade",
    "trading",
    "routing",
    "routage",
    "modélisation",
    "arbitrage",
    "arbitrages",
    "connectivité",
    "structuration",
    "reporting",
    "performance",
    "compliance",
    "contrôle",
    "réglementaire",
    "monitoring",
    "allocation",
    "benchmark",
    "api",
    "sepa",
    "payment",
    "paiement",
)
WORKFLOW_KEYWORDS = (
    "front office",
    "front office titres",
    "front-office",
    "trading operations",
    "front-to-back office trading operations",
    "portfolio management",
    "portfolio analytics",
    "order management",
    "portfolio reporting",
    "reporting",
    "fund operations",
    "fund administration",
    "reconciliation",
    "compliance",
    "risk management",
    "implementation",
    "workflow",
    "operations",
    "back office titres",
    "paiements & comptes espèces",
    "staffing",
    "shift replacement",
    "shift planning",
    "scheduling",
    "workforce planning",
    "internal mobility",
    "resource planning",
    "replacement planning",
    "voting rights",
)
GENERIC_WORKFLOW_PHRASES = {
    "front",
    "back",
    "front office",
    "back office",
    "workflow",
    "operations",
    "compliance",
    "implementation",
    "infrastructure",
}
CAPABILITY_CANONICAL_PATTERNS = (
    ("plateforme wealth management", "Wealth management platform"),
    ("wealth management platform", "Wealth management platform"),
    ("client lifecycle management", "Client lifecycle management"),
    ("order management system", "Order management system"),
    ("portfolio management system", "Portfolio management system"),
    ("pms/oms", "PMS/OMS"),
    ("oms", "OMS"),
    ("pms", "PMS"),
    ("clm", "CLM"),
    ("stp", "STP"),
)
CUSTOMER_ARCHETYPE_PATTERNS = (
    ("asset managers", "asset manager"),
    ("asset manager", "asset manager"),
    ("private banks", "private bank"),
    ("private bank", "private bank"),
    ("wealth managers", "wealth manager"),
    ("wealth manager", "wealth manager"),
    ("fund managers", "fund manager"),
    ("fund manager", "fund manager"),
    ("institutional investors", "institutional investor"),
    ("institutional investor", "institutional investor"),
    ("fund administrators", "fund administrator"),
    ("fund administrator", "fund administrator"),
    ("banks", "bank"),
    ("bank", "bank"),
    ("insurers", "insurer"),
    ("insurer", "insurer"),
    ("advisors", "advisor"),
    ("advisor", "advisor"),
    ("hospitals", "hospital"),
    ("hospital", "hospital"),
    ("care facilities", "care facility"),
    ("care facility", "care facility"),
    ("healthcare providers", "healthcare provider"),
    ("healthcare provider", "healthcare provider"),
    ("clinics", "clinic"),
    ("clinic", "clinic"),
    ("medical practices", "medical practice"),
    ("medical practice", "medical practice"),
    ("operations teams", "operations team"),
    ("operations team", "operations team"),
    ("finance teams", "finance team"),
    ("finance team", "finance team"),
    ("compliance teams", "compliance team"),
    ("compliance team", "compliance team"),
    ("banques privées", "private bank"),
    ("bourse en ligne", "online brokerage"),
    ("sociétés de gestion", "asset manager"),
    ("épargne retraite et salariale", "employee savings provider"),
    ("établissement de santé", "healthcare provider"),
    ("établissements de santé", "healthcare provider"),
    ("etablissement de santé", "healthcare provider"),
    ("etablissements de santé", "healthcare provider"),
    ("médico-social", "healthcare provider"),
    ("medico-social", "healthcare provider"),
    ("structure sanitaire", "healthcare provider"),
    ("structures sanitaires", "healthcare provider"),
)
NOISY_CUSTOMER_TERMS = (
    "award",
    "avatar",
    "barbu",
    "blonde",
    "chignon",
    "diagramme",
    "employee",
    "employees",
    "employé",
    "employés",
    "image",
    "image représentant",
    "image representing",
    "personnage",
    "portrait",
    "roux",
    "schema",
    "schéma",
    "provider",
    "software",
    "solution",
    "solutions",
    "platform",
    "portal",
    "consultancy",
    "consulting",
    "architecture",
    "process redesign",
    "onboarding",
    "registration",
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
    "utilisateur",
    "utilisatrice",
    "user",
)
NOISY_CUSTOMER_PREFIXES = (
    "how ",
    "image ",
    "illustration ",
    "why ",
    "our ",
    "with ",
    "secure ",
    "redeveloping ",
)
NOISY_NAMED_ACCOUNT_TERMS = (
    *NOISY_CUSTOMER_TERMS,
    "alumni",
    "association",
    "bde",
    "community",
    "data protection",
    "privacy policy",
    "gdpr",
    "investor",
    "investors",
    "portfolio",
    "photo",
    "team",
    "équipe",
    "schema",
    "schéma",
    "student",
    "students",
)
NOISY_EXPANSION_CAPABILITY_TERMS = (
    "about",
    "à propos",
    "accompagnée de toute documentation utile",
    "candidatdésigne",
    "ce procédé de chiffrement",
    "clientdésigne",
    "compte valide",
    "comptedésigne",
    "data protection",
    "contact",
    "contact us",
    "date de la notification",
    "désigne",
    "des dirigeants",
    "devenir un acteur",
    "documentation utile",
    "espace presse",
    "faq",
    "press",
    "procédé de chiffrement",
    "politique de protection des données",
    "policy",
    "policies",
    "privacy",
    "protection des données",
    "souscrivant",
    "security",
    "sécurité des serveurs",
    "security of servers",
    "server security",
    "consultant",
    "consultants",
    "cookie",
    "cookies",
    "conseil juridique",
    "formation",
    "mentions légales",
    "management de la sécurité",
    "recrutement formation",
    "reportings des remplacement",
    "secteur rh",
    "conditions générales",
)
NOISY_SOURCE_CAPABILITY_TERMS = (
    "on recrute",
    "join us",
    "we're hiring",
    "we are hiring",
    "career",
    "careers",
)
NOISY_ADJACENT_CUSTOMER_SEGMENT_TERMS = (
    "administrator",
    "administrators",
    "caregiver",
    "caregivers",
    "clinician",
    "clinicians",
    "doctor",
    "doctors",
    "employee",
    "employees",
    "nurse",
    "nurses",
    "paramedical",
    "paramedic",
    "paramedics",
    "physician",
    "physicians",
    "professional",
    "professionals",
    "staff",
    "team",
    "teams",
    "talent",
    "worker",
    "workers",
)
NOISY_NAMED_ACCOUNT_URL_TOKENS = (
    "/invest",
    "/investor",
    "/portfolio/",
    "/private-equity",
    "/ventures",
)
NOISY_NAMED_ACCOUNT_CONTEXT_TERMS = (
    "funding",
    "investor",
    "investors",
    "portfolio company",
    "portfolio partner",
    "private equity",
    "series a",
    "series b",
    "student",
)
NOISY_COMPARATOR_PAGE_URL_TOKENS = (
    "/a-propos",
    "/about",
    "/abonnement",
    "/cgu",
    "/conditions",
    "/contact",
    "/cookies",
    "/faq",
    "/inscription",
    "/login",
    "/mentions-legales",
    "/parrainage",
    "/partenariat",
    "/politique",
    "/privacy",
    "/signup",
)
NOISY_COMPARATOR_PAGE_TITLE_TERMS = (
    "abonnement",
    "conditions générales",
    "contact",
    "cookies",
    "faq",
    "inscription",
    "mentions légales",
    "partenariat",
    "politique de confidentialité",
    "privacy",
    "qui sommes nous",
)
INSTITUTION_SIGNAL_TOKENS = (
    "ap-",
    "assistance publique",
    "bank",
    "banque",
    "capital",
    "care",
    "centre",
    "chu",
    "chu-",
    "chuv",
    "clinic",
    "clinique",
    "clinics",
    "foundation",
    "groupe",
    "group",
    "health",
    "healthcare",
    "hospital",
    "hopital",
    "hôpital",
    "institute",
    "insurance",
    "labs",
    "ministry",
    "nhs",
    "partners",
    "pharma",
    "pharmacie",
    "systems",
    "university",
    "ventures",
)


def _slugify(value: str, *, max_len: int = 48) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return normalized[:max_len] or "item"


def _domain_brand_name(url: Any) -> str:
    domain = normalize_domain(url)
    if not domain:
        return ""
    parts = domain.split(".")
    base = parts[-2] if len(parts) >= 2 else parts[0]
    tokens = [token for token in re.split(r"[-_]+", base) if token]
    if not tokens:
        return domain
    if len(tokens) == 1 and len(tokens[0]) <= 4:
        return tokens[0].upper()
    return " ".join(token.upper() if len(token) <= 4 else token.capitalize() for token in tokens)


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "||".join([prefix, *[str(part or "").strip().lower() for part in parts]])
    return f"{prefix}_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _normalize_string_list(values: Any, *, max_items: int = 20, max_len: int = 180) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    iterable = values if isinstance(values, list) else [values]
    for item in iterable:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        results.append(text[:max_len])
        if len(results) >= max_items:
            break
    return results


def _context_pack_version(context_pack_json: Any) -> Optional[str]:
    if not isinstance(context_pack_json, dict):
        return None
    version = str(context_pack_json.get("version") or "").strip()
    return version or None


def _extract_page_headings(page: dict[str, Any]) -> list[str]:
    headings: list[str] = []
    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        if str(block.get("type") or "").strip().lower() != "heading":
            continue
        content = str(block.get("content") or "").strip()
        if content and content not in headings:
            headings.append(content[:200])
    return headings[:10]


def _safe_phrase(value: Any, *, max_len: int = 80) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" ,.;:-")
    return text[:max_len]


def _split_list_block_items(value: Any, *, max_items: int = 12) -> list[str]:
    items: list[str] = []
    for raw in str(value or "").splitlines():
        text = re.sub(r"^\s*[-*•]\s*", "", raw).strip()
        if not text:
            continue
        items.append(text)
        if len(items) >= max_items:
            break
    return items


def _block_heading_is_generic(value: Any) -> bool:
    lowered = _safe_phrase(value, max_len=120).lower()
    return lowered in GENERIC_CAPABILITY_HEADINGS or lowered in GENERIC_SUPPORT_HEADINGS


def _capability_phrase_from_text(value: Any) -> str:
    text = _safe_phrase(value, max_len=240)
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"([a-zà-ÿ])([A-Z]{2,})", r"\1 \2", text)
    text = re.sub(r"\bPATIO\s+(?:PMS|OMS)\b.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\bPATIO\b.*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s*[:;].*$", "", text).strip()
    text = re.sub(r"\s*\([^)]*\)\s*$", "", text).strip()
    text = re.sub(r"^(?:our|we|this|these)\s+", "", text, flags=re.IGNORECASE)
    if " - " in text:
        candidate_segments: list[tuple[int, str]] = []
        for raw_segment in text.split(" - "):
            segment = _safe_phrase(raw_segment, max_len=120)
            if not segment:
                continue
            segment = re.sub(r"^(?:our|we|this|these)\s+", "", segment, flags=re.IGNORECASE).strip()
            segment_parts = segment.split()
            while segment_parts and segment_parts[-1].lower() in TRAILING_CONNECTOR_WORDS:
                segment_parts.pop()
            segment = " ".join(segment_parts).strip()
            if not segment:
                continue
            lowered_segment = segment.lower()
            word_count = _phrase_word_count(segment)
            score = 0
            if 2 <= word_count <= 8:
                score += 2
            elif word_count == 1 and not segment.isupper():
                score -= 2
            if any(keyword in lowered_segment for keyword in CAPABILITY_QUALITY_KEYWORDS):
                score += 2
            if any(
                token in lowered_segment
                for token in (
                    "gestion",
                    "gérer",
                    "recrut",
                    "remplacement",
                    "planning",
                    "contrat",
                    "reporting",
                    "analytics",
                    "analyse",
                    "pilot",
                    "piloter",
                    "cré",
                    "vivier",
                    "pool",
                    "staff",
                )
            ):
                score += 1
            candidate_segments.append((score, segment))
        if candidate_segments:
            candidate_segments.sort(key=lambda item: (item[0], _phrase_word_count(item[1])), reverse=True)
            if candidate_segments[0][0] > 0:
                text = candidate_segments[0][1]
            else:
                return ""
    parts = text.split()
    if len(parts) > 6 and "," in text:
        text = text.split(",", 1)[0].strip()
    parts = text.split()
    if len(parts) > 8:
        keep = 8
        trailing_acronym = next(
            (idx for idx, token in enumerate(parts[:10]) if token.upper() == token and len(token) <= 5),
            None,
        )
        if trailing_acronym is not None:
            keep = min(max(keep, trailing_acronym + 1), len(parts))
        text = " ".join(parts[:keep]).strip()
    trimmed_parts = text.split()
    while trimmed_parts and trimmed_parts[-1].lower() in TRAILING_CONNECTOR_WORDS:
        trimmed_parts.pop()
    text = " ".join(trimmed_parts).strip()
    return text[:88]


def _normalize_phrase_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9+/ ]+", " ", str(value or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    if text.endswith("s") and len(text) > 4 and "/" not in text:
        text = text[:-1]
    return text


def _compact_phrase(value: Any, *, max_words: int = 7, max_len: int = 80) -> str:
    text = _safe_phrase(value, max_len=240)
    if not text:
        return ""
    text = re.sub(
        r"^(?:our|their|the|a|an)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"^(?:we offer|we provide|we help|platform for|solution for|software for)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    parts = text.split()
    text = " ".join(parts[:max_words]).strip()
    return text[:max_len]


def _is_job_page_url(url: Any) -> bool:
    lowered = str(url or "").lower()
    return any(token in lowered for token in ("/careers", "/career", "/jobs", "/job-", "/job/", "/apply"))


def _job_page_is_relevant(page: dict[str, Any]) -> bool:
    combined_parts = [
        str(page.get("title") or ""),
        str(page.get("raw_content") or "")[:5000],
        " ".join(_extract_page_headings(page)),
    ]
    combined = " ".join(combined_parts).lower()
    return any(token in combined for token in JOB_PAGE_KEYWORDS)


def _is_plausible_named_customer(
    name: Any,
    *,
    evidence_type: Any = None,
    context: Any = None,
    source_url: Any = None,
) -> bool:
    text = _safe_phrase(name, max_len=120)
    if not text:
        return False
    if not any(ch.isalpha() for ch in text):
        return False

    lowered = text.lower()
    words = [word for word in re.split(r"\s+", text) if word]
    if not words:
        return False
    if any(lowered.startswith(prefix) for prefix in NOISY_CUSTOMER_PREFIXES):
        return False
    if any(token in lowered for token in NOISY_CUSTOMER_TERMS):
        return False
    if re.search(r"[.!?]", text):
        return False

    evidence_kind = str(evidence_type or "").strip().lower()
    context_lower = str(context or "").strip().lower()
    source_lower = str(source_url or "").strip().lower()

    if evidence_kind in {"logo_alt", "aria_label"}:
        if len(words) > 6:
            return False
        if (
            any(token in lowered for token in ("private bank", "wealth management firm", "management solution"))
            and not any(suffix in lowered for suffix in ("bank", "capital", "partners", "group", "holdings"))
        ):
            return False
        if 2 <= len(words) <= 3:
            if all(re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ'-]+", word) for word in words):
                if not any(token in lowered for token in INSTITUTION_SIGNAL_TOKENS):
                    return False
        if not any(ch.isupper() or ch.isdigit() for ch in text):
            return False
    else:
        if len(words) > 10:
            return False

    if source_lower and any(token in source_lower for token in ("/awards", "/insights", "/capabilities")):
        if evidence_kind in {"logo_alt", "aria_label"} and len(words) > 4:
            return False
    if context_lower and "trusted by" not in context_lower and evidence_kind in {"logo_alt", "aria_label"}:
        if any(token in context_lower for token in ("latest", "results", "recent", "ability to", "designed to help")):
            return False
    return True


def _is_plausible_named_account_anchor(value: Any) -> bool:
    text = _safe_phrase(value, max_len=120)
    if not text:
        return False
    lowered = text.lower()
    if any(token in lowered for token in NOISY_NAMED_ACCOUNT_TERMS):
        return False
    if any(lowered.startswith(prefix) for prefix in NOISY_CUSTOMER_PREFIXES):
        return False
    if re.search(r"[.!?]", text):
        return False
    words = [word for word in re.split(r"\s+", text) if word]
    if 2 <= len(words) <= 3:
        if all(re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ'-]+", word) for word in words):
            if not any(token in lowered for token in INSTITUTION_SIGNAL_TOKENS):
                return False
    return True


def _is_plausible_named_account_anchor_item(item: Any) -> bool:
    if not isinstance(item, dict):
        return False
    label = _safe_phrase(item.get("label") or item.get("name"), max_len=120)
    if not _is_plausible_named_account_anchor(label):
        return False
    why_lower = str(item.get("why_it_matters") or "").strip().lower()
    source_names_lower = " ".join(_normalize_string_list(item.get("source_entity_names"), max_items=6, max_len=120)).lower()
    combined = " ".join(part for part in (why_lower, source_names_lower) if part).strip()
    if any(token in combined for token in NOISY_NAMED_ACCOUNT_CONTEXT_TERMS):
        return False
    for url in _normalize_string_list(item.get("evidence_urls"), max_items=6, max_len=500):
        lowered = normalize_url(url) or ""
        if any(token in lowered for token in NOISY_NAMED_ACCOUNT_URL_TOKENS):
            return False
    return True


def _is_plausible_adjacent_customer_segment(value: Any) -> bool:
    text = _safe_phrase(value, max_len=140)
    if not text:
        return False
    lowered = text.lower()
    normalized = _normalize_phrase_key(text)
    canonical_customer_keys = {
        _normalize_phrase_key(pattern)
        for pattern, canonical in CUSTOMER_ARCHETYPE_PATTERNS
        for pattern in (pattern, canonical)
        if _normalize_phrase_key(pattern)
    }
    if normalized in canonical_customer_keys:
        return True
    if any(token in lowered for token in NOISY_ADJACENT_CUSTOMER_SEGMENT_TERMS):
        return False
    return True


def _is_plausible_expansion_capability(value: Any) -> bool:
    text = _safe_phrase(value, max_len=140)
    if not text:
        return False
    lowered = text.lower()
    if "?" in text:
        return False
    if lowered.startswith("vous êtes "):
        return False
    if any(token in lowered for token in NOISY_EXPANSION_CAPABILITY_TERMS):
        return False
    return True


def _is_plausible_comparator_adjacent_capability(value: Any) -> bool:
    text = _safe_phrase(value, max_len=140)
    if not _is_plausible_expansion_capability(text):
        return False
    lowered = text.lower()
    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) > 6:
        return False
    if any(lowered.startswith(prefix) for prefix in ("ce ", "cette ", "date ", "des ", "devenir ")):
        return False
    if lowered.startswith("clientdésigne") or lowered.startswith("comptedésigne") or lowered.startswith("compte valide"):
        return False
    return True


def _is_high_quality_adjacent_capability_candidate(value: Any) -> bool:
    text = _safe_phrase(value, max_len=140)
    if not _is_plausible_comparator_adjacent_capability(text):
        return False
    lowered = text.lower()
    words = [word for word in re.split(r"\s+", text) if word]
    if any(
        lowered.startswith(prefix)
        for prefix in (
            "en ",
            "la ",
            "le ",
            "les ",
            "historique ",
            "illustration ",
            "je ",
            "la team ",
            "ne ",
        )
    ):
        return False
    if any(
        token in lowered
        for token in (
            "analytics",
            "contract",
            "contrat",
            "dépen",
            "expense",
            "gestion",
            "integration",
            "management",
            "planning",
            "platform",
            "plateforme",
            "recruit",
            "recrut",
            "renfort",
            "remplaç",
            "remplacement",
            "reporting",
            "staffing",
            "vivier",
            "voting",
            "workflow",
        )
    ):
        return True
    return False


def _is_noisy_comparator_page(*, url: str, title: str) -> bool:
    lowered_url = normalize_url(url).lower()
    lowered_title = _safe_phrase(title, max_len=160).lower()
    return any(token in lowered_url for token in NOISY_COMPARATOR_PAGE_URL_TOKENS) or any(
        token in lowered_title for token in NOISY_COMPARATOR_PAGE_TITLE_TERMS
    )


def _comparator_page_title_capability_phrase(title: Any) -> str:
    text = _safe_phrase(title, max_len=160)
    if not text:
        return ""
    core = re.split(r"\s[-–|:]\s", text, maxsplit=1)[0].strip()
    core = re.sub(r"^(?:solution de|logiciel de)\s+", "", core, flags=re.IGNORECASE).strip()
    core = re.sub(r"^trouvez\s+(?:des|de)\s+", "", core, flags=re.IGNORECASE).strip()
    core = re.sub(r"^vidéo\s+", "", core, flags=re.IGNORECASE).strip()
    if not core:
        return ""
    if core.lower().startswith(("je ", "nous ", "vous ", "renforcez ", "parraine", "partenariat ")):
        return ""
    if "remplacement" in core.lower() and "urgence" in core.lower():
        return "Gestion des remplacements en urgence"
    if "renfort" in core.lower() and "soignant" in core.lower():
        return "Renfort soignant"
    compact = _compact_phrase(core, max_words=8, max_len=120)
    if compact:
        compact = compact[:1].upper() + compact[1:]
    return compact


def _derive_comparator_page_capability_nodes(
    *,
    scoped_sites: list[dict[str, Any]],
    comparator_name: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for site in scoped_sites:
        for page in site.get("pages") or []:
            if not isinstance(page, dict):
                continue
            url = normalize_url(page.get("url") or site.get("website") or site.get("url") or "")
            title = str(page.get("title") or "").strip()
            if not url or not title or _is_noisy_comparator_page(url=url, title=title):
                continue
            phrase = _comparator_page_title_capability_phrase(title)
            if not phrase or not _is_high_quality_adjacent_capability_candidate(phrase):
                continue
            key = _normalize_phrase_key(phrase)
            if not key or key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "id": _stable_id("taxonomy", comparator_name, "capability", phrase),
                    "layer": "capability",
                    "phrase": phrase,
                    "aliases": [],
                    "confidence": 0.66,
                    "evidence_ids": [],
                    "source_url": url,
                    "scope_status": "in_scope",
                }
            )
    return candidates[:8]


def _is_plausible_comparator_name(value: Any) -> bool:
    text = _safe_phrase(value, max_len=140)
    if not text:
        return False
    lowered = text.lower()
    words = [word for word in re.split(r"\s+", text) if word]
    if len(words) > 6:
        return False
    if any(token in lowered for token in ("gestion", "mission", "missions", "replacement", "remplacements", "urgent", "urgence")):
        if not any(token in lowered for token in INSTITUTION_SIGNAL_TOKENS):
            return False
    if text.endswith(("!", ".", "?")):
        return False
    return True


def _keyword_phrases_from_text(text: Any) -> list[str]:
    lowered = str(text or "").lower()
    if not lowered:
        return []

    phrases: list[str] = []
    seen: set[str] = set()

    def _append(value: str) -> None:
        phrase = _safe_phrase(value, max_len=80)
        key = _normalize_phrase_key(phrase)
        if not phrase or not key or key in seen:
            return
        seen.add(key)
        phrases.append(phrase)

    for keyword in WORKFLOW_KEYWORDS:
        if keyword in lowered:
            _append(keyword)
    for pattern, canonical in CUSTOMER_ARCHETYPE_PATTERNS:
        if pattern in lowered:
            _append(canonical)
    for keyword in CAPABILITY_KEYWORDS:
        if keyword in lowered:
            _append(keyword)
    for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS:
        if keyword in lowered:
            _append(keyword)
    return phrases


def _taxonomy_signals_from_page_blocks(page: dict[str, Any]) -> list[tuple[str, str]]:
    page_type = str(page.get("page_type") or "").strip().lower()
    if page_type not in {"product", "solutions", "services", "docs"}:
        return []

    signals: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    last_heading = ""

    def add(kind: str, value: Any) -> None:
        phrase = _safe_phrase(value, max_len=180)
        if not phrase:
            return
        key = (kind, _normalize_phrase_key(phrase))
        if key in seen:
            return
        seen.add(key)
        signals.append((kind, phrase))

    for block in page.get("blocks") or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").strip().lower()
        content = _safe_phrase(block.get("content"), max_len=320)
        if not content:
            continue

        if block_type == "heading":
            last_heading = content
            lowered = content.lower()
            if _block_heading_is_generic(content):
                if "architecture" in lowered or "intégration" in lowered or "integration" in lowered:
                    add("service", content)
                continue
            if any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
                add("service", content)
            elif any(keyword in lowered for keyword in WORKFLOW_KEYWORDS):
                add("workflow", content)
            else:
                compact = _capability_phrase_from_text(content)
                if compact:
                    add("capability", compact)
            continue

        if block_type == "list":
            for item in _split_list_block_items(content):
                compact = _capability_phrase_from_text(item)
                if not compact:
                    continue
                lowered = compact.lower()
                if any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
                    add("service", compact)
                else:
                    add("capability", compact)
            continue

        if block_type == "paragraph" and last_heading:
            lowered_heading = last_heading.lower()
            if "architecture" in lowered_heading or "intégration" in lowered_heading or "integration" in lowered_heading:
                compact = _compact_phrase(content, max_words=7, max_len=88)
                if compact:
                    add("service", compact)

    return signals


def build_context_pack_v2(context_pack_json: Any) -> dict[str, Any]:
    if not isinstance(context_pack_json, dict):
        return {
            "version": "v2",
            "generated_at": None,
            "urls_crawled": [],
            "sites": [],
            "evidence_items": [],
            "named_customers": [],
            "integrations": [],
            "partners": [],
            "extracted_raw_phrases": [],
            "crawl_coverage": {
                "total_sites": 0,
                "total_pages": 0,
                "page_type_counts": {},
                "pages_with_signals": 0,
                "pages_with_customer_evidence": 0,
                "career_pages_selected": 0,
            },
        }
    version = _context_pack_version(context_pack_json)
    if version == "v2":
        sites = [site for site in (context_pack_json.get("sites") or []) if isinstance(site, dict)]
        has_raw_site_inputs = any(
            site.get("pages") or site.get("signals") or site.get("customer_evidence")
            for site in sites
        )
        if not has_raw_site_inputs:
            return context_pack_json

    sites_out: list[dict[str, Any]] = []
    aggregate_evidence: dict[str, dict[str, Any]] = {}
    aggregate_customers: dict[str, dict[str, Any]] = {}
    aggregate_integrations: dict[str, dict[str, Any]] = {}
    aggregate_phrases: set[str] = set()
    page_type_counts: dict[str, int] = {}
    pages_with_signals = 0
    pages_with_customer_evidence = 0
    total_pages = 0
    career_pages_selected = 0

    for site in context_pack_json.get("sites") or []:
        if not isinstance(site, dict):
            continue
        pages = [page for page in (site.get("pages") or []) if isinstance(page, dict)]
        evidence_items: dict[str, dict[str, Any]] = {}
        named_customers: dict[str, dict[str, Any]] = {}
        integrations: dict[str, dict[str, Any]] = {}
        selected_pages: list[dict[str, Any]] = []
        raw_phrases: set[str] = set()
        site_page_type_counts: dict[str, int] = {}

        def _register_evidence(
            *,
            source_url: Any,
            kind: str,
            text: Any,
            snippet: Any = "",
            page_type: Any = None,
            page_title: Any = None,
            confidence: float = 0.75,
        ) -> Optional[str]:
            url = normalize_url(source_url)
            phrase = _safe_phrase(text, max_len=180)
            if not url or not phrase:
                return None
            evidence_id = _stable_id("evidence", kind, url, phrase)
            payload = {
                "id": evidence_id,
                "kind": kind,
                "text": phrase,
                "snippet": _safe_phrase(snippet, max_len=320),
                "url": url,
                "page_type": str(page_type or "").strip() or None,
                "page_title": _safe_phrase(page_title, max_len=180) or None,
                "captured_at": context_pack_json.get("generated_at"),
                "confidence": _clamp_confidence(confidence, default=confidence),
            }
            evidence_items[evidence_id] = payload
            aggregate_evidence.setdefault(evidence_id, payload)
            return evidence_id

        site_summary = _safe_phrase(site.get("summary"), max_len=240)
        if site_summary:
            for phrase in _keyword_phrases_from_text(site_summary):
                raw_phrases.add(phrase)
                aggregate_phrases.add(phrase)

        for signal in site.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            evidence_id = _register_evidence(
                source_url=signal.get("source_url"),
                kind=f"site_signal:{signal.get('type') or 'unknown'}",
                text=signal.get("value"),
                snippet=signal.get("snippet"),
                page_type="site_summary",
                page_title=site.get("company_name") or site.get("website"),
                confidence=0.78,
            )
            phrase = _compact_phrase(signal.get("value"))
            if phrase:
                raw_phrases.add(phrase)
                aggregate_phrases.add(phrase)
            if str(signal.get("type") or "").strip().lower() == "integration":
                key = _normalize_phrase_key(signal.get("value"))
                if key:
                    integrations[key] = {
                        "name": _safe_phrase(signal.get("value")),
                        "source_url": normalize_url(signal.get("source_url")),
                        "evidence_id": evidence_id,
                    }
                    aggregate_integrations.setdefault(key, integrations[key])

        for customer in site.get("customer_evidence") or []:
            if not isinstance(customer, dict):
                continue
            if not _is_plausible_named_customer(
                customer.get("name"),
                evidence_type=customer.get("evidence_type"),
                context=customer.get("context"),
                source_url=customer.get("source_url"),
            ):
                continue
            evidence_id = _register_evidence(
                source_url=customer.get("source_url"),
                kind=f"site_customer:{customer.get('evidence_type') or 'unknown'}",
                text=customer.get("name"),
                snippet=customer.get("context"),
                page_type="customers",
                page_title=site.get("company_name") or site.get("website"),
                confidence=0.82,
            )
            name = _safe_phrase(customer.get("name"), max_len=120)
            if name:
                key = _normalize_phrase_key(name)
                named_customers[key] = {
                    "name": name,
                    "source_url": normalize_url(customer.get("source_url")),
                    "context": _safe_phrase(customer.get("context"), max_len=240),
                    "evidence_type": str(customer.get("evidence_type") or "").strip() or "customer",
                    "evidence_id": evidence_id,
                }
                aggregate_customers.setdefault(key, named_customers[key])
                raw_phrases.add(name)
                aggregate_phrases.add(name)

        for page in pages:
            page_type = str(page.get("page_type") or "other").strip() or "other"
            headings = _extract_page_headings(page)
            page_customers = [
                customer
                for customer in (page.get("customer_evidence") or [])
                if isinstance(customer, dict)
                and _is_plausible_named_customer(
                    customer.get("name"),
                    evidence_type=customer.get("evidence_type"),
                    context=customer.get("context"),
                    source_url=customer.get("source_url") or page.get("url"),
                )
            ]
            has_signals = bool(page.get("signals"))
            has_customers = bool(page_customers)
            site_page_type_counts[page_type] = site_page_type_counts.get(page_type, 0) + 1
            page_type_counts[page_type] = page_type_counts.get(page_type, 0) + 1
            total_pages += 1
            if has_signals:
                pages_with_signals += 1
            if has_customers:
                pages_with_customer_evidence += 1
            if _is_job_page_url(page.get("url")):
                if _job_page_is_relevant(page):
                    career_pages_selected += 1
                else:
                    continue

            selected_pages.append(
                {
                    "url": normalize_url(page.get("url")),
                    "title": _safe_phrase(page.get("title"), max_len=180),
                    "page_type": page_type,
                    "headings": headings,
                    "has_signals": has_signals,
                    "has_customer_evidence": has_customers,
                }
            )

            for heading in headings:
                phrase = _compact_phrase(heading)
                if phrase:
                    raw_phrases.add(phrase)
                    aggregate_phrases.add(phrase)
            page_title = _compact_phrase(page.get("title"), max_words=8, max_len=80)
            if page_title:
                raw_phrases.add(page_title)
                aggregate_phrases.add(page_title)
            for phrase in _keyword_phrases_from_text(page.get("raw_content")):
                raw_phrases.add(phrase)
                aggregate_phrases.add(phrase)

            for signal_kind, signal_value in _taxonomy_signals_from_page_blocks(page):
                evidence_id = _register_evidence(
                    source_url=page.get("url"),
                    kind=f"page_signal:{signal_kind}",
                    text=signal_value,
                    snippet=f"Structured {signal_kind} extracted from rendered page blocks",
                    page_type=page_type,
                    page_title=page.get("title"),
                    confidence=0.88 if signal_kind == "capability" else 0.82,
                )
                compact = _compact_phrase(signal_value, max_words=7, max_len=88)
                if compact:
                    raw_phrases.add(compact)
                    aggregate_phrases.add(compact)

            for signal in page.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                evidence_id = _register_evidence(
                    source_url=signal.get("source_url") or page.get("url"),
                    kind=f"page_signal:{signal.get('type') or 'unknown'}",
                    text=signal.get("value"),
                    snippet=signal.get("snippet"),
                    page_type=page_type,
                    page_title=page.get("title"),
                    confidence=0.8,
                )
                phrase = _compact_phrase(signal.get("value"))
                if phrase:
                    raw_phrases.add(phrase)
                    aggregate_phrases.add(phrase)
                if str(signal.get("type") or "").strip().lower() == "integration":
                    key = _normalize_phrase_key(signal.get("value"))
                    if key:
                        integrations[key] = {
                            "name": _safe_phrase(signal.get("value")),
                            "source_url": normalize_url(signal.get("source_url") or page.get("url")),
                            "evidence_id": evidence_id,
                        }
                        aggregate_integrations.setdefault(key, integrations[key])

            for customer in page_customers:
                evidence_id = _register_evidence(
                    source_url=customer.get("source_url") or page.get("url"),
                    kind=f"page_customer:{customer.get('evidence_type') or 'unknown'}",
                    text=customer.get("name"),
                    snippet=customer.get("context"),
                    page_type=page_type,
                    page_title=page.get("title"),
                    confidence=0.84,
                )
                name = _safe_phrase(customer.get("name"), max_len=120)
                if name:
                    key = _normalize_phrase_key(name)
                    named_customers[key] = {
                        "name": name,
                        "source_url": normalize_url(customer.get("source_url") or page.get("url")),
                        "context": _safe_phrase(customer.get("context"), max_len=240),
                        "evidence_type": str(customer.get("evidence_type") or "").strip() or "customer",
                        "evidence_id": evidence_id,
                    }
                    aggregate_customers.setdefault(key, named_customers[key])
                    raw_phrases.add(name)
                    aggregate_phrases.add(name)

        sites_out.append(
            {
                **site,
                "selected_pages": selected_pages,
                "evidence_items": list(evidence_items.values()),
                "named_customers": list(named_customers.values()),
                "integrations": list(integrations.values()),
                "partners": list(integrations.values()),
                "extracted_raw_phrases": sorted(raw_phrases)[:80],
                "crawl_coverage": {
                    "total_pages": len(pages),
                    "page_type_counts": site_page_type_counts,
                    "selected_pages": len(selected_pages),
                    "pages_with_signals": len([page for page in pages if page.get("signals")]),
                    "pages_with_customer_evidence": len(
                        [
                            page
                            for page in pages
                            if any(
                                isinstance(customer, dict)
                                and _is_plausible_named_customer(
                                    customer.get("name"),
                                    evidence_type=customer.get("evidence_type"),
                                    context=customer.get("context"),
                                    source_url=customer.get("source_url") or page.get("url"),
                                )
                                for customer in (page.get("customer_evidence") or [])
                            )
                        ]
                    ),
                    "career_pages_selected": len(
                        [page for page in selected_pages if str(page.get("page_type") or "") == "careers"]
                    ),
                },
            }
        )

    return {
        **context_pack_json,
        "version": "v2",
        "sites": sites_out,
        "evidence_items": list(aggregate_evidence.values()),
        "named_customers": list(aggregate_customers.values()),
        "integrations": list(aggregate_integrations.values()),
        "partners": list(aggregate_integrations.values()),
        "extracted_raw_phrases": sorted(aggregate_phrases)[:160],
        "crawl_coverage": {
            "total_sites": len(sites_out),
            "total_pages": total_pages,
            "page_type_counts": page_type_counts,
            "pages_with_signals": pages_with_signals,
            "pages_with_customer_evidence": pages_with_customer_evidence,
            "career_pages_selected": career_pages_selected,
        },
    }


def _domain_scoped_context_pack(context_pack_v2: dict[str, Any], *, domain: str) -> dict[str, Any]:
    domain = normalize_domain(domain)
    if not domain or not isinstance(context_pack_v2, dict):
        return context_pack_v2

    selected_sites: list[dict[str, Any]] = []
    for site in context_pack_v2.get("sites") or []:
        if not isinstance(site, dict):
            continue
        site_domain = normalize_domain(site.get("website") or site.get("url"))
        if not site_domain:
            continue
        if site_domain == domain or site_domain.endswith(f".{domain}") or domain.endswith(f".{site_domain}"):
            selected_sites.append(site)

    if not selected_sites:
        return context_pack_v2

    has_site_level_artifacts = any(
        (site.get("evidence_items") or site.get("named_customers") or site.get("integrations") or site.get("extracted_raw_phrases"))
        for site in selected_sites
        if isinstance(site, dict)
    )
    if not has_site_level_artifacts:
        return {
            **context_pack_v2,
            "sites": selected_sites,
        }

    evidence_items: dict[str, dict[str, Any]] = {}
    named_customers: dict[str, dict[str, Any]] = {}
    integrations: dict[str, dict[str, Any]] = {}
    raw_phrases: set[str] = set()
    page_type_counts: dict[str, int] = {}
    coverage = {
        "total_sites": len(selected_sites),
        "total_pages": 0,
        "page_type_counts": page_type_counts,
        "pages_with_signals": 0,
        "pages_with_customer_evidence": 0,
        "career_pages_selected": 0,
    }

    for site in selected_sites:
        for item in site.get("evidence_items") or []:
            if isinstance(item, dict) and item.get("id"):
                evidence_items[str(item["id"])] = item
        for item in site.get("named_customers") or []:
            if isinstance(item, dict) and item.get("name"):
                named_customers[_normalize_phrase_key(item.get("name"))] = item
        for item in site.get("integrations") or []:
            if isinstance(item, dict) and item.get("name"):
                integrations[_normalize_phrase_key(item.get("name"))] = item
        for phrase in site.get("extracted_raw_phrases") or []:
            compact = _safe_phrase(phrase, max_len=80)
            if compact:
                raw_phrases.add(compact)
        site_coverage = site.get("crawl_coverage") if isinstance(site.get("crawl_coverage"), dict) else {}
        coverage["total_pages"] += int(site_coverage.get("total_pages") or 0)
        coverage["pages_with_signals"] += int(site_coverage.get("pages_with_signals") or 0)
        coverage["pages_with_customer_evidence"] += int(site_coverage.get("pages_with_customer_evidence") or 0)
        coverage["career_pages_selected"] += int(site_coverage.get("career_pages_selected") or 0)
        for page_type, count in (site_coverage.get("page_type_counts") or {}).items():
            key = str(page_type or "").strip() or "other"
            page_type_counts[key] = page_type_counts.get(key, 0) + int(count or 0)

    return {
        **context_pack_v2,
        "sites": selected_sites,
        "evidence_items": list(evidence_items.values()),
        "named_customers": list(named_customers.values()),
        "integrations": list(integrations.values()),
        "partners": list(integrations.values()),
        "extracted_raw_phrases": sorted(raw_phrases)[:160],
        "crawl_coverage": coverage,
    }


def _source_scoped_context_pack(context_pack_v2: dict[str, Any], *, buyer_url: Any) -> dict[str, Any]:
    buyer_domain = normalize_domain(buyer_url)
    if not buyer_domain:
        return context_pack_v2
    return _domain_scoped_context_pack(context_pack_v2, domain=buyer_domain)


def _phrase_case(value: str) -> str:
    if not value:
        return ""
    if value.isupper() or "/" in value:
        return value
    return value[0].upper() + value[1:]


def _phrase_word_count(value: Any) -> int:
    return len([part for part in re.split(r"\s+", str(value or "").strip()) if part])


def _canonicalize_taxonomy_phrase(layer: str, value: Any) -> str:
    phrase = _safe_phrase(value, max_len=96)
    if not phrase:
        return ""

    lowered = phrase.lower()
    if layer == "customer_archetype":
        for pattern, canonical in CUSTOMER_ARCHETYPE_PATTERNS:
            if pattern == lowered:
                return _phrase_case(canonical)

    if layer == "capability":
        if " - " in phrase:
            prefix, suffix = [segment.strip() for segment in phrase.split(" - ", 1)]
            if suffix and any(token in suffix.lower() for token in ("platform", "plateforme", "management", "oms", "pms", "lifecycle")):
                phrase = suffix
                lowered = phrase.lower()
        for pattern, canonical in CAPABILITY_CANONICAL_PATTERNS:
            if lowered == pattern:
                return canonical

    if layer == "delivery_or_integration":
        for pattern, canonical in DELIVERY_OR_INTEGRATION_CANONICALS:
            if pattern in lowered:
                return canonical

    return _phrase_case(phrase)


def _is_valid_taxonomy_phrase(
    layer: str,
    phrase: Any,
    *,
    named_customer_keys: set[str],
    integration_keys: set[str],
) -> bool:
    normalized = _canonicalize_taxonomy_phrase(layer, phrase)
    if not normalized:
        return False

    phrase_key = _normalize_phrase_key(normalized)
    if not phrase_key:
        return False
    if phrase_key in named_customer_keys:
        return layer == "customer_archetype"
    if layer != "customer_archetype" and phrase_key in integration_keys:
        return False

    lowered = normalized.lower()
    word_count = _phrase_word_count(normalized)

    if layer == "workflow":
        if phrase_key in GENERIC_WORKFLOW_PHRASES:
            return False
        if word_count < 2:
            return False
        if any(ch.isdigit() for ch in normalized):
            return False
        if any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
            return False
    elif layer == "capability":
        if any(token in lowered for token in NOISY_SOURCE_CAPABILITY_TERMS):
            return False
        if any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
            return False
        if any(keyword in lowered for keyword in CUSTOMER_KEYWORDS):
            if not any(keyword in lowered for keyword in CAPABILITY_QUALITY_KEYWORDS):
                return False
        if " - " in normalized and not any(keyword in lowered for keyword in CAPABILITY_QUALITY_KEYWORDS):
            return False
        if word_count < 2 and "/" not in normalized and not normalized.isupper():
            return False
        if any(lowered.startswith(prefix) for prefix in SENTENCE_LIKE_CAPABILITY_PREFIXES):
            return False
        if word_count > 8 and "/" not in normalized and not normalized.isupper():
            return False
        if lowered in {"platform", "software", "solution", "solutions"}:
            return False
        if lowered.split() and lowered.split()[-1] in TRAILING_CONNECTOR_WORDS:
            return False
    elif layer == "delivery_or_integration":
        if phrase_key in integration_keys:
            return False
        if not any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
            return False

    return True


def _taxonomy_phrase_candidates(
    context_pack_v2: dict[str, Any],
    *,
    layer: str,
) -> list[dict[str, Any]]:
    if layer not in TAXONOMY_LAYERS:
        return []

    candidates: list[dict[str, Any]] = []
    evidence_items = context_pack_v2.get("evidence_items") or []
    raw_phrases = context_pack_v2.get("extracted_raw_phrases") or []
    named_customer_keys = {
        _normalize_phrase_key(item.get("name"))
        for item in (context_pack_v2.get("named_customers") or [])
        if isinstance(item, dict) and _normalize_phrase_key(item.get("name"))
    }
    integration_keys = {
        _normalize_phrase_key(item.get("name"))
        for item in (context_pack_v2.get("integrations") or [])
        if isinstance(item, dict) and _normalize_phrase_key(item.get("name"))
    }
    raw_joined = "\n".join(str(value or "") for value in raw_phrases)
    lower_blob = raw_joined.lower()

    if layer == "customer_archetype":
        for pattern, canonical in CUSTOMER_ARCHETYPE_PATTERNS:
            if pattern in lower_blob:
                evidence_ids = [
                    item.get("id")
                    for item in evidence_items
                    if isinstance(item, dict)
                    and pattern in str(item.get("text") or "").lower()
                ][:6]
                phrase = _canonicalize_taxonomy_phrase(layer, canonical)
                if not _is_valid_taxonomy_phrase(
                    layer,
                    phrase,
                    named_customer_keys=named_customer_keys,
                    integration_keys=integration_keys,
                ):
                    continue
                candidates.append(
                    {
                        "layer": layer,
                        "phrase": phrase,
                        "aliases": [pattern],
                        "confidence": 0.84 if evidence_ids else 0.68,
                        "evidence_ids": evidence_ids,
                    }
                )
        for customer in context_pack_v2.get("named_customers") or []:
            if not isinstance(customer, dict):
                continue
            context = str(customer.get("context") or "").lower()
            for pattern, canonical in CUSTOMER_ARCHETYPE_PATTERNS:
                if pattern in context:
                    phrase = _canonicalize_taxonomy_phrase(layer, canonical)
                    if not _is_valid_taxonomy_phrase(
                        layer,
                        phrase,
                        named_customer_keys=named_customer_keys,
                        integration_keys=integration_keys,
                    ):
                        continue
                    candidates.append(
                        {
                            "layer": layer,
                            "phrase": phrase,
                            "aliases": [customer.get("name") or pattern],
                            "confidence": 0.78,
                            "evidence_ids": [customer.get("evidence_id")] if customer.get("evidence_id") else [],
                        }
                    )

    if layer == "capability":
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            if "signal:capability" not in kind and "page_signal:capability" not in kind:
                continue
            phrase = _canonicalize_taxonomy_phrase(layer, _capability_phrase_from_text(item.get("text")))
            if not _is_valid_taxonomy_phrase(
                layer,
                phrase,
                named_customer_keys=named_customer_keys,
                integration_keys=integration_keys,
            ):
                continue
            candidates.append(
                {
                    "layer": layer,
                    "phrase": phrase,
                    "aliases": [item.get("text") or phrase],
                    "confidence": 0.82,
                    "evidence_ids": [item.get("id")] if item.get("id") else [],
                }
                )
        for phrase in raw_phrases:
            compact = _canonicalize_taxonomy_phrase(layer, _capability_phrase_from_text(phrase))
            if not compact:
                continue
            lowered = compact.lower()
            if any(token in lowered for token in ("api", "documentation", "infrastructure")):
                continue
            if any(
                token in lowered
                for token in ("platform", "software", "management", "analytics", "planning", "replacement", "oms", "pms", "stp")
            ):
                if not _is_valid_taxonomy_phrase(
                    layer,
                    compact,
                    named_customer_keys=named_customer_keys,
                    integration_keys=integration_keys,
                ):
                    continue
                candidates.append(
                    {
                        "layer": layer,
                        "phrase": compact,
                        "aliases": [phrase],
                        "confidence": 0.58,
                        "evidence_ids": [],
                    }
                )

    if layer == "workflow":
        for keyword in WORKFLOW_KEYWORDS:
            if keyword in lower_blob:
                evidence_ids = [
                    item.get("id")
                    for item in evidence_items
                    if isinstance(item, dict)
                    and keyword in str(item.get("text") or "").lower()
                ][:6]
                phrase = _canonicalize_taxonomy_phrase(layer, keyword)
                if not _is_valid_taxonomy_phrase(
                    layer,
                    phrase,
                    named_customer_keys=named_customer_keys,
                    integration_keys=integration_keys,
                ):
                    continue
                candidates.append(
                    {
                        "layer": layer,
                        "phrase": phrase,
                        "aliases": [keyword],
                        "confidence": 0.8 if evidence_ids else 0.64,
                        "evidence_ids": evidence_ids,
                    }
                )
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "")
            lowered = text.lower()
            kind = str(item.get("kind") or "").lower()
            if "signal:workflow" not in kind and "page_signal:workflow" not in kind:
                continue
            phrase = _canonicalize_taxonomy_phrase(layer, _compact_phrase(text, max_words=7, max_len=72))
            if not _is_valid_taxonomy_phrase(
                layer,
                phrase,
                named_customer_keys=named_customer_keys,
                integration_keys=integration_keys,
            ):
                continue
            candidates.append(
                {
                    "layer": layer,
                    "phrase": phrase,
                    "aliases": [text],
                    "confidence": 0.76,
                    "evidence_ids": [item.get("id")] if item.get("id") else [],
                }
            )

    if layer == "delivery_or_integration":
        for item in evidence_items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "").lower()
            text = str(item.get("text") or "")
            lowered = text.lower()
            if "signal:service" not in kind and "page_signal:service" not in kind:
                if not any(keyword in lowered for keyword in DELIVERY_OR_INTEGRATION_KEYWORDS):
                    continue
            phrase = _canonicalize_taxonomy_phrase(layer, _compact_phrase(text, max_words=6, max_len=72))
            if not _is_valid_taxonomy_phrase(
                layer,
                phrase,
                named_customer_keys=named_customer_keys,
                integration_keys=integration_keys,
            ):
                continue
            candidates.append(
                {
                    "layer": layer,
                    "phrase": phrase,
                    "aliases": [text],
                    "confidence": 0.78 if item.get("id") else 0.62,
                    "evidence_ids": [item.get("id")] if item.get("id") else [],
                }
            )
        for phrase in raw_phrases:
            compact = _canonicalize_taxonomy_phrase(layer, _compact_phrase(phrase, max_words=6, max_len=72))
            if not _is_valid_taxonomy_phrase(
                layer,
                compact,
                named_customer_keys=named_customer_keys,
                integration_keys=integration_keys,
            ):
                continue
            candidates.append(
                {
                    "layer": layer,
                    "phrase": compact,
                    "aliases": [phrase],
                    "confidence": 0.58,
                    "evidence_ids": [],
                }
            )

    return candidates


def normalize_taxonomy_nodes(nodes: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(nodes, list):
        nodes = []
    for item in nodes:
        if not isinstance(item, dict):
            continue
        layer = str(item.get("layer") or "").strip()
        phrase = _safe_phrase(item.get("phrase"), max_len=80)
        if layer not in TAXONOMY_LAYERS or not phrase:
            continue
        if layer == "capability" and not _is_plausible_expansion_capability(phrase):
            continue
        key = (layer, _normalize_phrase_key(phrase))
        record = grouped.get(key)
        aliases = _normalize_string_list(item.get("aliases"), max_items=12, max_len=80)
        aliases = [alias for alias in aliases if _normalize_phrase_key(alias) != key[1]]
        evidence_ids = _normalize_string_list(item.get("evidence_ids"), max_items=12, max_len=96)
        source_url = normalize_url(item.get("source_url"))
        scope_status = str(item.get("scope_status") or "in_scope").strip().lower()
        if scope_status not in TAXONOMY_SCOPE_STATUSES:
            scope_status = "in_scope"
        if record is None:
            record = {
                "id": str(item.get("id") or _stable_id("taxonomy", layer, phrase)),
                "layer": layer,
                "phrase": phrase,
                "aliases": aliases,
                "confidence": _clamp_confidence(item.get("confidence"), default=0.68),
                "evidence_ids": evidence_ids,
                "source_url": source_url,
                "scope_status": scope_status,
            }
            grouped[key] = record
            normalized.append(record)
            continue
        record["aliases"] = _normalize_string_list(record.get("aliases", []) + aliases, max_items=12, max_len=80)
        record["evidence_ids"] = _normalize_string_list(record.get("evidence_ids", []) + evidence_ids, max_items=12, max_len=96)
        record["confidence"] = _clamp_confidence(max(float(record.get("confidence") or 0.0), float(item.get("confidence") or 0.0)), default=0.68)
        if source_url:
            existing_source_url = normalize_url(record.get("source_url"))
            existing_path = urlparse(existing_source_url).path if existing_source_url else ""
            candidate_path = urlparse(source_url).path
            if not existing_source_url or len(candidate_path) > len(existing_path):
                record["source_url"] = source_url
        if record.get("scope_status") != "removed":
            record["scope_status"] = scope_status
    normalized = _suppress_redundant_taxonomy_nodes(normalized)
    layer_order = {
        "customer_archetype": 0,
        "workflow": 1,
        "capability": 2,
        "delivery_or_integration": 3,
    }
    scored_nodes = {id(node): _taxonomy_node_quality_score(node) for node in normalized}
    normalized.sort(
        key=lambda node: (
            layer_order.get(str(node.get("layer") or ""), 99),
            -scored_nodes[id(node)][0],
            -scored_nodes[id(node)][1],
            -scored_nodes[id(node)][2],
            scored_nodes[id(node)][3],
            str(node.get("phrase") or ""),
        )
    )
    return normalized[:80]


def _suppress_redundant_taxonomy_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phrases_by_layer = {
        layer: {_normalize_phrase_key(node.get("phrase")) for node in nodes if node.get("layer") == layer}
        for layer in TAXONOMY_LAYERS
    }
    suppressed_customer_phrases = set()
    customer_phrases = phrases_by_layer.get("customer_archetype", set())
    if "private bank" in customer_phrases:
        suppressed_customer_phrases.add("bank")
    if "asset manager" in customer_phrases:
        suppressed_customer_phrases.add("wealth manager")

    workflow_phrases = phrases_by_layer.get("workflow", set())
    delivery_phrases = phrases_by_layer.get("delivery_or_integration", set())
    suppressed_workflow_phrases = {
        phrase
        for phrase in workflow_phrases
        if phrase in delivery_phrases
        or (
            _phrase_word_count(phrase) <= 2
            and any(other != phrase and other.startswith(f"{phrase} ") for other in workflow_phrases)
        )
    }

    filtered: list[dict[str, Any]] = []
    for node in nodes:
        layer = node.get("layer")
        phrase_key = _normalize_phrase_key(node.get("phrase"))
        if layer == "customer_archetype" and phrase_key in suppressed_customer_phrases:
            continue
        if layer == "workflow" and phrase_key in suppressed_workflow_phrases:
            continue
        filtered.append(node)
    return filtered


def normalize_taxonomy_edges(edges: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    if not isinstance(edges, list):
        edges = []
    for item in edges:
        if not isinstance(item, dict):
            continue
        from_node_id = str(item.get("from_node_id") or "").strip()
        to_node_id = str(item.get("to_node_id") or "").strip()
        relation_type = str(item.get("relation_type") or "").strip()
        if not from_node_id or not to_node_id or relation_type not in RELATION_TYPES:
            continue
        key = (from_node_id, to_node_id, relation_type)
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "from_node_id": from_node_id,
                "to_node_id": to_node_id,
                "relation_type": relation_type,
                "evidence_ids": _normalize_string_list(item.get("evidence_ids"), max_items=12, max_len=96),
            }
        )
    return normalized[:120]


def normalize_lens_seeds(seeds: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(seeds, list):
        seeds = []
    for item in seeds:
        if not isinstance(item, dict):
            continue
        lens_type = str(item.get("lens_type") or "").strip()
        rationale = _safe_phrase(item.get("rationale"), max_len=320)
        if lens_type not in LENS_TYPES or not rationale:
            continue
        normalized.append(
            {
                "id": str(item.get("id") or _stable_id("lens", lens_type, rationale)),
                "lens_type": lens_type,
                "label": _safe_phrase(item.get("label"), max_len=120) or lens_type.replace("_", " "),
                "query_phrase": _safe_phrase(item.get("query_phrase"), max_len=120) or None,
                "rationale": rationale,
                "supporting_node_ids": _normalize_string_list(item.get("supporting_node_ids"), max_items=12, max_len=96),
                "evidence_ids": _normalize_string_list(item.get("evidence_ids"), max_items=12, max_len=96),
                "confidence": _clamp_confidence(item.get("confidence"), default=0.7),
            }
        )
    return normalized[:12]


def _build_taxonomy_map(
    context_pack_v2: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = normalize_taxonomy_nodes(
        _taxonomy_phrase_candidates(context_pack_v2, layer="customer_archetype")
        + _taxonomy_phrase_candidates(context_pack_v2, layer="workflow")
        + _taxonomy_phrase_candidates(context_pack_v2, layer="capability")
        + _taxonomy_phrase_candidates(context_pack_v2, layer="delivery_or_integration")
    )
    nodes_by_layer: dict[str, list[dict[str, Any]]] = {
        layer: [node for node in nodes if node.get("layer") == layer and node.get("scope_status") != "removed"]
        for layer in TAXONOMY_LAYERS
    }
    edges: list[dict[str, Any]] = []
    for capability in nodes_by_layer.get("capability", [])[:8]:
        for workflow in nodes_by_layer.get("workflow", [])[:6]:
            shared = sorted(
                set(capability.get("evidence_ids") or []) & set(workflow.get("evidence_ids") or [])
            )
            if not shared:
                continue
            edges.append(
                {
                    "from_node_id": capability["id"],
                    "to_node_id": workflow["id"],
                    "relation_type": "supports_workflow",
                    "evidence_ids": shared[:6],
                }
            )
    for customer in nodes_by_layer.get("customer_archetype", [])[:8]:
        for capability in nodes_by_layer.get("capability", [])[:8]:
            shared = sorted(
                set(customer.get("evidence_ids") or []) & set(capability.get("evidence_ids") or [])
            )
            if not shared:
                continue
            edges.append(
                {
                    "from_node_id": customer["id"],
                    "to_node_id": capability["id"],
                    "relation_type": "buys_capability",
                    "evidence_ids": shared[:6],
                }
            )
    workflows = nodes_by_layer.get("workflow", [])[:8]
    for idx, workflow in enumerate(workflows):
        for other in workflows[idx + 1 : idx + 3]:
            edges.append(
                {
                    "from_node_id": workflow["id"],
                    "to_node_id": other["id"],
                    "relation_type": "adjacent_to",
                    "evidence_ids": _normalize_string_list(
                        list(set((workflow.get("evidence_ids") or []) + (other.get("evidence_ids") or []))),
                        max_items=6,
                        max_len=96,
                    ),
                }
            )
    return nodes, normalize_taxonomy_edges(edges)


def _taxonomy_node_quality_score(node: dict[str, Any]) -> tuple[float, int, int, int]:
    phrase = _safe_phrase(node.get("phrase"), max_len=120)
    lowered = phrase.lower()
    evidence_count = len(node.get("evidence_ids") or [])
    alias_count = len(node.get("aliases") or [])
    word_count = _phrase_word_count(phrase)
    quality = float(node.get("confidence") or 0.0)
    capability_keyword_hits = sum(1 for keyword in CAPABILITY_QUALITY_KEYWORDS if keyword in lowered)
    is_broad_cluster_phrase = (
        capability_keyword_hits >= 2
        and word_count >= 3
        and any(token in lowered for token in (",", " et ", " and ", " / "))
    )

    if node.get("layer") == "capability":
        if capability_keyword_hits:
            quality += 0.18
        if capability_keyword_hits >= 2:
            quality += 0.08
        if any(lowered.startswith(prefix) for prefix in CAPABILITY_DEMOTION_PREFIXES):
            quality -= 0.22
        if 2 <= word_count <= 6:
            quality += 0.12
        if 2 <= word_count <= 5 and any(keyword in lowered for keyword in ("pms", "oms", "stp", "trading", "routage", "modélisation", "arbitrage", "benchmark", "compliance", "reporting")):
            quality += 0.08
        if is_broad_cluster_phrase:
            quality += 0.12
        if ("," in phrase and not is_broad_cluster_phrase) or word_count > 8:
            quality -= 0.12
        if any(lowered.startswith(prefix) for prefix in SENTENCE_LIKE_CAPABILITY_PREFIXES):
            quality -= 0.28
    elif node.get("layer") == "workflow":
        if 2 <= word_count <= 5:
            quality += 0.1
    elif node.get("layer") == "customer_archetype":
        if 2 <= word_count <= 4:
            quality += 0.08

    return (round(quality, 4), evidence_count, alias_count, -word_count)


def _generate_sourcing_brief_summary(
    company_name: str,
    *,
    capabilities: list[dict[str, Any]],
    workflows: list[dict[str, Any]],
    customers: list[dict[str, Any]],
    named_customers: list[dict[str, Any]],
) -> str:
    capability_names = [node.get("phrase") for node in capabilities[:3] if node.get("phrase")]
    workflow_names = [node.get("phrase") for node in workflows[:3] if node.get("phrase")]
    customer_names = [node.get("phrase") for node in customers[:3] if node.get("phrase")]
    named_names = [item.get("name") for item in named_customers[:2] if item.get("name")]
    sentences: list[str] = []
    if capability_names and customer_names:
        sentences.append(
            f"{company_name} appears to offer {', '.join(capability_names)} to {', '.join(customer_names)} buyers."
        )
    elif capability_names:
        sentences.append(f"{company_name} appears to offer {', '.join(capability_names)}.")
    if workflow_names:
        sentences.append(
            f"The strongest workflow signals cluster around {', '.join(workflow_names)}."
        )
    if named_names:
        sentences.append(
            f"Named customer proof includes {', '.join(named_names)}."
        )
    if customer_names and workflow_names:
        sentences.append(
            f"This should anchor the sourcing brief around {', '.join(customer_names[:2])} buyers and adjacent workflows around {workflow_names[0]}."
        )
    if not sentences:
        sentences.append(
            f"{company_name} has limited first-party evidence; add more product, customer, or integration pages before trusting the sourcing brief."
        )
    return " ".join(sentences)[:800]


def _build_sourcing_brief_artifacts(
    profile: CompanyProfile,
    *,
    source_pills: list[dict[str, Any]],
    override_nodes: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    full_context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    context_pack_v2 = _source_scoped_context_pack(
        full_context_pack_v2,
        buyer_url=profile.buyer_company_url,
    )
    nodes, edges = _build_taxonomy_map(context_pack_v2)
    if override_nodes is not None:
        nodes = normalize_taxonomy_nodes(override_nodes)
        edges = normalize_taxonomy_edges(edges)

    active_nodes = [node for node in nodes if node.get("scope_status") != "removed"]
    customers = [node for node in active_nodes if node.get("layer") == "customer_archetype"]
    workflows = [node for node in active_nodes if node.get("layer") == "workflow"]
    capabilities = [node for node in active_nodes if node.get("layer") == "capability"]
    delivery_nodes = [node for node in active_nodes if node.get("layer") == "delivery_or_integration"]
    named_customers = [
        item for item in (context_pack_v2.get("named_customers") or []) if isinstance(item, dict)
    ][:8]
    integrations = [
        item for item in (context_pack_v2.get("integrations") or []) if isinstance(item, dict)
    ][:8]
    evidence_ids_for_customers = _normalize_string_list(
        [item.get("evidence_id") for item in named_customers if isinstance(item, dict)],
        max_items=12,
        max_len=96,
    )
    evidence_ids_for_integrations = _normalize_string_list(
        [item.get("evidence_id") for item in integrations if isinstance(item, dict)],
        max_items=12,
        max_len=96,
    )
    company_name = (
        str((((context_pack_v2 or {}).get("sites") or [{}])[0].get("company_name") or "")).strip()
        or normalize_domain(profile.buyer_company_url)
        or "Source company"
    )

    lens_seeds: list[dict[str, Any]] = []
    if named_customers or customers:
        lens_seeds.append(
            {
                "lens_type": "same_customer_different_product",
                "label": "Same Customer, Different Product",
                "query_phrase": (workflows[0]["phrase"] if workflows else capabilities[0]["phrase"] if capabilities else None),
                "rationale": (
                    "Named customer and customer-archetype evidence suggests adjacent products sold into the same buying accounts."
                ),
                "supporting_node_ids": [node["id"] for node in (customers[:2] + capabilities[:2])],
                "evidence_ids": evidence_ids_for_customers[:6],
                "confidence": 0.82 if named_customers else 0.7,
            }
        )
    if capabilities:
        lens_seeds.append(
            {
                "lens_type": "same_product_different_customer",
                "label": "Same Product, Different Customer",
                "query_phrase": capabilities[0]["phrase"],
                "rationale": "Capability evidence is strong enough to map vendors offering the same solution into other customer segments.",
                "supporting_node_ids": [node["id"] for node in (capabilities[:2] + customers[:2])],
                "evidence_ids": _normalize_string_list(
                    [evidence_id for node in capabilities[:2] for evidence_id in (node.get("evidence_ids") or [])],
                    max_items=6,
                    max_len=96,
                ),
                "confidence": 0.8 if len(capabilities) >= 2 else 0.68,
            }
        )
    if workflows and (len(capabilities) >= 2 or integrations):
        lens_seeds.append(
            {
                "lens_type": "different_product_different_customer_within_market_box",
                "label": "Different Product, Different Customer, Same Market Box",
                "query_phrase": workflows[0]["phrase"],
                "rationale": "Workflow and integration evidence bounds a market box wide enough for adjacency mapping without drifting into generic industry search.",
                "supporting_node_ids": [node["id"] for node in (workflows[:2] + capabilities[:2])],
                "evidence_ids": evidence_ids_for_integrations[:6]
                or _normalize_string_list(
                    [evidence_id for node in workflows[:2] for evidence_id in (node.get("evidence_ids") or [])],
                    max_items=6,
                    max_len=96,
                ),
                "confidence": 0.66,
            }
        )
    lens_seeds = normalize_lens_seeds(lens_seeds)

    adjacency_hypotheses: list[dict[str, Any]] = []
    if customers and workflows:
        adjacency_hypotheses.append(
            {
                "id": _stable_id("hypothesis", customers[0]["phrase"], workflows[0]["phrase"]),
                "text": (
                    f"Customers like {customers[0]['phrase']} that buy {capabilities[0]['phrase'] if capabilities else 'the source product'} "
                    f"likely evaluate adjacent workflows around {workflows[0]['phrase']}."
                )[:280],
                "supporting_node_ids": [customers[0]["id"], workflows[0]["id"], *( [capabilities[0]["id"]] if capabilities else [])],
                "evidence_ids": _normalize_string_list(
                    list((customers[0].get("evidence_ids") or []) + (workflows[0].get("evidence_ids") or [])),
                    max_items=6,
                    max_len=96,
                ),
                "confidence": 0.72,
            }
        )
    if integrations and capabilities:
        adjacency_hypotheses.append(
            {
                "id": _stable_id("hypothesis", capabilities[0]["phrase"], integrations[0].get("name") or ""),
                "text": (
                    f"Integration evidence around {integrations[0].get('name') or 'the ecosystem'} suggests buyers may bundle {capabilities[0]['phrase']} with adjacent tooling."
                )[:280],
                "supporting_node_ids": [capabilities[0]["id"]],
                "evidence_ids": _normalize_string_list(
                    [integrations[0].get("evidence_id")] + (capabilities[0].get("evidence_ids") or []),
                    max_items=6,
                    max_len=96,
                ),
                "confidence": 0.68,
            }
        )

    confidence_gaps: list[str] = []
    if not customers:
        confidence_gaps.append("Customer archetype evidence is still thin.")
    if not capabilities:
        confidence_gaps.append("Capability evidence is too weak to anchor competitor mapping.")
    if not workflows:
        confidence_gaps.append("Workflow evidence is too weak to bound the market box.")
    if not named_customers:
        confidence_gaps.append("No named customer proof yet; same-customer adjacency is lower confidence.")

    open_questions = confidence_gaps.copy()
    if not open_questions:
        open_questions.extend(
            [
                "Which customer segment is most strategic for the first sourcing brief pass?",
                "Which adjacent workflow should discovery prioritize first?",
            ]
        )

    sourcing_brief = {
        "source_company": {
            "name": company_name,
            "website": normalize_url(profile.buyer_company_url) if profile.buyer_company_url else None,
        },
        "source_summary": _generate_sourcing_brief_summary(
            company_name,
            capabilities=capabilities,
            workflows=workflows,
            customers=customers,
            named_customers=named_customers,
        ),
        "reasoning_status": "not_run",
        "reasoning_warning": None,
        "reasoning_provider": None,
        "reasoning_model": None,
        "customer_nodes": customers[:8],
        "workflow_nodes": workflows[:8],
        "capability_nodes": capabilities[:8],
        "delivery_or_integration_nodes": delivery_nodes[:8],
        "named_customer_proof": named_customers,
        "partner_integration_proof": integrations,
        "active_lenses": lens_seeds,
        "adjacency_hypotheses": adjacency_hypotheses[:6],
        "strongest_evidence_buckets": [
            {"label": "Customers", "count": len(named_customers)},
            {"label": "Customer archetypes", "count": len(customers)},
            {"label": "Capabilities", "count": len(capabilities)},
            {"label": "Workflows", "count": len(workflows)},
            {"label": "Delivery / Integration", "count": len(delivery_nodes)},
            {"label": "Integrations", "count": len(integrations)},
        ],
        "confidence_gaps": confidence_gaps[:6],
        "open_questions": open_questions[:8],
        "crawl_coverage": context_pack_v2.get("crawl_coverage") or {},
        "confirmed_at": None,
    }

    sourcing_brief = _reason_sourcing_brief(
        company_name=company_name,
        company_url=normalize_url(profile.buyer_company_url) if profile.buyer_company_url else None,
        crawl_coverage=context_pack_v2.get("crawl_coverage") or {},
        nodes=active_nodes,
        lens_seeds=lens_seeds,
        named_customers=named_customers,
        integrations=integrations,
        fallback_brief=sourcing_brief,
    )

    return (
        context_pack_v2,
        nodes,
        edges,
        lens_seeds,
        open_questions[:8],
        sourcing_brief,
    )


def _clamp_confidence(value: Any, *, default: float = 0.65) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = default
    return max(0.0, min(1.0, round(confidence, 2)))


def _extract_json_object(blob: Any) -> Optional[dict[str, Any]]:
    text = str(blob or "").strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(text[start_idx : end_idx + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _truncate_words(value: Any, *, max_words: int, max_chars: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    last_space = clipped.rfind(" ")
    if last_space >= max_chars * 0.6:
        clipped = clipped[:last_space].rstrip()
    return clipped


def _compact_sourcing_brief_payload(
    *,
    company_name: str,
    company_url: Optional[str],
    crawl_coverage: dict[str, Any],
    nodes: list[dict[str, Any]],
    lens_seeds: list[dict[str, Any]],
    named_customers: list[dict[str, Any]],
    integrations: list[dict[str, Any]],
    fallback_brief: dict[str, Any],
) -> dict[str, Any]:
    def _node_stub(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": node.get("id"),
            "layer": node.get("layer"),
            "phrase": node.get("phrase"),
            "aliases": (node.get("aliases") or [])[:4],
            "confidence": node.get("confidence"),
            "evidence_ids": (node.get("evidence_ids") or [])[:6],
        }

    def _proof_stub(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": item.get("name"),
            "context": _safe_phrase(item.get("context"), max_len=180),
            "evidence_type": item.get("evidence_type"),
            "evidence_id": item.get("evidence_id"),
        }

    ranked_nodes_by_layer: dict[str, list[dict[str, Any]]] = {}
    for layer in ("customer_archetype", "workflow", "capability", "delivery_or_integration"):
        ranked_nodes_by_layer[layer] = [
            _node_stub(node)
            for node in nodes
            if node.get("layer") == layer and node.get("scope_status") != "removed"
        ][:8]

    evidence_highlights = {
        "named_customer_names": [item.get("name") for item in named_customers[:8] if item.get("name")],
        "integration_partner_names": [item.get("name") for item in integrations[:8] if item.get("name")],
        "top_capability_phrases": [node.get("phrase") for node in ranked_nodes_by_layer.get("capability", [])[:6]],
        "top_workflow_phrases": [node.get("phrase") for node in ranked_nodes_by_layer.get("workflow", [])[:6]],
        "top_customer_phrases": [node.get("phrase") for node in ranked_nodes_by_layer.get("customer_archetype", [])[:6]],
        "top_delivery_phrases": [
            node.get("phrase") for node in ranked_nodes_by_layer.get("delivery_or_integration", [])[:6]
        ],
    }

    return {
        "prompt_version": MARKET_MAP_REASONING_PROMPT_VERSION,
        "source_company": {
            "name": company_name,
            "website": company_url,
        },
        "crawl_coverage": crawl_coverage or {},
        "taxonomy_nodes": [_node_stub(node) for node in nodes[:36]],
        "ranked_nodes_by_layer": ranked_nodes_by_layer,
        "lens_seeds": [
            {
                "id": item.get("id"),
                "lens_type": item.get("lens_type"),
                "label": item.get("label"),
                "query_phrase": item.get("query_phrase"),
                "rationale": item.get("rationale"),
                "supporting_node_ids": (item.get("supporting_node_ids") or [])[:6],
                "evidence_ids": (item.get("evidence_ids") or [])[:6],
                "confidence": item.get("confidence"),
            }
            for item in lens_seeds[:8]
        ],
        "named_customer_proof": [_proof_stub(item) for item in named_customers[:8]],
        "partner_integration_proof": [_proof_stub(item) for item in integrations[:8]],
        "evidence_highlights": evidence_highlights,
        "selection_rules": {
            "summary_max_words": 120,
            "max_customer_nodes": 4,
            "max_workflow_nodes": 4,
            "max_capability_nodes": 6,
            "max_delivery_nodes": 4,
            "max_active_lenses": 3,
            "max_adjacency_hypotheses": 4,
            "max_confidence_gaps": 4,
            "max_open_questions": 4,
        },
        "fallback_brief": {
            "source_summary": fallback_brief.get("source_summary"),
            "customer_node_ids": [node.get("id") for node in (fallback_brief.get("customer_nodes") or [])[:8]],
            "workflow_node_ids": [node.get("id") for node in (fallback_brief.get("workflow_nodes") or [])[:8]],
            "capability_node_ids": [node.get("id") for node in (fallback_brief.get("capability_nodes") or [])[:8]],
            "delivery_or_integration_node_ids": [
                node.get("id") for node in (fallback_brief.get("delivery_or_integration_nodes") or [])[:8]
            ],
            "active_lens_ids": [lens.get("id") for lens in (fallback_brief.get("active_lenses") or [])[:8]],
            "confidence_gaps": (fallback_brief.get("confidence_gaps") or [])[:8],
            "open_questions": (fallback_brief.get("open_questions") or [])[:8],
        },
    }


def _sourcing_brief_reasoning_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are an M&A sourcing analyst generating a phase-1 Sourcing Brief from normalized source-company evidence.\n\n"
        "Your job is discovery-first, not transaction-first.\n"
        "Use only the provided source-scoped artifacts.\n"
        "Do not invent nodes, customers, integrations, or lenses.\n"
        "Select from the provided node IDs and lens IDs only.\n"
        "Keep workflow, capability, and delivery/integration separate.\n"
        "Prefer source-company rendered product and solution evidence over generic summaries.\n"
        "Use the source company's own vocabulary from the provided nodes and proof, not fixed industry or product examples.\n"
        "If evidence is thin, keep fields sparse rather than generic.\n\n"
        "Selection rules:\n"
        "- Keep the source summary under 120 words and make it specific.\n"
        "- The source summary is the opening paragraph of the sourcing brief and should help decide what expansion work to prioritize next.\n"
        "- Use the summary to state what the source company appears to sell, to whom, across which workflows, and what adjacency box it suggests.\n"
        "- Prefer ranked nodes and evidence highlights over fallback summary text.\n"
        "- Customer nodes must be buyer/operator archetypes, never named accounts.\n"
        "- Workflow nodes must be operating jobs or workflow clusters.\n"
        "- Capability nodes must be feature clusters or product functions, not delivery traits.\n"
        "- When capability candidates mix core workflow-enabling functions with secondary controls, analytics, or client-service details, prefer the core workflow-enabling functions.\n"
        "- Prefer broader platform capability clusters over narrow leaf features or isolated safeguard checks when both are available.\n"
        "- Do not simply take the first ranked nodes if lower-ranked nodes better express the product's core operating value.\n"
        "- Delivery/integration nodes must include APIs, docs, ecosystem, or architecture traits only.\n"
        "- Activate the third lens only when the market box is clearly bounded by customers, workflows, or capabilities.\n"
        "- Open questions must be real evidence gaps grounded in the current source evidence, not roadmap, strategy, or generic follow-ups.\n"
        "- Prefer questions about missing segmentation, workflow depth, integration scope, customer proof, or capability detail.\n\n"
        "Return only valid JSON with this exact shape:\n"
        f"{json.dumps(MARKET_MAP_REASONING_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Prioritize, in order:\n"
        "1. ranked source-company capability nodes\n"
        "2. ranked source-company workflow nodes\n"
        "3. named customer proof\n"
        "4. integration proof\n"
        "5. fallback brief summary\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _compact_expansion_research_payload(
    *,
    company_name: str,
    company_url: Optional[str],
    sourcing_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
    expansion_inputs: list[dict[str, Any]],
    geo_scope: dict[str, Any],
    fallback_brief: dict[str, Any],
) -> dict[str, Any]:
    def _node_stub(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": node.get("id"),
            "layer": node.get("layer"),
            "phrase": node.get("phrase"),
            "confidence": node.get("confidence"),
            "evidence_ids": (node.get("evidence_ids") or [])[:4],
        }

    def _proof_stub(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": item.get("name"),
            "source_url": item.get("source_url"),
            "context": _safe_phrase(item.get("context"), max_len=180),
            "evidence_type": item.get("evidence_type"),
        }

    expansion_input_stubs = []
    for expansion_input in expansion_inputs[:6]:
        if not isinstance(expansion_input, dict):
            continue
        expansion_input_stubs.append(
            {
                "name": expansion_input.get("name"),
                "website": expansion_input.get("website"),
                "source_summary": _truncate_words(expansion_input.get("source_summary"), max_words=50, max_chars=300),
                "top_capabilities": [
                    node.get("phrase")
                    for node in (expansion_input.get("taxonomy_nodes") or [])
                    if isinstance(node, dict) and node.get("layer") == "capability" and node.get("phrase")
                ][:6],
                "top_customer_segments": [
                    node.get("phrase")
                    for node in (expansion_input.get("taxonomy_nodes") or [])
                    if isinstance(node, dict) and node.get("layer") == "customer_archetype" and node.get("phrase")
                ][:6],
                "named_customer_proof": [_proof_stub(item) for item in (expansion_input.get("named_customer_proof") or [])[:4]],
            }
        )

    return {
        "prompt_version": EXPANSION_RESEARCH_PROMPT_VERSION,
        "source_company": {
            "name": company_name,
            "website": company_url,
        },
        "source_brief": {
            "source_summary": _truncate_words(sourcing_brief.get("source_summary"), max_words=120, max_chars=800),
            "customer_nodes": [_node_stub(node) for node in (sourcing_brief.get("customer_nodes") or [])[:6] if isinstance(node, dict)],
            "workflow_nodes": [_node_stub(node) for node in (sourcing_brief.get("workflow_nodes") or [])[:6] if isinstance(node, dict)],
            "capability_nodes": [_node_stub(node) for node in (sourcing_brief.get("capability_nodes") or [])[:8] if isinstance(node, dict)],
            "delivery_nodes": [
                _node_stub(node)
                for node in (sourcing_brief.get("delivery_or_integration_nodes") or [])[:6]
                if isinstance(node, dict)
            ],
            "named_customer_proof": [_proof_stub(item) for item in (sourcing_brief.get("named_customer_proof") or [])[:8] if isinstance(item, dict)],
            "partner_integration_proof": [_proof_stub(item) for item in (sourcing_brief.get("partner_integration_proof") or [])[:8] if isinstance(item, dict)],
            "secondary_evidence_proof": [
                {
                    "title": item.get("title"),
                    "claim_type": item.get("claim_type"),
                    "publisher_type": item.get("publisher_type"),
                    "url": item.get("url"),
                    "claim_text": _safe_phrase(item.get("claim_text"), max_len=180),
                }
                for item in (sourcing_brief.get("secondary_evidence_proof") or [])[:8]
                if isinstance(item, dict)
            ],
        },
        "taxonomy_nodes": [_node_stub(node) for node in taxonomy_nodes[:32]],
        "expansion_inputs": expansion_input_stubs,
        "geo_scope": {
            "region": geo_scope.get("region"),
            "include_countries": _normalize_string_list(geo_scope.get("include_countries"), max_items=8, max_len=64),
            "exclude_countries": _normalize_string_list(geo_scope.get("exclude_countries"), max_items=8, max_len=64),
        },
        "fallback_expansion_brief": normalize_expansion_brief(fallback_brief),
        "selection_rules": {
            "max_adjacent_capabilities": 8,
            "max_adjacent_customer_segments": 6,
            "max_named_account_anchors": 8,
            "max_geography_expansions": 6,
        },
    }


def _expansion_research_prompt(payload: dict[str, Any]) -> str:
    return (
        "You are generating a bounded Expansion Research artifact for M&A sourcing.\n\n"
        "Start only from the provided source-company brief, taxonomy, expansion inputs, and geo scope.\n"
        "You may use web search when available, but remain tightly bounded to one-hop adjacencies around the source evidence.\n"
        "Do not generate a full expansion map and do not return a long company list.\n"
        "Prefer categories that repeatedly appear across source evidence, expansion inputs, named accounts, integrations, or nearby public evidence.\n"
        "Demote niche, edge-case, or peripheral use cases that are too small or too optional to steer primary discovery.\n"
        "If an adjacency is real but likely too narrow to drive target discovery on its own, keep it and mark it low importance rather than dropping it.\n\n"
        "You must propose only these buckets:\n"
        "- adjacent_capabilities\n"
        "- adjacent_customer_segments\n"
        "- named_account_anchors\n"
        "- geography_expansions\n\n"
        "For each item provide:\n"
        "- label\n"
        "- why_it_matters\n"
        "- evidence_urls\n"
        "- source_entity_names\n"
        "- market_importance: high, medium, or low\n"
        "- operational_centrality: core, meaningful, or peripheral\n"
        "- workflow_criticality: high, medium, or low\n"
        "- daily_operator_usage: high, medium, or low\n"
        "- switching_cost_intensity: high, medium, or low\n"
        "- confidence\n\n"
        "Interpretation rules:\n"
        "- market_importance = a secondary signal about whether the adjacency appears broad enough to matter beyond a niche edge case\n"
        "- operational_centrality = whether the adjacency appears central to day-to-day customer operations versus peripheral or optional\n"
        "- workflow_criticality = whether the capability sits on a business-critical workflow path\n"
        "- daily_operator_usage = whether operators appear to rely on it frequently in routine execution\n"
        "- switching_cost_intensity = whether replacing it would likely be operationally painful due to workflow embedding, data gravity, controls, or integration depth\n"
        "- avoid treating a narrow compliance check, isolated add-on, or rarely emphasized feature as equal to a system-of-record workflow\n"
        "- named_account_anchors should be concrete institutions useful for tracing peer vendors or adjacent stack components\n"
        "- geography_expansions should be bounded adjacent markets, not all plausible countries\n\n"
        "Return only valid JSON with this exact shape:\n"
        f"{json.dumps(EXPANSION_RESEARCH_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Input:\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def _expansion_normalization_prompt(
    *,
    company_name: str,
    research_payload: dict[str, Any],
    fallback_brief: dict[str, Any],
) -> str:
    return (
        "You are normalizing an Expansion Research artifact into a graph-safe Expansion Brief.\n\n"
        f"The source company is {company_name}.\n"
        "Do not invent items not present in the research payload unless they already exist in the fallback brief.\n"
        "Preserve selective breadth. Fewer strong items are better than many weak ones.\n"
        "Use the fallback brief only when the research payload is missing a useful item already grounded in source evidence.\n"
        "Each item must receive expansion_type, status, confidence, market_importance, operational_centrality, workflow_criticality, daily_operator_usage, switching_cost_intensity, and priority_tier.\n\n"
        "Priority rules:\n"
        "- Use priority_tier=core_adjacent only when workflow_criticality is high and either operational_centrality is core or switching_cost_intensity is high.\n"
        "- Use priority_tier=meaningful_adjacent when the adjacency is relevant but not clearly a primary discovery lane.\n"
        "- Use priority_tier=edge_case for narrow, peripheral, or weakly repeated adjacencies.\n"
        "- Small, optional, or rarely repeated use cases should generally be edge_case or meaningful_adjacent, not core_adjacent, even if they look adjacent.\n"
        "- Named accounts grounded in source evidence can stay source_grounded.\n"
        "- Geography expansions sourced only from workspace scope may remain source_grounded but should not automatically imply a high-priority lane.\n\n"
        "Return only valid JSON with this exact shape:\n"
        f"{json.dumps(EXPANSION_BRIEF_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"Research payload:\n{json.dumps(research_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Fallback brief:\n{json.dumps(normalize_expansion_brief(fallback_brief), ensure_ascii=False, indent=2)}"
    )


def _merge_reasoned_sourcing_brief(
    *,
    response_text: str,
    nodes: list[dict[str, Any]],
    lens_seeds: list[dict[str, Any]],
    fallback_brief: dict[str, Any],
    reasoning_provider: Optional[str] = None,
    reasoning_model: Optional[str] = None,
) -> dict[str, Any]:
    parsed = _extract_json_object(response_text)
    if not parsed:
        return {
            **fallback_brief,
            "reasoning_status": "degraded",
            "reasoning_warning": "Market map reasoning returned invalid structured output. Showing deterministic fallback from the source evidence graph.",
            "reasoning_provider": reasoning_provider,
            "reasoning_model": reasoning_model,
        }

    node_by_id = {
        str(node.get("id")): node
        for node in nodes
        if isinstance(node, dict) and node.get("id") and node.get("scope_status") != "removed"
    }
    lens_by_id = {
        str(item.get("id")): item
        for item in lens_seeds
        if isinstance(item, dict) and item.get("id")
    }

    def _select_nodes(key: str, fallback_key: str, *, max_items: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for node_id in _normalize_string_list(parsed.get(key), max_items=max_items, max_len=96):
            node = node_by_id.get(node_id)
            if node and node not in selected:
                selected.append(node)
        return selected or list(fallback_brief.get(fallback_key) or [])[:max_items]

    selected_customers = _select_nodes("customer_node_ids", "customer_nodes", max_items=4)
    selected_workflows = _select_nodes("workflow_node_ids", "workflow_nodes", max_items=4)
    selected_capabilities = _select_nodes("capability_node_ids", "capability_nodes", max_items=6)
    selected_delivery = _select_nodes(
        "delivery_or_integration_node_ids",
        "delivery_or_integration_nodes",
        max_items=4,
    )

    selected_lenses: list[dict[str, Any]] = []
    for lens_id in _normalize_string_list(parsed.get("active_lens_ids"), max_items=3, max_len=96):
        lens = lens_by_id.get(lens_id)
        if lens and lens not in selected_lenses:
            selected_lenses.append(lens)
    if not selected_lenses:
        selected_lenses = list(fallback_brief.get("active_lenses") or [])[:3]

    adjacency_hypotheses: list[dict[str, Any]] = []
    seen_hypothesis_keys: set[str] = set()
    for item in parsed.get("adjacency_hypotheses") or []:
        if not isinstance(item, dict):
            continue
        text = _safe_phrase(item.get("text"), max_len=280)
        if not text:
            continue
        text_key = _normalize_phrase_key(text)
        if text_key in seen_hypothesis_keys:
            continue
        seen_hypothesis_keys.add(text_key)
        supporting_node_ids = _normalize_string_list(item.get("supporting_node_ids"), max_items=8, max_len=96)
        supporting_node_ids = [node_id for node_id in supporting_node_ids if node_id in node_by_id]
        derived_evidence_ids = _normalize_string_list(
            [
                evidence_id
                for node_id in supporting_node_ids
                for evidence_id in (node_by_id[node_id].get("evidence_ids") or [])
            ],
            max_items=6,
            max_len=96,
        )
        adjacency_hypotheses.append(
            {
                "id": _stable_id("hypothesis", text, ",".join(supporting_node_ids)),
                "text": text,
                "supporting_node_ids": supporting_node_ids,
                "evidence_ids": derived_evidence_ids,
                "confidence": _clamp_confidence(item.get("confidence"), default=0.68),
            }
        )
        if len(adjacency_hypotheses) >= 4:
            break
    if not adjacency_hypotheses:
        adjacency_hypotheses = list(fallback_brief.get("adjacency_hypotheses") or [])[:6]

    merged = {
        **fallback_brief,
        "source_summary": _truncate_words(
            _safe_phrase(parsed.get("source_summary"), max_len=1200) or fallback_brief.get("source_summary"),
            max_words=120,
            max_chars=800,
        )
        or fallback_brief.get("source_summary"),
        "reasoning_status": "success",
        "reasoning_warning": None,
        "reasoning_provider": reasoning_provider,
        "reasoning_model": reasoning_model,
        "customer_nodes": selected_customers,
        "workflow_nodes": selected_workflows,
        "capability_nodes": selected_capabilities,
        "delivery_or_integration_nodes": selected_delivery,
        "active_lenses": selected_lenses,
        "adjacency_hypotheses": adjacency_hypotheses[:4],
        "confidence_gaps": normalize_open_questions(parsed.get("confidence_gaps"))
        or list(fallback_brief.get("confidence_gaps") or [])[:4],
        "open_questions": normalize_open_questions(parsed.get("open_questions"))
        or list(fallback_brief.get("open_questions") or [])[:4],
    }
    return merged


def _reason_sourcing_brief(
    *,
    company_name: str,
    company_url: Optional[str],
    crawl_coverage: dict[str, Any],
    nodes: list[dict[str, Any]],
    lens_seeds: list[dict[str, Any]],
    named_customers: list[dict[str, Any]],
    integrations: list[dict[str, Any]],
    fallback_brief: dict[str, Any],
) -> dict[str, Any]:
    if not nodes:
        return {
            **fallback_brief,
            "reasoning_status": "not_applicable",
            "reasoning_warning": "Reasoning did not run because the source crawl did not produce enough taxonomy evidence yet.",
            "reasoning_provider": None,
            "reasoning_model": None,
        }
    settings = get_settings()
    if not any([settings.gemini_api_key, settings.openai_api_key, settings.anthropic_api_key]):
        return {
            **fallback_brief,
            "reasoning_status": str(fallback_brief.get("reasoning_status") or "degraded"),
            "reasoning_warning": fallback_brief.get("reasoning_warning"),
            "reasoning_provider": fallback_brief.get("reasoning_provider"),
            "reasoning_model": fallback_brief.get("reasoning_model"),
        }

    prompt = _sourcing_brief_reasoning_prompt(
        _compact_sourcing_brief_payload(
            company_name=company_name,
            company_url=company_url,
            crawl_coverage=crawl_coverage,
            nodes=nodes,
            lens_seeds=lens_seeds,
            named_customers=named_customers,
            integrations=integrations,
            fallback_brief=fallback_brief,
        )
    )
    try:
        response = LLMOrchestrator().run_stage(
            LLMRequest(
                stage=LLMStage.market_map_reasoning,
                prompt=prompt,
                timeout_seconds=60,
                use_web_search=False,
                expect_json=True,
                metadata={"company_name": company_name, "phase": "sourcing_brief"},
            )
        )
    except LLMOrchestrationError as exc:
        attempt_error = ""
        if getattr(exc, "attempts", None):
            last_attempt = exc.attempts[-1]
            attempt_error = str(getattr(last_attempt, "error_message", "") or "").strip()
        warning = "Market map reasoning failed after model retries. Showing deterministic fallback from the source evidence graph."
        if attempt_error:
            warning = f"{warning} Last error: {attempt_error[:240]}"
        return {
            **fallback_brief,
            "reasoning_status": "degraded",
            "reasoning_warning": warning,
            "reasoning_provider": None,
            "reasoning_model": None,
        }
    except Exception as exc:
        return {
            **fallback_brief,
            "reasoning_status": "degraded",
            "reasoning_warning": (
                "Market map reasoning failed unexpectedly. Showing deterministic fallback from the source evidence graph. "
                f"Last error: {str(exc)[:240]}"
            ),
            "reasoning_provider": None,
            "reasoning_model": None,
        }

    return _merge_reasoned_sourcing_brief(
        response_text=response.text,
        nodes=nodes,
        lens_seeds=lens_seeds,
        fallback_brief=fallback_brief,
        reasoning_provider=getattr(response, "provider", None),
        reasoning_model=getattr(response, "model", None),
    )


def _pill_label_for_url(
    url: str,
    *,
    buyer_url: Optional[str],
    comparator_seed_urls: Iterable[str],
    supporting_evidence_urls: Iterable[str],
) -> str:
    normalized = normalize_url(url)
    domain = normalize_domain(normalized) or "source"
    if normalized and normalize_domain(normalized) == normalize_domain(buyer_url):
        return "Buyer website"
    if normalized in {normalize_url(item) for item in comparator_seed_urls if normalize_url(item)}:
        return f"Comparator seed: {domain}"
    if normalized in {normalize_url(item) for item in supporting_evidence_urls if normalize_url(item)}:
        return f"Evidence source: {domain}"
    return domain


def normalize_source_pills(
    source_pills: Any,
    *,
    buyer_url: Optional[str] = None,
    comparator_seed_urls: Optional[Iterable[str]] = None,
    supporting_evidence_urls: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    pills: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    company_urls = list(comparator_seed_urls or [])
    evidence_urls = list(supporting_evidence_urls or [])

    if not isinstance(source_pills, list):
        source_pills = []

    for item in source_pills:
        if isinstance(item, str):
            payload = {"url": item}
        elif isinstance(item, dict):
            payload = item
        else:
            continue
        url = normalize_url(payload.get("url"))
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        label = str(payload.get("label") or "").strip() or _pill_label_for_url(
            url,
            buyer_url=buyer_url,
            comparator_seed_urls=company_urls,
            supporting_evidence_urls=evidence_urls,
        )
        pills.append(
            {
                "id": str(payload.get("id") or _stable_id("pill", url)),
                "label": label[:120],
                "url": url,
            }
        )
    return pills[:40]

def normalize_open_questions(open_questions: Any) -> list[str]:
    return _normalize_string_list(open_questions, max_items=12, max_len=240)


def normalize_expansion_items(
    items: Any,
    *,
    item_type: str,
    max_items: int = 8,
) -> list[dict[str, Any]]:
    if item_type not in EXPANSION_ITEM_TYPES:
        return []
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items or []:
        if not isinstance(raw, dict):
            continue
        label = _compact_phrase(raw.get("label") or raw.get("name"), max_words=8, max_len=120)
        if not label:
            continue
        if item_type == "named_account_anchor" and not _is_plausible_named_account_anchor_item({**raw, "label": label}):
            continue
        if item_type == "adjacent_capability" and not _is_high_quality_adjacent_capability_candidate(label):
            continue
        if item_type == "adjacent_customer_segment" and not _is_plausible_adjacent_customer_segment(label):
            continue
        key = _normalize_phrase_key(label)
        if key in seen:
            continue
        seen.add(key)
        status = str(raw.get("status") or "hypothesis").strip().lower()
        if status not in EXPANSION_ITEM_STATUSES:
            status = "hypothesis"
        normalized.append(
            {
                "id": str(raw.get("id") or _stable_id("expansion", item_type, label)),
                "label": label,
                "expansion_type": item_type,
                "status": status,
                "confidence": _clamp_confidence(raw.get("confidence"), default=0.6),
                "why_it_matters": str(raw.get("why_it_matters") or "").strip()[:400],
                "evidence_urls": [
                    url
                    for url in [
                        normalize_url(url)
                        for url in _normalize_string_list(raw.get("evidence_urls"), max_items=6, max_len=500)
                    ]
                    if url
                ],
                "supporting_node_ids": _normalize_string_list(raw.get("supporting_node_ids"), max_items=8, max_len=120),
                "source_entity_names": _normalize_string_list(raw.get("source_entity_names"), max_items=6, max_len=120),
                "market_importance": (
                    str(raw.get("market_importance") or "medium").strip().lower()
                    if str(raw.get("market_importance") or "medium").strip().lower() in {"high", "medium", "low"}
                    else "medium"
                ),
                "operational_centrality": (
                    str(raw.get("operational_centrality") or "meaningful").strip().lower()
                    if str(raw.get("operational_centrality") or "meaningful").strip().lower() in {"core", "meaningful", "peripheral"}
                    else "meaningful"
                ),
                "workflow_criticality": (
                    str(raw.get("workflow_criticality") or "medium").strip().lower()
                    if str(raw.get("workflow_criticality") or "medium").strip().lower() in {"high", "medium", "low"}
                    else "medium"
                ),
                "daily_operator_usage": (
                    str(raw.get("daily_operator_usage") or "medium").strip().lower()
                    if str(raw.get("daily_operator_usage") or "medium").strip().lower() in {"high", "medium", "low"}
                    else "medium"
                ),
                "switching_cost_intensity": (
                    str(raw.get("switching_cost_intensity") or "medium").strip().lower()
                    if str(raw.get("switching_cost_intensity") or "medium").strip().lower() in {"high", "medium", "low"}
                    else "medium"
                ),
                "priority_tier": (
                    str(raw.get("priority_tier") or "meaningful_adjacent").strip().lower()
                    if str(raw.get("priority_tier") or "meaningful_adjacent").strip().lower() in {"core_adjacent", "meaningful_adjacent", "edge_case"}
                    else "meaningful_adjacent"
                ),
            }
        )
        if len(normalized) >= max_items:
            break
    return normalized


def normalize_expansion_brief(expansion_brief: Any) -> dict[str, Any]:
    payload = expansion_brief if isinstance(expansion_brief, dict) else {}
    return {
        "reasoning_status": str(payload.get("reasoning_status") or "not_run"),
        "reasoning_warning": str(payload.get("reasoning_warning") or "").strip() or None,
        "reasoning_provider": str(payload.get("reasoning_provider") or "").strip() or None,
        "reasoning_model": str(payload.get("reasoning_model") or "").strip() or None,
        "confirmed_at": payload.get("confirmed_at"),
        "adjacent_capabilities": normalize_expansion_items(
            payload.get("adjacent_capabilities"),
            item_type="adjacent_capability",
            max_items=10,
        ),
        "adjacent_customer_segments": normalize_expansion_items(
            payload.get("adjacent_customer_segments"),
            item_type="adjacent_customer_segment",
            max_items=8,
        ),
        "named_account_anchors": normalize_expansion_items(
            payload.get("named_account_anchors"),
            item_type="named_account_anchor",
            max_items=8,
        ),
        "geography_expansions": normalize_expansion_items(
            payload.get("geography_expansions"),
            item_type="geography_expansion",
            max_items=8,
        ),
    }


def _supporting_node_ids_for_phrase(
    phrase: str,
    nodes: list[dict[str, Any]],
    *,
    layers: set[str],
    max_items: int = 3,
) -> list[str]:
    phrase_key = _normalize_phrase_key(phrase)
    supporting: list[str] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if str(node.get("layer") or "").strip() not in layers:
            continue
        node_phrase = str(node.get("phrase") or "").strip()
        if _normalize_phrase_key(node_phrase) != phrase_key:
            continue
        node_id = str(node.get("id") or "").strip()
        if node_id and node_id not in supporting:
            supporting.append(node_id)
        if len(supporting) >= max_items:
            break
    return supporting


def _derive_adjacent_nodes_from_expansion_inputs(
    *,
    expansion_inputs: list[dict[str, Any]],
    source_keys: set[str],
    layer: str,
    item_type: str,
    source_summary: str,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for expansion_input in expansion_inputs:
        if not isinstance(expansion_input, dict):
            continue
        comparator_name = str(expansion_input.get("name") or expansion_input.get("website") or "Expansion input").strip()
        comparator_url = normalize_url(expansion_input.get("website") or expansion_input.get("url") or "")
        for node in expansion_input.get("taxonomy_nodes") or []:
            if not isinstance(node, dict):
                continue
            if str(node.get("layer") or "").strip() != layer:
                continue
            label = _compact_phrase(node.get("phrase"), max_words=8, max_len=120)
            if not label:
                continue
            if item_type == "adjacent_capability" and not _is_high_quality_adjacent_capability_candidate(label):
                continue
            key = _normalize_phrase_key(label)
            if not key or key in source_keys:
                continue
            evidence_url = normalize_url(node.get("source_url") or comparator_url)
            entry = candidates.setdefault(
                key,
                {
                    "id": _stable_id("expansion", item_type, label),
                    "label": label,
                    "expansion_type": item_type,
                    "comparators": [],
                    "evidence_urls": [],
                },
            )
            if comparator_name not in entry["comparators"]:
                entry["comparators"].append(comparator_name)
            if evidence_url and evidence_url not in entry["evidence_urls"]:
                entry["evidence_urls"].append(evidence_url)

    results: list[dict[str, Any]] = []
    for entry in sorted(
        candidates.values(),
        key=lambda item: (-len(item.get("comparators") or []), str(item.get("label") or "")),
    ):
        comparator_count = len(entry.get("comparators") or [])
        status = "corroborated_expansion" if comparator_count >= 2 else "hypothesis"
        confidence = 0.78 if comparator_count >= 2 else 0.6
        if item_type == "adjacent_capability":
            why = (
                f"Comparator context extends the source capability neighborhood beyond {source_summary}. "
                f"Surfaced by {', '.join((entry.get('comparators') or [])[:2])}."
            )
        else:
            why = (
                f"Comparator context points to a neighboring buyer segment around the current source market box. "
                f"Surfaced by {', '.join((entry.get('comparators') or [])[:2])}."
            )
        results.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "expansion_type": item_type,
                "status": status,
                "confidence": confidence,
                "why_it_matters": why[:400],
                "evidence_urls": entry.get("evidence_urls") or [],
                "supporting_node_ids": [],
                "source_entity_names": entry.get("comparators") or [],
            }
        )
    return results


def _build_deterministic_expansion_brief(
    *,
    profile: CompanyProfile,
    sourcing_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
    expansion_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    source_capabilities = [
        str(item.get("phrase") or "").strip()
        for item in (sourcing_brief.get("capability_nodes") or [])
        if isinstance(item, dict) and str(item.get("phrase") or "").strip()
    ]
    source_customers = [
        str(item.get("phrase") or "").strip()
        for item in (sourcing_brief.get("customer_nodes") or [])
        if isinstance(item, dict) and str(item.get("phrase") or "").strip()
    ]
    source_summary = ", ".join(source_capabilities[:2]) or "the current source brief"
    source_capability_keys = {_normalize_phrase_key(item) for item in source_capabilities if _normalize_phrase_key(item)}
    source_customer_keys = {_normalize_phrase_key(item) for item in source_customers if _normalize_phrase_key(item)}

    adjacent_capabilities = _derive_adjacent_nodes_from_expansion_inputs(
        expansion_inputs=expansion_inputs,
        source_keys=source_capability_keys,
        layer="capability",
        item_type="adjacent_capability",
        source_summary=source_summary,
    )
    adjacent_customer_segments = _derive_adjacent_nodes_from_expansion_inputs(
        expansion_inputs=expansion_inputs,
        source_keys=source_customer_keys,
        layer="customer_archetype",
        item_type="adjacent_customer_segment",
        source_summary=source_summary,
    )

    named_account_anchors: list[dict[str, Any]] = []
    for item in sourcing_brief.get("named_customer_proof") or []:
        if not isinstance(item, dict):
            continue
        label = _compact_phrase(item.get("name"), max_words=6, max_len=120)
        if not label:
            continue
        if not _is_plausible_named_account_anchor(label):
            continue
        named_account_anchors.append(
            {
                "id": _stable_id("expansion", "named_account_anchor", label),
                "label": label,
                "expansion_type": "named_account_anchor",
                "status": "source_grounded",
                "confidence": _clamp_confidence(item.get("confidence"), default=0.84),
                "why_it_matters": (
                    "Named customer evidence provides a concrete account anchor for tracing peer institutions and nearby vendor footprints."
                ),
                "evidence_urls": [item.get("source_url")] if normalize_url(item.get("source_url")) else [],
                "supporting_node_ids": [],
                "source_entity_names": [label],
            }
        )

    geography_expansions: list[dict[str, Any]] = []
    geo_scope = profile.geo_scope or {}
    include_countries = _normalize_string_list(geo_scope.get("include_countries"), max_items=8, max_len=64)
    for country in include_countries:
        geography_expansions.append(
            {
                "id": _stable_id("expansion", "geography_expansion", country),
                "label": country,
                "expansion_type": "geography_expansion",
                "status": "source_grounded",
                "confidence": 0.92,
                "why_it_matters": "Explicit geographic scope set on the workspace and should be available for discovery review.",
                "evidence_urls": [],
                "supporting_node_ids": [],
                "source_entity_names": [],
            }
        )
    if not geography_expansions:
        region = str(geo_scope.get("region") or "").strip()
        if region:
            geography_expansions.append(
                {
                    "id": _stable_id("expansion", "geography_expansion", region),
                    "label": region,
                    "expansion_type": "geography_expansion",
                    "status": "source_grounded",
                    "confidence": 0.88,
                    "why_it_matters": "Workspace-level regional scope should be visible during scope review before universe discovery.",
                    "evidence_urls": [],
                    "supporting_node_ids": [],
                    "source_entity_names": [],
                }
            )

    adjacent_capabilities = [
        {
            **item,
            "supporting_node_ids": item.get("supporting_node_ids")
            or _supporting_node_ids_for_phrase(
                str(item.get("label") or ""),
                taxonomy_nodes,
                layers={"capability"},
            ),
        }
        for item in adjacent_capabilities
    ]
    adjacent_customer_segments = [
        {
            **item,
            "supporting_node_ids": item.get("supporting_node_ids")
            or _supporting_node_ids_for_phrase(
                str(item.get("label") or ""),
                taxonomy_nodes,
                layers={"customer_archetype"},
            ),
        }
        for item in adjacent_customer_segments
    ]

    return normalize_expansion_brief(
        {
            "reasoning_status": "not_run",
            "reasoning_warning": (
                "Expansion brief currently uses deterministic scaffolding from the source brief, expansion inputs, and workspace geo scope. "
                "Bounded model-based deep research can enrich this artifact next."
            ),
            "adjacent_capabilities": adjacent_capabilities,
            "adjacent_customer_segments": adjacent_customer_segments,
            "named_account_anchors": named_account_anchors,
            "geography_expansions": geography_expansions,
        }
    )


def _merge_reasoned_expansion_brief(
    *,
    response_text: str,
    fallback_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
    source_company: dict[str, Any],
    comparator_domains: set[str],
    reasoning_provider: Optional[str] = None,
    reasoning_model: Optional[str] = None,
) -> dict[str, Any]:
    parsed = _extract_json_object(response_text)
    if not parsed:
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": (
                "Expansion brief normalization returned invalid structured output. "
                "Showing deterministic fallback from the source brief, expansion inputs, and geo scope."
            ),
            "reasoning_provider": reasoning_provider,
            "reasoning_model": reasoning_model,
        }

    normalized = normalize_expansion_brief(parsed)
    fallback_normalized = normalize_expansion_brief(fallback_brief)
    for key in ("adjacent_capabilities", "adjacent_customer_segments"):
        normalized[key] = _merge_expansion_group(
            item_type=(
                "adjacent_capability"
                if key == "adjacent_capabilities"
                else "adjacent_customer_segment"
            ),
            fallback_items=list(fallback_normalized.get(key) or []),
            reasoned_items=list(normalized.get(key) or []),
            source_company=source_company,
            comparator_domains=comparator_domains,
            prefer_fallback=True,
        )
    for key in (
        "named_account_anchors",
        "geography_expansions",
    ):
        if normalized.get(key):
            continue
        if fallback_normalized.get(key):
            normalized[key] = fallback_normalized[key]
    for key, layers in (
        ("adjacent_capabilities", {"capability"}),
        ("adjacent_customer_segments", {"customer_archetype"}),
    ):
        enriched: list[dict[str, Any]] = []
        for item in normalized.get(key) or []:
            enriched.append(
                {
                    **item,
                    "supporting_node_ids": item.get("supporting_node_ids")
                    or _supporting_node_ids_for_phrase(
                        item.get("label"),
                        taxonomy_nodes,
                        layers=layers,
                    ),
                }
            )
        normalized[key] = enriched
    return {
        **normalized,
        "reasoning_status": str(parsed.get("reasoning_status") or "success"),
        "reasoning_warning": str(parsed.get("reasoning_warning") or "").strip() or None,
        "reasoning_provider": reasoning_provider,
        "reasoning_model": reasoning_model,
    }


def _reason_expansion_brief(
    *,
    profile: CompanyProfile,
    sourcing_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
    expansion_inputs: list[dict[str, Any]],
    fallback_brief: dict[str, Any],
) -> dict[str, Any]:
    if not sourcing_brief.get("capability_nodes") and not sourcing_brief.get("customer_nodes"):
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "not_applicable",
            "reasoning_warning": "Expansion reasoning did not run because the source brief is too thin to bound adjacency research.",
            "reasoning_provider": None,
            "reasoning_model": None,
        }
    settings = get_settings()
    if not any([settings.gemini_api_key, settings.openai_api_key, settings.anthropic_api_key]):
        return normalize_expansion_brief(fallback_brief)

    research_payload = _compact_expansion_research_payload(
        company_name=str(((sourcing_brief.get("source_company") or {}).get("name")) or _domain_brand_name(profile.buyer_company_url or "") or "Source Company"),
        company_url=normalize_url(profile.buyer_company_url) if profile.buyer_company_url else None,
        sourcing_brief=sourcing_brief,
        taxonomy_nodes=taxonomy_nodes,
        expansion_inputs=expansion_inputs,
        geo_scope=profile.geo_scope or {},
        fallback_brief=fallback_brief,
    )

    try:
        research_response = LLMOrchestrator().run_stage(
            LLMRequest(
                stage=LLMStage.expansion_brief_reasoning,
                prompt=_expansion_research_prompt(research_payload),
                timeout_seconds=90,
                use_web_search=True,
                expect_json=True,
                metadata={"company_name": research_payload["source_company"]["name"], "phase": "expansion_research"},
            )
        )
    except LLMOrchestrationError as exc:
        attempt_error = ""
        if getattr(exc, "attempts", None):
            last_attempt = exc.attempts[-1]
            attempt_error = str(getattr(last_attempt, "error_message", "") or "").strip()
        warning = (
            "Expansion research failed after model retries. "
            "Showing deterministic fallback from the source brief, expansion inputs, and geo scope."
        )
        if attempt_error:
            warning = f"{warning} Last error: {attempt_error[:240]}"
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": warning,
            "reasoning_provider": None,
            "reasoning_model": None,
        }
    except Exception as exc:
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": (
                "Expansion research failed unexpectedly. Showing deterministic fallback from the source brief, expansion inputs, and geo scope. "
                f"Last error: {str(exc)[:240]}"
            ),
            "reasoning_provider": None,
            "reasoning_model": None,
        }

    research_json = _extract_json_object(research_response.text)
    if not research_json:
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": (
                "Expansion research returned invalid structured output. "
                "Showing deterministic fallback from the source brief, expansion inputs, and geo scope."
            ),
            "reasoning_provider": getattr(research_response, "provider", None),
            "reasoning_model": getattr(research_response, "model", None),
        }

    try:
        normalization_response = LLMOrchestrator().run_stage(
            LLMRequest(
                stage=LLMStage.structured_normalization,
                prompt=_expansion_normalization_prompt(
                    company_name=research_payload["source_company"]["name"],
                    research_payload=research_json,
                    fallback_brief=fallback_brief,
                ),
                timeout_seconds=60,
                use_web_search=False,
                expect_json=True,
                metadata={"company_name": research_payload["source_company"]["name"], "phase": "expansion_normalization"},
            )
        )
    except LLMOrchestrationError as exc:
        attempt_error = ""
        if getattr(exc, "attempts", None):
            last_attempt = exc.attempts[-1]
            attempt_error = str(getattr(last_attempt, "error_message", "") or "").strip()
        warning = (
            "Expansion normalization failed after model retries. "
            "Showing deterministic fallback from the source brief, expansion inputs, and geo scope."
        )
        if attempt_error:
            warning = f"{warning} Last error: {attempt_error[:240]}"
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": warning,
            "reasoning_provider": getattr(research_response, "provider", None),
            "reasoning_model": getattr(research_response, "model", None),
        }
    except Exception as exc:
        return {
            **normalize_expansion_brief(fallback_brief),
            "reasoning_status": "degraded",
            "reasoning_warning": (
                "Expansion normalization failed unexpectedly. Showing deterministic fallback from the source brief, expansion inputs, and geo scope. "
                f"Last error: {str(exc)[:240]}"
            ),
            "reasoning_provider": getattr(research_response, "provider", None),
            "reasoning_model": getattr(research_response, "model", None),
        }

    return _merge_reasoned_expansion_brief(
        response_text=normalization_response.text,
        fallback_brief=fallback_brief,
        taxonomy_nodes=taxonomy_nodes,
        source_company=(sourcing_brief.get("source_company") or {}),
        comparator_domains={
            normalize_domain(item.get("website") or item.get("url"))
            for item in expansion_inputs
            if isinstance(item, dict) and normalize_domain(item.get("website") or item.get("url"))
        },
        reasoning_provider=getattr(research_response, "provider", None),
        reasoning_model=getattr(research_response, "model", None),
    )


def build_expansion_brief(
    *,
    profile: CompanyProfile,
    sourcing_brief: dict[str, Any],
    taxonomy_nodes: list[dict[str, Any]],
    expansion_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    fallback_brief = _build_deterministic_expansion_brief(
        profile=profile,
        sourcing_brief=sourcing_brief,
        taxonomy_nodes=taxonomy_nodes,
        expansion_inputs=expansion_inputs,
    )
    return _reason_expansion_brief(
        profile=profile,
        sourcing_brief=sourcing_brief,
        taxonomy_nodes=taxonomy_nodes,
        expansion_inputs=expansion_inputs,
        fallback_brief=fallback_brief,
    )


def _sanitize_reasoned_open_questions(open_questions: Any) -> list[str]:
    sanitized: list[str] = []
    seen: set[str] = set()
    for question in normalize_open_questions(open_questions):
        lowered = question.lower()
        if any(term in lowered for term in OPEN_QUESTION_BLOCKLIST_TERMS):
            continue
        if "which customer segment" in lowered and (
            "priorit" in lowered or "growth" in lowered or "strateg" in lowered
        ):
            continue
        key = _normalize_phrase_key(question)
        if key in seen:
            continue
        seen.add(key)
        sanitized.append(question)
        if len(sanitized) >= 4:
            break
    return sanitized


def _extract_context_text(profile: CompanyProfile) -> str:
    parts: list[str] = []
    comparator_seed_summaries = (
        list((profile.comparator_seed_summaries or {}).values())
        if isinstance(profile.comparator_seed_summaries, dict)
        else []
    )
    for value in [
        profile.context_pack_markdown,
        *comparator_seed_summaries,
    ]:
        text = str(value or "").strip()
        if text:
            parts.append(text)
    return "\n".join(parts)


def _site_has_meaningful_content(site: Any) -> bool:
    if not isinstance(site, dict):
        return False
    if str(site.get("summary") or "").strip():
        return True
    if any(isinstance(signal, dict) and str(signal.get("value") or "").strip() for signal in site.get("signals") or []):
        return True
    if any(
        isinstance(customer, dict)
        and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
        for customer in site.get("customer_evidence") or []
    ):
        return True
    for page in site.get("pages") or []:
        if not isinstance(page, dict):
            continue
        if str(page.get("raw_content") or "").strip():
            return True
        if any(isinstance(block, dict) and str(block.get("content") or "").strip() for block in page.get("blocks") or []):
            return True
        if any(isinstance(signal, dict) and str(signal.get("value") or "").strip() for signal in page.get("signals") or []):
            return True
        if any(
            isinstance(customer, dict)
            and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
            for customer in page.get("customer_evidence") or []
        ):
            return True
    return False


def _extract_site_context_text(site: Any, *, max_chars: int = 12000) -> str:
    if not isinstance(site, dict):
        return ""

    parts: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text:
            parts.append(text)

    add(site.get("company_name"))
    add(site.get("summary"))

    for signal in site.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        add(signal.get("value"))
        add(signal.get("snippet"))

    for customer in site.get("customer_evidence") or []:
        if not isinstance(customer, dict):
            continue
        name = str(customer.get("name") or "").strip()
        context = str(customer.get("context") or "").strip()
        if name and context:
            add(f"{name}: {context}")
        else:
            add(name or context)

    for page in site.get("pages") or []:
        if not isinstance(page, dict):
            continue
        add(page.get("title"))
        add(page.get("raw_content"))
        for block in page.get("blocks") or []:
            if isinstance(block, dict):
                add(block.get("content"))
        for signal in page.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            add(signal.get("value"))
            add(signal.get("snippet"))
        for customer in page.get("customer_evidence") or []:
            if not isinstance(customer, dict):
                continue
            name = str(customer.get("name") or "").strip()
            context = str(customer.get("context") or "").strip()
            if name and context:
                add(f"{name}: {context}")
            else:
                add(name or context)

    return "\n".join(parts)[:max_chars]


def _identity_tokens_from_url(url: Any) -> set[str]:
    normalized = normalize_domain(normalize_url(url))
    if not normalized:
        return set()
    tokens = {normalized.lower()}
    first_label = normalized.split(".")[0].strip().lower()
    if first_label:
        tokens.add(first_label)
    return {token for token in tokens if token}


def _identity_tokens_from_name(value: Any) -> set[str]:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not cleaned:
        return set()
    tokens = {part for part in cleaned.split() if len(part) >= 3}
    if cleaned.replace(" ", ""):
        tokens.add(cleaned.replace(" ", ""))
    return tokens


def _source_company_identity_keys(source_company: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    if not isinstance(source_company, dict):
        return keys
    keys.update(_identity_tokens_from_name(source_company.get("name")))
    keys.update(_identity_tokens_from_url(source_company.get("website")))
    return {key for key in keys if key}


def _is_subject_only_expansion_item(
    item: dict[str, Any],
    *,
    source_company: dict[str, Any],
    comparator_domains: set[str],
) -> bool:
    if not isinstance(item, dict):
        return False
    source_keys = _source_company_identity_keys(source_company)
    if not source_keys:
        return False

    evidence_domains = {
        normalize_domain(url)
        for url in _normalize_string_list(item.get("evidence_urls"), max_items=8, max_len=500)
        if normalize_domain(url)
    }
    if any(domain in comparator_domains for domain in evidence_domains):
        return False

    raw_entity_names = _normalize_string_list(item.get("source_entity_names"), max_items=6, max_len=120)
    if not raw_entity_names:
        return False

    entity_name_keys = [_identity_tokens_from_name(name) for name in raw_entity_names]
    entity_name_keys = [tokens for tokens in entity_name_keys if tokens]
    if not entity_name_keys:
        return False

    return all(tokens & source_keys for tokens in entity_name_keys)


def _merge_expansion_group(
    *,
    item_type: str,
    fallback_items: list[dict[str, Any]],
    reasoned_items: list[dict[str, Any]],
    source_company: dict[str, Any],
    comparator_domains: set[str],
    prefer_fallback: bool = False,
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    filtered_reasoned = [
        item
        for item in reasoned_items
        if (
            item_type != "adjacent_capability"
            or _is_high_quality_adjacent_capability_candidate(item.get("label"))
        )
        if not _is_subject_only_expansion_item(
            item,
            source_company=source_company,
            comparator_domains=comparator_domains,
        )
    ]

    groups = [fallback_items, filtered_reasoned] if prefer_fallback else [filtered_reasoned, fallback_items]
    for group in groups:
        for item in group:
            if not isinstance(item, dict):
                continue
            key = _normalize_phrase_key(item.get("label"))
            if not key:
                continue
            existing = by_key.get(key)
            if existing is None:
                copy = dict(item)
                by_key[key] = copy
                ordered.append(copy)
                continue
            existing["evidence_urls"] = _normalize_string_list(
                list(existing.get("evidence_urls") or []) + list(item.get("evidence_urls") or []),
                max_items=6,
                max_len=500,
            )
            existing["source_entity_names"] = _normalize_string_list(
                list(existing.get("source_entity_names") or []) + list(item.get("source_entity_names") or []),
                max_items=6,
                max_len=120,
            )
            existing["supporting_node_ids"] = _normalize_string_list(
                list(existing.get("supporting_node_ids") or []) + list(item.get("supporting_node_ids") or []),
                max_items=8,
                max_len=120,
            )
            if not str(existing.get("why_it_matters") or "").strip() and str(item.get("why_it_matters") or "").strip():
                existing["why_it_matters"] = item.get("why_it_matters")
            existing["confidence"] = max(
                _clamp_confidence(existing.get("confidence"), default=0.6),
                _clamp_confidence(item.get("confidence"), default=0.6),
            )
    return ordered


def _resolve_buyer_site(profile: CompanyProfile) -> dict[str, Any]:
    buyer_url = normalize_url(profile.buyer_company_url)
    buyer_domain = normalize_domain(buyer_url)
    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        site_domain = normalize_domain(site.get("url"))
        if buyer_domain and site_domain == buyer_domain:
            return site
    if isinstance(context_pack.get("sites"), list) and context_pack.get("sites"):
        first_site = context_pack.get("sites")[0]
        if isinstance(first_site, dict):
            return first_site
    return {}


def assess_buyer_evidence(profile: CompanyProfile) -> dict[str, Any]:
    buyer_url = normalize_url(profile.buyer_company_url)
    if not buyer_url:
        return {
            "mode": "source_company_missing",
            "status": "not_applicable",
            "score": 0,
            "used_for_inference": False,
            "warning": None,
            "metrics": {
                "pages_crawled": 0,
                "content_pages": 0,
                "signal_count": 0,
                "customer_evidence_count": 0,
                "summary_chars": 0,
            },
        }

    buyer_site = _resolve_buyer_site(profile)
    pages = buyer_site.get("pages") or []
    pages_crawled = len([page for page in pages if isinstance(page, dict)])
    content_pages = 0
    page_signal_count = 0
    page_customer_count = 0
    for page in pages:
        if not isinstance(page, dict):
            continue
        has_page_content = bool(
            str(page.get("raw_content") or "").strip()
            or any(
                isinstance(block, dict) and str(block.get("content") or "").strip()
                for block in page.get("blocks") or []
            )
            or any(
                isinstance(signal, dict) and str(signal.get("value") or "").strip()
                for signal in page.get("signals") or []
            )
            or any(
                isinstance(customer, dict)
                and (str(customer.get("name") or "").strip() or str(customer.get("context") or "").strip())
                for customer in page.get("customer_evidence") or []
            )
        )
        if has_page_content:
            content_pages += 1
        page_signal_count += len([signal for signal in page.get("signals") or [] if isinstance(signal, dict)])
        page_customer_count += len([customer for customer in page.get("customer_evidence") or [] if isinstance(customer, dict)])

    top_level_signal_count = len([signal for signal in buyer_site.get("signals") or [] if isinstance(signal, dict)])
    top_level_customer_count = len([customer for customer in buyer_site.get("customer_evidence") or [] if isinstance(customer, dict)])
    signal_count = top_level_signal_count + page_signal_count
    customer_evidence_count = top_level_customer_count + page_customer_count
    summary_chars = len(str(buyer_site.get("summary") or "").strip())

    score = 0
    if summary_chars >= 120:
        score += 2
    elif summary_chars >= 40:
        score += 1
    score += min(content_pages, 2)
    score += min(signal_count, 2)
    score += min(customer_evidence_count, 2)

    sufficient = score >= BUYER_EVIDENCE_MIN_SCORE
    return {
        "mode": "buyer_website",
        "status": "sufficient" if sufficient else "insufficient",
        "score": score,
        "used_for_inference": sufficient,
        "warning": (
            None
            if sufficient
            else (
                "Buyer evidence is too weak for reliable inference. Add first-party product pages, PDFs, "
                "case studies, or supporting evidence before trusting customer, capability, and deployment claims."
            )
        ),
        "metrics": {
            "pages_crawled": pages_crawled,
            "content_pages": content_pages,
            "signal_count": signal_count,
            "customer_evidence_count": customer_evidence_count,
            "summary_chars": summary_chars,
        },
    }


def _sanitize_focus_phrase(value: str) -> str:
    cleaned = re.sub(r"\([^)]*\)", "", str(value or "")).strip(" ,.;:-")
    cleaned = cleaned.split("(", 1)[0].strip(" ,.;:-")
    cleaned = re.split(
        r"\b(?:as|with|that|which|who|where|when|and\s+they|and\s+that|but)\b",
        cleaned,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,.;:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    words = cleaned.split()
    return " ".join(words[:8]).strip()


def _derive_source_pills_from_profile(profile: CompanyProfile) -> list[dict[str, Any]]:
    source_items: list[dict[str, Any]] = []
    for url in [profile.buyer_company_url, *(profile.comparator_seed_urls or []), *(profile.supporting_evidence_urls or [])]:
        if normalize_url(url):
            source_items.append({"url": url})

    context_pack = profile.context_pack_json if isinstance(profile.context_pack_json, dict) else {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        if normalize_url(site.get("url")):
            source_items.append(
                {
                    "url": site.get("url"),
                    "label": str(site.get("company_name") or normalize_domain(site.get("url")) or "source"),
                }
            )
        for signal in site.get("signals") or []:
            if not isinstance(signal, dict):
                continue
            if normalize_url(signal.get("source_url")):
                source_items.append({"url": signal.get("source_url")})
        for page in site.get("pages") or []:
            if not isinstance(page, dict):
                continue
            if normalize_url(page.get("url")):
                source_items.append({"url": page.get("url"), "label": page.get("title")})
            for signal in page.get("signals") or []:
                if not isinstance(signal, dict):
                    continue
                if normalize_url(signal.get("source_url")):
                    source_items.append({"url": signal.get("source_url")})
            for evidence in page.get("customer_evidence") or []:
                if not isinstance(evidence, dict):
                    continue
                if normalize_url(evidence.get("source_url")):
                    source_items.append({"url": evidence.get("source_url")})

    return normalize_source_pills(
        source_items,
        buyer_url=profile.buyer_company_url,
        comparator_seed_urls=profile.comparator_seed_urls or [],
        supporting_evidence_urls=profile.supporting_evidence_urls or [],
    )

def build_expansion_inputs(
    context_pack_v2: dict[str, Any],
    *,
    comparator_seed_urls: Iterable[str],
    buyer_url: Any,
) -> list[dict[str, Any]]:
    def _candidate_domains_from_sites() -> list[str]:
        domains: list[str] = []
        for site in context_pack_v2.get("sites") or []:
            if not isinstance(site, dict):
                continue
            comparator_domain = normalize_domain(site.get("website") or site.get("url"))
            if comparator_domain:
                domains.append(comparator_domain)
        return domains

    def _build_from_domains(candidate_domains: list[str]) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        seen_domains: set[str] = set()
        seen_entities: set[str] = set()
        for comparator_domain in candidate_domains:
            if not comparator_domain or comparator_domain == buyer_domain or comparator_domain in seen_domains:
                continue
            seen_domains.add(comparator_domain)
            scoped_pack = _domain_scoped_context_pack(context_pack_v2, domain=comparator_domain)
            scoped_sites = []
            for site in scoped_pack.get("sites") or []:
                if not isinstance(site, dict):
                    continue
                site_domain = normalize_domain(site.get("website") or site.get("url"))
                if (
                    site_domain
                    and (
                        site_domain == comparator_domain
                        or site_domain.endswith(f".{comparator_domain}")
                        or comparator_domain.endswith(f".{site_domain}")
                    )
                ):
                    scoped_sites.append(site)
            if not scoped_sites:
                continue
            taxonomy_nodes, taxonomy_edges = _build_taxonomy_map(scoped_pack)
            nodes_by_layer = {
                layer: [node for node in taxonomy_nodes if node.get("layer") == layer]
                for layer in TAXONOMY_LAYERS
            }
            named_customers = [
                item
                for item in (scoped_pack.get("named_customers") or [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ]
            partner_integrations = [
                item
                for item in (scoped_pack.get("integrations") or [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ]
            primary_site = scoped_sites[0]
            raw_name = str(primary_site.get("company_name") or "").strip()
            domain_name = _domain_brand_name(primary_site.get("website") or primary_site.get("url") or comparator_domain)
            comparator_name = raw_name or domain_name or comparator_domain
            if not raw_name or not _is_plausible_comparator_name(raw_name):
                comparator_name = domain_name or comparator_name
            entity_key = _normalize_phrase_key(comparator_name)
            if entity_key and entity_key in seen_entities:
                continue
            if entity_key:
                seen_entities.add(entity_key)
            page_capability_nodes = _derive_comparator_page_capability_nodes(
                scoped_sites=scoped_sites,
                comparator_name=comparator_name,
            )
            if page_capability_nodes:
                taxonomy_nodes = normalize_taxonomy_nodes([*taxonomy_nodes, *page_capability_nodes])
                nodes_by_layer = {
                    layer: [node for node in taxonomy_nodes if node.get("layer") == layer]
                    for layer in TAXONOMY_LAYERS
                }
            contexts.append(
                {
                    "name": comparator_name,
                    "website": str(primary_site.get("website") or primary_site.get("url") or comparator_domain),
                    "source_summary": _generate_sourcing_brief_summary(
                        comparator_name,
                        capabilities=nodes_by_layer.get("capability", []),
                        workflows=nodes_by_layer.get("workflow", []),
                        customers=nodes_by_layer.get("customer_archetype", []),
                        named_customers=named_customers,
                    ),
                    "taxonomy_nodes": taxonomy_nodes,
                    "taxonomy_edges": taxonomy_edges,
                    "named_customer_proof": named_customers[:12],
                    "partner_integration_proof": partner_integrations[:12],
                    "crawl_coverage": scoped_pack.get("crawl_coverage") or {},
                }
            )
        return contexts

    buyer_domain = normalize_domain(buyer_url)
    candidate_domains: list[str] = []
    for seed_url in comparator_seed_urls or []:
        comparator_domain = normalize_domain(seed_url)
        if comparator_domain:
            candidate_domains.append(comparator_domain)
    if not candidate_domains:
        candidate_domains = _candidate_domains_from_sites()

    expansion_inputs = _build_from_domains(candidate_domains)
    if expansion_inputs or not comparator_seed_urls:
        return expansion_inputs

    return _build_from_domains(_candidate_domains_from_sites())


def build_company_context_artifacts(
    profile: CompanyProfile,
    *,
    override_nodes: Any = None,
    source_summary_override: Any = None,
    open_questions_override: Any = None,
    confirmed_at: datetime | None = None,
) -> dict[str, Any]:
    full_context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    source_pills = _derive_source_pills_from_profile(profile)
    buyer_evidence = assess_buyer_evidence(profile)
    (
        context_pack_v2,
        taxonomy_nodes,
        taxonomy_edges,
        lens_seeds,
        sourcing_brief_open_questions,
        sourcing_brief,
    ) = _build_sourcing_brief_artifacts(
        profile,
        source_pills=source_pills,
        override_nodes=override_nodes,
    )
    source_summary = str(
        source_summary_override
        or ((sourcing_brief.get("source_summary") if isinstance(sourcing_brief, dict) else None) or "")
        or (
            "Buyer website crawled, but no first-party product or customer evidence was extracted yet. "
            "Add supporting evidence or regenerate after improving the crawl target."
            if profile.buyer_company_url
            else "Source company context is still too thin to generate a sourcing brief."
        )
    ).strip()[:8000]

    final_open_questions = normalize_open_questions(
        open_questions_override if open_questions_override is not None else sourcing_brief.get("open_questions")
    )
    if not final_open_questions:
        final_open_questions = normalize_open_questions(sourcing_brief_open_questions)
    if buyer_evidence.get("status") == "insufficient" and buyer_evidence.get("warning"):
        warning = str(buyer_evidence.get("warning"))
        if warning not in final_open_questions:
            final_open_questions.insert(0, warning)
    final_open_questions = normalize_open_questions(final_open_questions)[:8]

    sourcing_brief = {
        **(sourcing_brief if isinstance(sourcing_brief, dict) else {}),
        "source_summary": source_summary,
        "open_questions": final_open_questions,
        "confirmed_at": confirmed_at.isoformat() if confirmed_at else None,
    }
    expansion_inputs = build_expansion_inputs(
        full_context_pack_v2,
        comparator_seed_urls=profile.comparator_seed_urls or [],
        buyer_url=profile.buyer_company_url,
    )

    return {
        "source_pills": source_pills,
        "buyer_evidence": buyer_evidence,
        "context_pack_v2": context_pack_v2,
        "taxonomy_nodes": taxonomy_nodes,
        "taxonomy_edges": taxonomy_edges,
        "lens_seeds": lens_seeds,
        "sourcing_brief": sourcing_brief,
        "expansion_inputs": expansion_inputs,
        "generated_at": datetime.utcnow(),
        "confirmed_at": confirmed_at,
    }


def build_expansion_artifacts(
    profile: CompanyProfile,
    *,
    sourcing_brief: Any,
    taxonomy_nodes: Any,
    confirmed_at: datetime | None = None,
) -> dict[str, Any]:
    full_context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    normalized_sourcing_brief = sourcing_brief if isinstance(sourcing_brief, dict) else {}
    normalized_taxonomy_nodes = normalize_taxonomy_nodes(taxonomy_nodes or [])
    expansion_inputs = build_expansion_inputs(
        full_context_pack_v2,
        comparator_seed_urls=profile.comparator_seed_urls or [],
        buyer_url=profile.buyer_company_url,
    )
    expansion_brief = build_expansion_brief(
        profile=profile,
        sourcing_brief=normalized_sourcing_brief,
        taxonomy_nodes=normalized_taxonomy_nodes,
        expansion_inputs=expansion_inputs,
    )
    if confirmed_at:
        expansion_brief = {
            **normalize_expansion_brief(expansion_brief),
            "confirmed_at": confirmed_at.isoformat(),
        }
    return {
        "expansion_inputs": expansion_inputs,
        "expansion_brief": normalize_expansion_brief(expansion_brief),
        "generated_at": datetime.utcnow(),
    }


def _pretty_source_host_label(host: str) -> str:
    host = host.removeprefix("www.")
    special_cases = {
        "play.google.com": "Google Play",
        "bcorporation.net": "B Corporation",
        "rothschildandco.com": "Rothschild & Co",
    }
    if host in special_cases:
        return special_cases[host]
    brand = host.split(".", 1)[0]
    if not brand:
        return "Source"
    tokens = [token for token in re.split(r"[-_]+", brand) if token]
    normalized_tokens: list[str] = []
    for token in tokens:
        if any(ch.isdigit() for ch in token):
            normalized_tokens.append("".join(ch.upper() if ch.isalpha() else ch for ch in token))
        elif len(token) <= 4:
            normalized_tokens.append(token.upper())
        else:
            normalized_tokens.append(token.capitalize())
    pretty = " ".join(normalized_tokens).strip()
    return pretty or "Source"


def _report_source_priority(source: dict[str, Any], *, preferred_host: str | None = None) -> tuple[int, int, int, str]:
    host = normalize_domain(source.get("url"))
    source_tier = str(source.get("source_tier") or "").strip().lower()
    publisher_channel = str(source.get("publisher_channel") or "").strip().lower()
    source_kind = str(source.get("source_kind") or "").strip().lower()
    is_preferred_host = bool(preferred_host and host == preferred_host)
    is_primary = source_tier == "primary"
    is_first_party = publisher_channel in {"primary", "company_website", "product", "solutions", "services", "docs"}
    is_taxonomy_or_evidence = source_kind in {"taxonomy_evidence", "source_document", "evidence_item"}
    return (
        0 if is_preferred_host and is_primary else 1,
        0 if is_primary and is_first_party else 1,
        0 if is_taxonomy_or_evidence else 1,
        str(source.get("label") or "").strip().lower(),
    )


def _prioritize_report_sources(
    sources: list[dict[str, Any]],
    *,
    preferred_host: str | None = None,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    deduped = _dedupe_report_sources(sources)
    deduped.sort(key=lambda item: _report_source_priority(item, preferred_host=preferred_host))
    return deduped[:max_items]


def _normalize_report_label_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if re.fullmatch(r"(?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}(?:/.*)?", text.lower()):
        host = normalize_domain(text)
        if host:
            return _pretty_source_host_label(host)
    return text


def _report_label_path_suffix(url: str) -> str:
    path = urlparse(normalize_url(url)).path.strip("/")
    if not path:
        return ""
    if len(path) > 48:
        path = path[:48].rstrip("/")
    return path


def _report_source_label(url: str, *, publisher: Any = None, fallback: Any = None) -> str:
    publisher_text = _normalize_report_label_text(publisher)
    if publisher_text:
        suffix = _report_label_path_suffix(url)
        if suffix:
            return f"{publisher_text} / {suffix}"[:120]
        return publisher_text[:120]
    fallback_text = _normalize_report_label_text(fallback)
    if fallback_text:
        return fallback_text[:120]
    host = normalize_domain(url)
    if host:
        pretty = _pretty_source_host_label(host)
        if pretty:
            return pretty[:120]
    return "Source"


def _dedupe_report_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source in sources:
        if not isinstance(source, dict):
            continue
        key = (
            str(source.get("url") or "").strip(),
            str(source.get("publisher") or "").strip(),
            str(source.get("label") or "").strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(source)
    return deduped


def _build_source_document_lookup(
    source_documents: Any,
    context_pack_v2: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    documents_by_url: dict[str, dict[str, Any]] = {}
    evidence_by_id: dict[str, dict[str, Any]] = {}

    for item in source_documents or []:
        if not isinstance(item, dict):
            continue
        url = normalize_url(item.get("url"))
        if not url:
            continue
        documents_by_url[url] = {
            "id": str(item.get("id") or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]),
            "label": _report_source_label(url, publisher=item.get("publisher"), fallback=item.get("name")),
            "url": url,
            "publisher": str(item.get("publisher") or "").strip() or None,
            "publisher_channel": str(item.get("publisher_channel") or "company_website").strip() or "company_website",
            "publisher_type": str(item.get("publisher_type") or "").strip() or None,
            "source_tier": str(item.get("evidence_tier") or "primary").strip() or "primary",
            "source_kind": str(item.get("evidence_type") or "source_document").strip() or "source_document",
            "evidence_type": str(item.get("evidence_type") or "source_document").strip() or "source_document",
            "claim_scope": str(item.get("claim_scope") or "").strip() or None,
            "published_at": None,
            "captured_at": None,
        }

    def _register_evidence_item(item: Any) -> None:
        if not isinstance(item, dict):
            return
        evidence_id = str(item.get("id") or "").strip()
        url = normalize_url(item.get("url"))
        page_type = str(item.get("page_type") or "company_website").strip() or "company_website"
        publisher = _domain_brand_name(url) if page_type == "site_summary" else None
        fallback_label = publisher or item.get("page_title") or item.get("text")
        if url and url not in documents_by_url:
            documents_by_url[url] = {
                "id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:12],
                "label": _report_source_label(url, publisher=publisher, fallback=fallback_label),
                "url": url,
                "publisher": publisher,
                "publisher_channel": page_type,
                "publisher_type": None,
                "source_tier": "primary",
                "source_kind": str(item.get("kind") or "evidence_item").strip() or "evidence_item",
                "evidence_type": str(item.get("kind") or "evidence_item").strip() or "evidence_item",
                "claim_scope": "about_subject_company",
                "published_at": None,
                "captured_at": item.get("captured_at"),
            }
        if evidence_id and url:
            evidence_by_id[evidence_id] = documents_by_url[url]

    for item in (context_pack_v2 or {}).get("evidence_items") or []:
        _register_evidence_item(item)
    for site in (context_pack_v2 or {}).get("sites") or []:
        if not isinstance(site, dict):
            continue
        for item in site.get("evidence_items") or []:
            _register_evidence_item(item)

    return documents_by_url, evidence_by_id


def _register_report_source(
    source_order: list[str],
    sources_by_id: dict[str, dict[str, Any]],
    source: dict[str, Any],
) -> str:
    source_id = str(source.get("id") or "").strip() or hashlib.sha1(
        f"{source.get('url') or ''}|{source.get('label') or ''}".encode("utf-8")
    ).hexdigest()[:12]
    if source_id not in sources_by_id:
        sources_by_id[source_id] = {
            "id": source_id,
            "label": str(source.get("label") or "Source").strip()[:120] or "Source",
            "url": str(source.get("url") or "").strip(),
            "publisher": source.get("publisher"),
            "publisher_channel": str(source.get("publisher_channel") or "company_website").strip() or "company_website",
            "publisher_type": source.get("publisher_type"),
            "source_tier": str(source.get("source_tier") or "primary").strip() or "primary",
            "source_kind": str(source.get("source_kind") or "source_document").strip() or "source_document",
            "evidence_type": str(source.get("evidence_type") or "source_document").strip() or "source_document",
            "claim_scope": source.get("claim_scope"),
            "published_at": source.get("published_at"),
            "captured_at": source.get("captured_at"),
        }
        source_order.append(source_id)
    return source_id


def _resolve_report_sources(
    *,
    evidence_ids: Any = None,
    urls: Any = None,
    source_documents_by_url: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    fallback_label: Any = None,
    publisher: Any = None,
    publisher_channel: Any = None,
    publisher_type: Any = None,
    source_tier: Any = None,
    source_kind: Any = None,
    evidence_type: Any = None,
    claim_scope: Any = None,
    published_at: Any = None,
    captured_at: Any = None,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for evidence_id in evidence_ids or []:
        key = str(evidence_id or "").strip()
        source = evidence_by_id.get(key)
        if not source:
            continue
        source_id = str(source.get("id") or "").strip()
        if source_id and source_id in seen:
            continue
        if source_id:
            seen.add(source_id)
        resolved.append(deepcopy(source))

    for url in urls or []:
        normalized = normalize_url(url)
        if not normalized:
            continue
        source = source_documents_by_url.get(normalized)
        if source:
            source_id = str(source.get("id") or "").strip()
            if source_id and source_id in seen:
                continue
            if source_id:
                seen.add(source_id)
            resolved.append(deepcopy(source))
            continue
        fallback_source = {
            "id": hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12],
            "label": _report_source_label(normalized, publisher=publisher, fallback=fallback_label),
            "url": normalized,
            "publisher": str(publisher or "").strip() or None,
            "publisher_channel": str(publisher_channel or "web").strip() or "web",
            "publisher_type": str(publisher_type or "").strip() or None,
            "source_tier": str(source_tier or "secondary").strip() or "secondary",
            "source_kind": str(source_kind or "web_source").strip() or "web_source",
            "evidence_type": str(evidence_type or "web_source").strip() or "web_source",
            "claim_scope": str(claim_scope or "").strip() or None,
            "published_at": published_at,
            "captured_at": captured_at,
        }
        if fallback_source["id"] in seen:
            continue
        seen.add(fallback_source["id"])
        resolved.append(fallback_source)

    return resolved


def _report_sentence(
    sentence_id: str,
    text: str,
    sources: list[dict[str, Any]],
    *,
    source_order: list[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    citation_pill_ids: list[str] = []
    seen_pill_ids: set[str] = set()
    for source in _dedupe_report_sources(sources):
        if not isinstance(source, dict) or not str(source.get("url") or "").strip():
            continue
        pill_id = _register_report_source(source_order, sources_by_id, source)
        if pill_id in seen_pill_ids:
            continue
        seen_pill_ids.add(pill_id)
        citation_pill_ids.append(pill_id)
    return {
        "id": sentence_id,
        "text": str(text or "").strip(),
        "citation_pill_ids": citation_pill_ids,
    }


def _paragraph_block(
    block_id: str,
    text: str,
    sources: list[dict[str, Any]],
    *,
    source_order: list[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "paragraph",
        "sentences": [
            _report_sentence(block_id, text, sources, source_order=source_order, sources_by_id=sources_by_id)
        ],
    }


def _bullet_list_block(
    block_id: str,
    items: list[tuple[str, list[dict[str, Any]]]],
    *,
    source_order: list[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "bullet_list",
        "items": [
            _report_sentence(
                f"{block_id}_{idx}",
                text,
                item_sources,
                source_order=source_order,
                sources_by_id=sources_by_id,
            )
            for idx, (text, item_sources) in enumerate(items, start=1)
            if str(text or "").strip()
        ],
    }


def _callout_block(
    block_id: str,
    tone: str,
    title: str | None,
    items: list[tuple[str, list[dict[str, Any]]]],
    *,
    source_order: list[str],
    sources_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "callout",
        "tone": tone,
        "title": title,
        "sentences": [
            _report_sentence(
                f"{block_id}_{idx}",
                text,
                item_sources,
                source_order=source_order,
                sources_by_id=sources_by_id,
            )
            for idx, (text, item_sources) in enumerate(items, start=1)
            if str(text or "").strip()
        ],
    }


def _brief_sources_from_nodes(
    nodes: Any,
    *,
    source_documents_by_url: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        sources.extend(
            _resolve_report_sources(
                evidence_ids=node.get("evidence_ids") or [],
                source_documents_by_url=source_documents_by_url,
                evidence_by_id=evidence_by_id,
                fallback_label=node.get("phrase"),
                source_tier="primary",
                source_kind="taxonomy_node",
                evidence_type="taxonomy_evidence",
                claim_scope="about_subject_company",
            )
        )
    return sources


def _brief_sources_from_named_proof(
    items: Any,
    *,
    source_documents_by_url: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    source_kind: str,
    evidence_type: str,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        sources.extend(
            _resolve_report_sources(
                evidence_ids=[item.get("evidence_id")] if item.get("evidence_id") else [],
                urls=[item.get("source_url")] if item.get("source_url") else [],
                source_documents_by_url=source_documents_by_url,
                evidence_by_id=evidence_by_id,
                fallback_label=item.get("name"),
                source_tier="primary",
                source_kind=source_kind,
                evidence_type=evidence_type,
                claim_scope="about_subject_company",
            )
        )
    return sources


def _sourcing_summary_sources(
    brief: dict[str, Any],
    *,
    source_documents_by_url: dict[str, dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    preferred_host = normalize_domain(((brief.get("source_company") or {}).get("website")))
    summary_sources = _brief_sources_from_nodes(
        (brief.get("customer_nodes") or [])[:2]
        + (brief.get("workflow_nodes") or [])[:2]
        + (brief.get("capability_nodes") or [])[:3]
        + (brief.get("delivery_or_integration_nodes") or [])[:2],
        source_documents_by_url=source_documents_by_url,
        evidence_by_id=evidence_by_id,
    )
    prioritized_summary_sources = _prioritize_report_sources(summary_sources, preferred_host=preferred_host, max_items=4)
    if prioritized_summary_sources:
        return prioritized_summary_sources

    first_party_sources = [
        source
        for source in source_documents_by_url.values()
        if str(source.get("source_tier") or "").strip().lower() == "primary"
        and (not preferred_host or normalize_domain(source.get("url")) == preferred_host)
    ]
    prioritized_first_party_sources = _prioritize_report_sources(
        first_party_sources,
        preferred_host=preferred_host,
        max_items=4,
    )
    if prioritized_first_party_sources:
        return prioritized_first_party_sources

    fallback_sources: list[dict[str, Any]] = []
    fallback_sources.extend(
        _brief_sources_from_named_proof(
            (brief.get("named_customer_proof") or [])[:2],
            source_documents_by_url=source_documents_by_url,
            evidence_by_id=evidence_by_id,
            source_kind="named_customer",
            evidence_type="named_customer",
        )
    )
    fallback_sources.extend(
        _brief_sources_from_named_proof(
            (brief.get("partner_integration_proof") or [])[:2],
            source_documents_by_url=source_documents_by_url,
            evidence_by_id=evidence_by_id,
            source_kind="partner_integration",
            evidence_type="partner_integration",
        )
    )
    for item in (brief.get("secondary_evidence_proof") or [])[:2]:
        if not isinstance(item, dict):
            continue
        fallback_sources.extend(
            _resolve_report_sources(
                urls=[item.get("url")] if item.get("url") else [],
                source_documents_by_url=source_documents_by_url,
                evidence_by_id=evidence_by_id,
                fallback_label=item.get("title") or item.get("publisher"),
                publisher=item.get("publisher"),
                publisher_channel=item.get("publisher_channel"),
                publisher_type=item.get("publisher_type"),
                source_tier=item.get("evidence_tier") or "secondary",
                source_kind=item.get("claim_type") or "secondary_evidence",
                evidence_type=item.get("claim_type") or "secondary_evidence",
                claim_scope=item.get("claim_scope"),
                published_at=item.get("published_at"),
            )
        )
    if fallback_sources:
        return _prioritize_report_sources(fallback_sources, preferred_host=preferred_host, max_items=4)

    return _prioritize_report_sources(list(source_documents_by_url.values()), preferred_host=preferred_host, max_items=4)


def build_sourcing_report_artifact(
    *,
    sourcing_brief: dict[str, Any],
    source_documents: Any,
    context_pack_v2: Any,
    confirmed_at: Any = None,
) -> dict[str, Any]:
    brief = sourcing_brief if isinstance(sourcing_brief, dict) else {}
    source_company = brief.get("source_company") or {}
    company_name = str(source_company.get("name") or normalize_domain(source_company.get("website")) or "Source Company")
    source_documents_by_url, evidence_by_id = _build_source_document_lookup(source_documents, context_pack_v2)
    source_order: list[str] = []
    sources_by_id: dict[str, dict[str, Any]] = {}

    summary_sources = _sourcing_summary_sources(
        brief,
        source_documents_by_url=source_documents_by_url,
        evidence_by_id=evidence_by_id,
    )

    sections: list[dict[str, Any]] = []

    summary_text = str(brief.get("source_summary") or "").strip()
    if summary_text:
        sections.append(
            {
                "id": "summary",
                "heading": None,
                "blocks": [
                    _paragraph_block(
                        "summary_intro",
                        summary_text,
                        summary_sources,
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    signal_items: list[tuple[str, list[dict[str, Any]]]] = []
    for node in (brief.get("customer_nodes") or [])[:4]:
        signal_items.append(
            (
                f"Customer signal: {node.get('phrase')}.",
                _resolve_report_sources(
                    evidence_ids=node.get("evidence_ids") or [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=node.get("phrase"),
                    source_tier="primary",
                    source_kind="taxonomy_node",
                    evidence_type="customer_signal",
                    claim_scope="about_subject_company",
                ),
            )
        )
    for node in (brief.get("workflow_nodes") or [])[:4]:
        signal_items.append(
            (
                f"Workflow signal: {node.get('phrase')}.",
                _resolve_report_sources(
                    evidence_ids=node.get("evidence_ids") or [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=node.get("phrase"),
                    source_tier="primary",
                    source_kind="taxonomy_node",
                    evidence_type="workflow_signal",
                    claim_scope="about_subject_company",
                ),
            )
        )
    for node in (brief.get("capability_nodes") or [])[:6]:
        signal_items.append(
            (
                f"Capability signal: {node.get('phrase')}.",
                _resolve_report_sources(
                    evidence_ids=node.get("evidence_ids") or [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=node.get("phrase"),
                    source_tier="primary",
                    source_kind="taxonomy_node",
                    evidence_type="capability_signal",
                    claim_scope="about_subject_company",
                ),
            )
        )
    for node in (brief.get("delivery_or_integration_nodes") or [])[:4]:
        signal_items.append(
            (
                f"Delivery or integration signal: {node.get('phrase')}.",
                _resolve_report_sources(
                    evidence_ids=node.get("evidence_ids") or [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=node.get("phrase"),
                    source_tier="primary",
                    source_kind="taxonomy_node",
                    evidence_type="delivery_signal",
                    claim_scope="about_subject_company",
                ),
            )
        )
    if signal_items:
        sections.append(
            {
                "id": "signals",
                "heading": "Customer, workflow, and capability signals",
                "blocks": [
                    _bullet_list_block(
                        "signals_list",
                        signal_items[:12],
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    proof_items: list[tuple[str, list[dict[str, Any]]]] = []
    for item in (brief.get("named_customer_proof") or [])[:8]:
        if not isinstance(item, dict):
            continue
        text = f"Named customer proof: {item.get('name')}."
        if str(item.get("context") or "").strip():
            text = f"{text[:-1]} — {str(item.get('context')).strip()}."
        proof_items.append(
            (
                text,
                _resolve_report_sources(
                    evidence_ids=[item.get("evidence_id")] if item.get("evidence_id") else [],
                    urls=[item.get("source_url")] if item.get("source_url") else [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=item.get("name"),
                    source_tier="primary",
                    source_kind=str(item.get("evidence_type") or "named_customer"),
                    evidence_type=str(item.get("evidence_type") or "named_customer"),
                    claim_scope="about_subject_company",
                ),
            )
        )
    for item in (brief.get("partner_integration_proof") or [])[:8]:
        if not isinstance(item, dict):
            continue
        proof_items.append(
            (
                f"Partner or integration proof: {item.get('name')}.",
                _resolve_report_sources(
                    evidence_ids=[item.get("evidence_id")] if item.get("evidence_id") else [],
                    urls=[item.get("source_url")] if item.get("source_url") else [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=item.get("name"),
                    source_tier="primary",
                    source_kind="partner_integration",
                    evidence_type="partner_integration",
                    claim_scope="about_subject_company",
                ),
            )
        )
    if proof_items:
        sections.append(
            {
                "id": "proof",
                "heading": "Named proof",
                "blocks": [
                    _bullet_list_block(
                        "proof_list",
                        proof_items,
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    secondary_items: list[tuple[str, list[dict[str, Any]]]] = []
    for item in (brief.get("secondary_evidence_proof") or [])[:10]:
        if not isinstance(item, dict):
            continue
        secondary_items.append(
            (
                str(item.get("claim_text") or item.get("evidence_snippet") or item.get("title") or "").strip(),
                _resolve_report_sources(
                    urls=[item.get("url")] if item.get("url") else [],
                    source_documents_by_url=source_documents_by_url,
                    evidence_by_id=evidence_by_id,
                    fallback_label=item.get("title") or item.get("publisher"),
                    publisher=item.get("publisher"),
                    publisher_channel=item.get("publisher_channel"),
                    publisher_type=item.get("publisher_type"),
                    source_tier=item.get("evidence_tier") or "secondary",
                    source_kind=item.get("claim_type") or "secondary_evidence",
                    evidence_type=item.get("claim_type") or "secondary_evidence",
                    claim_scope=item.get("claim_scope"),
                    published_at=item.get("published_at"),
                ),
            )
        )
    if secondary_items:
        sections.append(
            {
                "id": "secondary",
                "heading": "Secondary corroboration",
                "blocks": [
                    _bullet_list_block(
                        "secondary_list",
                        secondary_items,
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    gap_items = [(gap, []) for gap in (brief.get("confidence_gaps") or [])[:4] if str(gap or "").strip()]
    question_items = [(question, []) for question in (brief.get("open_questions") or [])[:4] if str(question or "").strip()]
    if gap_items or question_items:
        blocks: list[dict[str, Any]] = []
        if gap_items:
            blocks.append(
                _callout_block(
                    "confidence_gaps",
                    "warning",
                    "Confidence gaps",
                    gap_items,
                    source_order=source_order,
                    sources_by_id=sources_by_id,
                )
            )
        if question_items:
            blocks.append(
                _bullet_list_block(
                    "open_questions",
                    question_items,
                    source_order=source_order,
                    sources_by_id=sources_by_id,
                )
            )
        sections.append(
            {
                "id": "gaps",
                "heading": "Confidence gaps and open questions",
                "blocks": blocks,
            }
        )

    return {
        "artifact_type": "report_artifact",
        "report_kind": "sourcing_brief",
        "version": "v1",
        "status": "degraded" if str(brief.get("reasoning_status") or "not_run") != "success" else "ready",
        "generated_at": datetime.utcnow().isoformat(),
        "confirmed_at": confirmed_at.isoformat() if isinstance(confirmed_at, datetime) else confirmed_at,
        "reasoning_status": brief.get("reasoning_status") or "not_run",
        "reasoning_warning": brief.get("reasoning_warning"),
        "reasoning_provider": brief.get("reasoning_provider"),
        "reasoning_model": brief.get("reasoning_model"),
        "title": f"{company_name} Sourcing Brief",
        "summary": summary_text or None,
        "sections": sections,
        "sources": [sources_by_id[source_id] for source_id in source_order],
        "footer_actions": ["copy", "good", "bad", "share", "regenerate"],
    }


def build_expansion_report_artifact(
    *,
    source_company: Any,
    expansion_brief: dict[str, Any],
    source_documents: Any,
    context_pack_v2: Any,
    confirmed_at: Any = None,
) -> dict[str, Any]:
    brief = normalize_expansion_brief(expansion_brief or {})
    company_name = str((source_company or {}).get("name") or normalize_domain((source_company or {}).get("website")) or "Source Company")
    source_documents_by_url, evidence_by_id = _build_source_document_lookup(source_documents, context_pack_v2)
    source_order: list[str] = []
    sources_by_id: dict[str, dict[str, Any]] = {}

    def _item_sources(item: dict[str, Any]) -> list[dict[str, Any]]:
        return _resolve_report_sources(
            evidence_ids=item.get("evidence_ids") or [],
            urls=item.get("evidence_urls") or [],
            source_documents_by_url=source_documents_by_url,
            evidence_by_id=evidence_by_id,
            fallback_label=item.get("label"),
            publisher=", ".join(item.get("source_entity_names") or []) or None,
            publisher_channel="expansion_input",
            publisher_type="other",
            source_tier="secondary",
            source_kind=item.get("expansion_type") or "expansion_signal",
            evidence_type=item.get("expansion_type") or "expansion_signal",
            claim_scope="about_subject_company",
        )

    sections: list[dict[str, Any]] = []

    summary_bits: list[str] = []
    if (brief.get("adjacent_capabilities") or []):
        summary_bits.append(
            "Adjacent capabilities center on "
            + ", ".join(str(item.get("label") or "").strip() for item in (brief.get("adjacent_capabilities") or [])[:3] if str(item.get("label") or "").strip())
            + "."
        )
    if (brief.get("adjacent_customer_segments") or []):
        summary_bits.append(
            "Adjacent customer segments include "
            + ", ".join(str(item.get("label") or "").strip() for item in (brief.get("adjacent_customer_segments") or [])[:3] if str(item.get("label") or "").strip())
            + "."
        )
    if (brief.get("named_account_anchors") or []):
        summary_bits.append(
            "Named account anchors include "
            + ", ".join(str(item.get("label") or "").strip() for item in (brief.get("named_account_anchors") or [])[:3] if str(item.get("label") or "").strip())
            + "."
        )
    summary_text = " ".join(bit for bit in summary_bits if bit.strip()).strip()
    summary_sources: list[dict[str, Any]] = []
    for group in ("adjacent_capabilities", "adjacent_customer_segments", "named_account_anchors", "geography_expansions"):
        for item in (brief.get(group) or [])[:2]:
            if isinstance(item, dict):
                summary_sources.extend(_item_sources(item))
    if summary_text:
        sections.append(
            {
                "id": "summary",
                "heading": None,
                "blocks": [
                    _paragraph_block(
                        "expansion_summary",
                        summary_text,
                        summary_sources[:4],
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    corroborated_items: list[tuple[str, list[dict[str, Any]]]] = []
    hypothesis_items: list[tuple[str, list[dict[str, Any]]]] = []
    for group in ("adjacent_capabilities", "adjacent_customer_segments", "named_account_anchors", "geography_expansions"):
        for item in (brief.get(group) or []):
            if not isinstance(item, dict):
                continue
            text = str(item.get("why_it_matters") or item.get("label") or "").strip()
            if text and text != str(item.get("label") or "").strip():
                text = f"{item.get('label')}: {text}"
            else:
                text = str(item.get("label") or "").strip()
            bucket = corroborated_items if str(item.get("status") or "").strip() in {"corroborated_expansion", "source_grounded", "user_kept"} else hypothesis_items
            bucket.append((text, _item_sources(item)))

    if corroborated_items:
        sections.append(
            {
                "id": "corroborated",
                "heading": "Corroborated expansions",
                "blocks": [
                    _bullet_list_block(
                        "corroborated_list",
                        corroborated_items[:10],
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    if hypothesis_items:
        sections.append(
            {
                "id": "hypotheses",
                "heading": "Hypotheses to validate",
                "blocks": [
                    _callout_block(
                        "hypotheses_list",
                        "info",
                        None,
                        hypothesis_items[:10],
                        source_order=source_order,
                        sources_by_id=sources_by_id,
                    )
                ],
            }
        )

    for group, heading in (
        ("named_account_anchors", "Named account anchors"),
        ("geography_expansions", "Geography expansions"),
    ):
        items = [
            (
                str(item.get("label") or "").strip() if not str(item.get("why_it_matters") or "").strip()
                else f"{item.get('label')}: {str(item.get('why_it_matters')).strip()}",
                _item_sources(item),
            )
            for item in (brief.get(group) or [])[:8]
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        ]
        if items:
            sections.append(
                {
                    "id": group,
                    "heading": heading,
                    "blocks": [
                        _bullet_list_block(
                            group,
                            items,
                            source_order=source_order,
                            sources_by_id=sources_by_id,
                        )
                    ],
                }
            )

    return {
        "artifact_type": "report_artifact",
        "report_kind": "expansion_brief",
        "version": "v1",
        "status": "degraded" if str(brief.get("reasoning_status") or "not_run") != "success" else "ready",
        "generated_at": datetime.utcnow().isoformat(),
        "confirmed_at": confirmed_at.isoformat() if isinstance(confirmed_at, datetime) else confirmed_at,
        "reasoning_status": brief.get("reasoning_status") or "not_run",
        "reasoning_warning": brief.get("reasoning_warning"),
        "reasoning_provider": brief.get("reasoning_provider"),
        "reasoning_model": brief.get("reasoning_model"),
        "title": f"{company_name} Expansion Brief",
        "summary": summary_text or None,
        "sections": sections,
        "sources": [sources_by_id[source_id] for source_id in source_order],
        "footer_actions": ["copy", "good", "bad", "share", "regenerate"],
    }


def _scope_status_from_taxonomy_scope(scope_status: Any) -> str:
    value = str(scope_status or "in_scope").strip().lower()
    if value == "removed":
        return "user_removed"
    if value == "out_of_scope":
        return "user_deprioritized"
    return "source_grounded"


def _taxonomy_scope_from_scope_status(status: Any) -> str:
    value = str(status or "").strip().lower()
    if value == "user_removed":
        return "removed"
    if value == "user_deprioritized":
        return "out_of_scope"
    return "in_scope"


def _group_scope_item_type(layer: str) -> str:
    mapping = {
        "capability": "source_capability",
        "customer_archetype": "source_customer_segment",
        "workflow": "source_workflow",
        "delivery_or_integration": "source_delivery_or_integration",
    }
    return mapping.get(str(layer or "").strip(), "source_node")


def derive_scope_review_payload(
    company_context_pack: CompanyContextPack | dict[str, Any],
    profile: CompanyProfile,
) -> dict[str, Any]:
    if isinstance(company_context_pack, CompanyContextPack):
        taxonomy_nodes = normalize_taxonomy_nodes(company_context_pack.taxonomy_nodes_json or [])
        sourcing_brief = company_context_pack.sourcing_brief_json or {}
        expansion_brief = normalize_expansion_brief(company_context_pack.expansion_brief_json or {})
    else:
        taxonomy_nodes = normalize_taxonomy_nodes(company_context_pack.get("taxonomy_nodes") or [])
        sourcing_brief = company_context_pack.get("sourcing_brief") or {}
        expansion_brief = normalize_expansion_brief(company_context_pack.get("expansion_brief") or {})

    context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    evidence_url_by_id: dict[str, str] = {}
    for item in (context_pack_v2.get("evidence_items") or []):
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("id") or "").strip()
        url = normalize_url(item.get("url"))
        if evidence_id and url:
            evidence_url_by_id[evidence_id] = url
    for site in (context_pack_v2.get("sites") or []):
        if not isinstance(site, dict):
            continue
        for item in (site.get("evidence_items") or []):
            if not isinstance(item, dict):
                continue
            evidence_id = str(item.get("id") or "").strip()
            url = normalize_url(item.get("url"))
            if evidence_id and url:
                evidence_url_by_id[evidence_id] = url

    selected_ids = {
        str(item.get("id") or "")
        for key in ("customer_nodes", "workflow_nodes", "capability_nodes", "delivery_or_integration_nodes")
        for item in (sourcing_brief.get(key) or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }

    def _source_group(layer: str) -> list[dict[str, Any]]:
        rows = []
        for node in taxonomy_nodes:
            if not isinstance(node, dict):
                continue
            if str(node.get("layer") or "").strip() != layer:
                continue
            label = str(node.get("phrase") or "").strip()
            if not label:
                continue
            rows.append(
                {
                    "id": str(node.get("id") or ""),
                    "label": label,
                    "scope_item_type": _group_scope_item_type(layer),
                    "origin": "source_brief",
                    "status": _scope_status_from_taxonomy_scope(node.get("scope_status")),
                    "confidence": _clamp_confidence(node.get("confidence"), default=0.68),
                    "evidence_ids": _normalize_string_list(node.get("evidence_ids"), max_items=8, max_len=96),
                    "evidence_urls": _normalize_string_list(
                        [
                            evidence_url_by_id[evidence_id]
                            for evidence_id in _normalize_string_list(node.get("evidence_ids"), max_items=8, max_len=96)
                            if evidence_id in evidence_url_by_id
                        ],
                        max_items=6,
                        max_len=500,
                    ),
                    "supporting_node_ids": [str(node.get("id") or "")] if str(node.get("id") or "").strip() else [],
                    "source_entity_names": [],
                    "why_it_matters": None,
                    "priority_tier": "core" if str(node.get("id") or "") in selected_ids else "supporting",
                }
            )
        return sorted(
            rows,
            key=lambda item: (
                0 if item.get("priority_tier") == "core" else 1,
                -float(item.get("confidence") or 0.0),
                str(item.get("label") or ""),
            ),
        )[:12]

    return {
        "workspace_geo_scope": profile.geo_scope or {},
        "confirmed_at": expansion_brief.get("confirmed_at"),
        "source_capabilities": _source_group("capability"),
        "source_customer_segments": _source_group("customer_archetype"),
        "source_workflows": _source_group("workflow"),
        "source_delivery_or_integration": _source_group("delivery_or_integration"),
        "adjacent_capabilities": expansion_brief.get("adjacent_capabilities") or [],
        "adjacent_customer_segments": expansion_brief.get("adjacent_customer_segments") or [],
        "named_account_anchors": expansion_brief.get("named_account_anchors") or [],
        "geography_expansions": expansion_brief.get("geography_expansions") or [],
    }


def apply_scope_review_decisions(
    company_context_pack: CompanyContextPack | dict[str, Any],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(company_context_pack, CompanyContextPack):
        payload = {
            "taxonomy_nodes": deepcopy(company_context_pack.taxonomy_nodes_json or []),
            "expansion_brief": deepcopy(company_context_pack.expansion_brief_json or {}),
        }
    else:
        payload = {
            "taxonomy_nodes": deepcopy(company_context_pack.get("taxonomy_nodes") or []),
            "expansion_brief": deepcopy(company_context_pack.get("expansion_brief") or {}),
        }

    taxonomy_nodes = normalize_taxonomy_nodes(payload["taxonomy_nodes"])
    expansion_brief = normalize_expansion_brief(payload["expansion_brief"])
    taxonomy_by_id = {str(node.get("id") or ""): node for node in taxonomy_nodes if str(node.get("id") or "").strip()}

    expansion_groups = (
        "adjacent_capabilities",
        "adjacent_customer_segments",
        "named_account_anchors",
        "geography_expansions",
    )

    for raw in decisions or []:
        if not isinstance(raw, dict):
            continue
        item_id = str(raw.get("id") or "").strip()
        status = str(raw.get("status") or "").strip().lower()
        if not item_id or status not in {"user_kept", "user_removed", "user_deprioritized", "source_grounded"}:
            continue
        if item_id in taxonomy_by_id:
            taxonomy_by_id[item_id]["scope_status"] = _taxonomy_scope_from_scope_status(status)
            continue
        for group in expansion_groups:
            updated_group = []
            found = False
            for item in expansion_brief.get(group) or []:
                if not isinstance(item, dict):
                    continue
                if str(item.get("id") or "").strip() == item_id:
                    updated_group.append({**item, "status": status})
                    found = True
                else:
                    updated_group.append(item)
            if found:
                expansion_brief[group] = updated_group
                break

    return {
        "taxonomy_nodes": normalize_taxonomy_nodes(list(taxonomy_by_id.values())),
        "expansion_brief": normalize_expansion_brief(expansion_brief),
    }

def derive_discovery_scope_hints(
    company_context_pack: CompanyContextPack | dict[str, Any],
    profile: CompanyProfile,
) -> dict[str, Any]:
    scope = derive_scope_review_payload(company_context_pack, profile)

    source_capabilities = [
        item["label"]
        for item in scope["source_capabilities"]
        if item.get("status") in {"source_grounded", "user_kept"}
    ]
    source_workflows = [
        item["label"]
        for item in scope["source_workflows"]
        if item.get("status") in {"source_grounded", "user_kept"}
    ]
    source_customers = [
        item["label"]
        for item in scope["source_customer_segments"]
        if item.get("status") in {"source_grounded", "user_kept"}
    ]
    adjacent_capabilities = [
        item["label"]
        for item in scope["adjacent_capabilities"]
        if item.get("status") in {"source_grounded", "corroborated_expansion", "user_kept"}
        and (item.get("priority_tier") != "edge_case" or item.get("status") == "user_kept")
    ]
    adjacent_customers = [
        item["label"]
        for item in scope["adjacent_customer_segments"]
        if item.get("status") in {"source_grounded", "corroborated_expansion", "user_kept"}
        and (item.get("priority_tier") != "edge_case" or item.get("status") == "user_kept")
    ]
    named_accounts = [
        item["label"]
        for item in scope["named_account_anchors"]
        if item.get("status") in {"source_grounded", "corroborated_expansion", "user_kept"}
    ]

    return {
        "source_capabilities": _normalize_string_list(source_capabilities + source_workflows[:2], max_items=8, max_len=140)
        or _normalize_string_list(source_workflows, max_items=6, max_len=140),
        "adjacent_capabilities": _normalize_string_list(adjacent_capabilities, max_items=8, max_len=140),
        "source_customer_segments": _normalize_string_list(source_customers, max_items=6, max_len=140),
        "adjacent_customer_segments": _normalize_string_list(adjacent_customers, max_items=6, max_len=140),
        "named_account_anchors": _normalize_string_list(named_accounts, max_items=8, max_len=140),
        "comparator_seed_urls": [
            normalized
            for normalized in (
                normalize_url(url)
                for url in _normalize_string_list(profile.comparator_seed_urls or [], max_items=8, max_len=240)
            )
            if normalized
        ],
        "confirmed": bool(
            normalize_expansion_brief(
                company_context_pack.expansion_brief_json or {}
                if isinstance(company_context_pack, CompanyContextPack)
                else company_context_pack.get("expansion_brief") or {}
            ).get("confirmed_at")
        ),
    }
