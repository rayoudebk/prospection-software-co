from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - optional runtime dependency during local tests
    GraphDatabase = None

from app.config import get_settings
from app.models.workspace import CompanyProfile
from app.services.retrieval.crawl_connectors import fetch_page_fast
from app.services.retrieval.search_orchestrator import run_external_search_queries
from app.services.reporting import normalize_domain
from app.services.company_context import (
    build_company_context_artifacts,
    build_context_pack_v2,
    build_expansion_inputs,
)

GRAPH_NODE_LABELS = {
    "Company",
    "CustomerEntity",
    "CustomerArchetype",
    "Workflow",
    "Capability",
    "DeliveryIntegration",
    "PartnerEntity",
    "Category",
    "SourceDocument",
    "Claim",
}
GRAPH_EDGE_TYPES = {
    "OFFERS_CAPABILITY",
    "SUPPORTS_WORKFLOW",
    "SERVES_CUSTOMER_ENTITY",
    "SERVES_CUSTOMER_ARCHETYPE",
    "INTEGRATES_WITH",
    "LISTED_IN_CATEGORY",
    "ANNOUNCED_BY_CUSTOMER",
    "SUPPORTED_BY",
    "CONTRADICTED_BY",
    "MENTIONS",
}
PRIMARY_EVIDENCE_TIER = "primary"
SECONDARY_EVIDENCE_TIER = "secondary"
INFERRED_EVIDENCE_TIER = "inferred"

DIRECTORY_HOST_TOKENS = (
    "thewealthmosaic.com",
    "g2.com",
    "capterra.com",
    "crunchbase.com",
)
DIRECTORY_PRIORITY_HOSTS = (
    "thewealthmosaic.com",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
)
COMPANY_PROFILE_HOST_TOKENS = (
    "crunchbase.com",
    "zoominfo.com",
    "verif.com",
    "rubypayeur.com",
)
COMPANY_PROFILE_HOSTS = (
    "crunchbase.com",
    "zoominfo.com",
    "verif.com",
    "rubypayeur.com",
)
GENERIC_INTEGRATION_NAME_TOKENS = (
    "api",
    "rest",
    "rest api",
    "webhook",
    "webhooks",
    "sso",
    "oauth",
    "sdk",
    "documentation",
    "fix",
    "swift",
    "ftp",
    "sftp",
)
SECONDARY_QUERY_TYPE_BUCKETS = {
    "directory_category": ("directory", "category_positioning"),
    "customer_corroboration": ("press_release", "deployment_announcement"),
    "partner_corroboration": ("partner_page", "integration_announcement"),
    "market_context": ("trade_publication", "workflow_description"),
}
SOURCE_TYPE_TO_QUERY_TYPE = {
    "directory": "directory_category",
    "press_release": "customer_corroboration",
    "partner_page": "partner_corroboration",
    "trade_publication": "market_context",
}
PRESS_RELEASE_TOKENS = ("press release", "announces", "announcement", "newsroom", "media")
PARTNER_TOKENS = ("partner", "partnership", "partners with", "integration", "integrates with")
DEPLOYMENT_TOKENS = ("deploy", "deployment", "go live", "go-live", "selects", "chooses", "implements", "implementation")
LOW_SIGNAL_HOST_TOKENS = ("youtube.com", "facebook.com", "instagram.com", "x.com", "twitter.com")
LOW_SIGNAL_CORROBORATION_HOST_TOKENS = (
    "linkedin.com",
    "rocketreach.co",
    "pappers.fr",
    "northdata.com",
    "sec.gov",
    "societe.com",
    "lefigaro.fr",
    "bodacc.fr",
)
LOW_SIGNAL_SECONDARY_TEXT_TOKENS = (
    "password should contain",
    "confirm password",
    "email @",
    "sign in",
    "log in",
    "javascript required",
)
FALSE_POSITIVE_SCIENCE_TOKENS = (
    "crystal structure",
    "protein data bank",
    "pde10a",
    "inhibitors",
    "ligand",
    "molecule",
)
PRIMARY_GRAPH_LAYER_LIMITS = {
    "customer_archetype": 6,
    "workflow": 8,
    "capability": 12,
    "delivery_or_integration": 6,
}


def _utcnow_iso() -> str:
    return datetime.utcnow().isoformat()


def _stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join([prefix, *[str(part or "").strip().lower() for part in parts]])
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def _clean_phrase(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _slug_phrase(value: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")
    return lowered or "unknown"


def _confidence(value: Any, default: float = 0.7) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(0.0, min(parsed, 1.0))


def company_context_graph_ref(workspace_id: int, company_name: str | None = None) -> str:
    suffix = _slug_phrase(company_name or f"workspace-{workspace_id}")
    return f"workspace-{int(workspace_id)}-{suffix}"


def _empty_graph(graph_ref: str, workspace_id: int) -> Dict[str, Any]:
    return {
        "graph_ref": graph_ref,
        "workspace_id": workspace_id,
        "generated_at": _utcnow_iso(),
        "nodes": [],
        "edges": [],
        "sourcing_brief": {},
        "source_documents": [],
        "secondary_evidence_proof": [],
        "graph_stats": {
            "node_count": 0,
            "edge_count": 0,
            "source_document_count": 0,
            "primary_evidence_count": 0,
            "secondary_evidence_count": 0,
        },
    }


def _node(
    node_id: str,
    label: str,
    *,
    name: str,
    evidence_tier: str,
    source_type: str,
    evidence_type: str,
    confidence: float = 0.7,
    source_document_id: Optional[str] = None,
    freshness: Optional[str] = None,
    published_at: Optional[str] = None,
    extracted_at: Optional[str] = None,
    job_id: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if label not in GRAPH_NODE_LABELS:
        raise ValueError(f"Unsupported graph node label: {label}")
    payload = {
        "id": node_id,
        "label": label,
        "name": _clean_phrase(name),
        "evidence_tier": evidence_tier,
        "source_type": source_type,
        "evidence_type": evidence_type,
        "confidence": _confidence(confidence),
        "source_document_id": source_document_id,
        "freshness": freshness,
        "published_at": published_at,
        "extracted_at": extracted_at or _utcnow_iso(),
        "job_id": job_id,
    }
    if properties:
        payload.update(properties)
    return payload


def _edge(
    edge_id: str,
    edge_type: str,
    from_id: str,
    to_id: str,
    *,
    evidence_tier: str,
    source_type: str,
    evidence_type: str,
    confidence: float = 0.7,
    source_document_id: Optional[str] = None,
    freshness: Optional[str] = None,
    published_at: Optional[str] = None,
    extracted_at: Optional[str] = None,
    job_id: Optional[str] = None,
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if edge_type not in GRAPH_EDGE_TYPES:
        raise ValueError(f"Unsupported graph edge type: {edge_type}")
    payload = {
        "id": edge_id,
        "type": edge_type,
        "from_id": from_id,
        "to_id": to_id,
        "evidence_tier": evidence_tier,
        "source_type": source_type,
        "evidence_type": evidence_type,
        "confidence": _confidence(confidence),
        "source_document_id": source_document_id,
        "freshness": freshness,
        "published_at": published_at,
        "extracted_at": extracted_at or _utcnow_iso(),
        "job_id": job_id,
    }
    if properties:
        payload.update(properties)
    return payload


def _source_document_node(
    *,
    graph_ref: str,
    url: str,
    title: Optional[str],
    source_type: str,
    evidence_tier: str,
    evidence_type: str,
    publisher: Optional[str] = None,
    published_at: Optional[str] = None,
    snippet: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_url = str(url or "").strip()
    node_id = _stable_id("source_document", graph_ref, normalized_url)
    host = normalize_domain(normalized_url) or ""
    display = _clean_phrase(title or host or normalized_url)
    return _node(
        node_id,
        "SourceDocument",
        name=display,
        evidence_tier=evidence_tier,
        source_type=source_type,
        evidence_type=evidence_type,
        confidence=0.9 if evidence_tier == PRIMARY_EVIDENCE_TIER else 0.7,
        published_at=published_at,
        properties={
            "url": normalized_url,
            "publisher": publisher or host,
            "snippet": (snippet or "")[:1200],
            "hostname": host,
            "publisher_channel": source_type,
            "publisher_type": "source_company" if evidence_tier == PRIMARY_EVIDENCE_TIER else None,
            "claim_scope": "about_self" if evidence_tier == PRIMARY_EVIDENCE_TIER else "about_subject_company",
            "graph_ref": graph_ref,
        },
    )


def _infer_source_document_title(item: Dict[str, Any]) -> Optional[str]:
    return (
        item.get("page_title")
        or item.get("title")
        or item.get("label")
        or item.get("name")
        or None
    )


def _first_party_source_documents(payload: Dict[str, Any], graph_ref: str) -> List[Dict[str, Any]]:
    documents_by_url: Dict[str, Dict[str, Any]] = {}

    def _merge_document(candidate: Dict[str, Any]) -> None:
        normalized_url = str(((candidate.get("url") if isinstance(candidate, dict) else "") or "")).strip()
        if not normalized_url:
            return
        existing = documents_by_url.get(normalized_url)
        if existing is None:
            documents_by_url[normalized_url] = candidate
            return
        merged = dict(existing)
        for field in ("name", "publisher", "evidence_type", "page_type", "page_title", "snippet", "published_at"):
            if not merged.get(field) and candidate.get(field):
                merged[field] = candidate.get(field)
        merged["confidence"] = max(float(existing.get("confidence") or 0.0), float(candidate.get("confidence") or 0.0))
        documents_by_url[normalized_url] = merged

    for pill in payload.get("source_pills") or []:
        if not isinstance(pill, dict) or not pill.get("url"):
            continue
        document = _source_document_node(
            graph_ref=graph_ref,
            url=str(pill.get("url")),
            title=_infer_source_document_title(pill),
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            evidence_type="source_document",
        )
        _merge_document(document)
    context_pack = payload.get("context_pack_v2") or {}
    for site in context_pack.get("sites") or []:
        if not isinstance(site, dict):
            continue
        page_lookup = {
            str(page.get("url") or "").strip(): page
            for page in (site.get("pages") or [])
            if isinstance(page, dict) and str(page.get("url") or "").strip()
        }
        for item in site.get("selected_pages") or []:
            if not isinstance(item, dict) or not item.get("url"):
                continue
            page_url = str(item.get("url") or "").strip()
            page = page_lookup.get(page_url) or {}
            snippet = " | ".join(
                [
                    value
                    for value in [
                        " / ".join(item.get("headings") or [])[:240] if isinstance(item.get("headings"), list) else "",
                        str(page.get("raw_content") or "")[:320],
                    ]
                    if str(value or "").strip()
                ]
            )
            document = _source_document_node(
                graph_ref=graph_ref,
                url=page_url,
                title=str(item.get("title") or page.get("title") or "").strip() or None,
                source_type=PRIMARY_EVIDENCE_TIER,
                evidence_tier=PRIMARY_EVIDENCE_TIER,
                evidence_type=str(item.get("page_type") or page.get("page_type") or "source_document"),
                snippet=snippet,
            )
            document["page_type"] = str(item.get("page_type") or page.get("page_type") or "").strip() or None
            document["page_title"] = str(item.get("title") or page.get("title") or "").strip() or None
            _merge_document(document)
    for item in context_pack.get("evidence_items") or []:
        if not isinstance(item, dict) or not item.get("url"):
            continue
        document = _source_document_node(
            graph_ref=graph_ref,
            url=str(item.get("url")),
            title=_infer_source_document_title(item),
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            evidence_type=str(item.get("kind") or "evidence"),
            snippet=str(item.get("snippet") or item.get("text") or "")[:1200],
        )
        document["page_type"] = str(item.get("page_type") or "").strip() or None
        document["page_title"] = str(item.get("page_title") or "").strip() or None
        document["evidence_item_id"] = str(item.get("id") or "").strip() or None
        document["captured_at"] = item.get("captured_at")
        _merge_document(document)
    return list(documents_by_url.values())


def _comparator_seed_domains(profile: CompanyProfile) -> set[str]:
    domains: set[str] = set()
    for url in profile.comparator_seed_urls or []:
        domain = normalize_domain(str(url or "").strip())
        if domain:
            domains.add(domain)
            www_variant = domain[4:] if domain.startswith("www.") else f"www.{domain}"
            domains.add(www_variant)
    return domains


def _filter_source_company_documents(
    documents: List[Dict[str, Any]],
    *,
    profile: CompanyProfile,
) -> List[Dict[str, Any]]:
    comparator_domains = _comparator_seed_domains(profile)
    if not comparator_domains:
        return documents
    filtered: List[Dict[str, Any]] = []
    for item in documents:
        if not isinstance(item, dict):
            continue
        hostname = normalize_domain(str(item.get("url") or item.get("hostname") or "").strip())
        if hostname and hostname in comparator_domains:
            continue
        filtered.append(item)
    return filtered


def _first_party_evidence_items(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    context_pack = payload.get("context_pack_v2") or {}
    return {
        str(item.get("id")): item
        for item in (context_pack.get("evidence_items") or [])
        if isinstance(item, dict) and str(item.get("id") or "").strip()
    }


def _source_document_id_by_url(source_documents: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in source_documents:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        doc_id = str(item.get("id") or "").strip()
        if url and doc_id:
            mapping[url] = doc_id
    return mapping


def _canonicalize_graph(graph: Dict[str, Any]) -> Dict[str, Any]:
    canonical = deepcopy(graph)

    source_documents: Dict[str, Dict[str, Any]] = {}
    for item in canonical.get("source_documents") or []:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        if not item_id:
            continue
        source_documents[item_id] = item

    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    for node in canonical.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            continue
        nodes_by_id[node_id] = node
        if str(node.get("label") or "") == "SourceDocument":
            source_documents[node_id] = node

    edges_by_id: Dict[str, Dict[str, Any]] = {}
    valid_node_ids = set(nodes_by_id.keys())
    for edge in canonical.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        edge_id = str(edge.get("id") or "").strip()
        from_id = str(edge.get("from_id") or "").strip()
        to_id = str(edge.get("to_id") or "").strip()
        if not edge_id or not from_id or not to_id:
            continue
        if from_id not in valid_node_ids or to_id not in valid_node_ids:
            continue
        edges_by_id[edge_id] = edge

    canonical["nodes"] = list(nodes_by_id.values())
    canonical["edges"] = list(edges_by_id.values())
    canonical["source_documents"] = list(source_documents.values())
    stats = dict(canonical.get("graph_stats") or {})
    stats["node_count"] = len(canonical["nodes"])
    stats["edge_count"] = len(canonical["edges"])
    stats["source_document_count"] = len(canonical["source_documents"])
    canonical["graph_stats"] = stats
    return canonical


def _resolve_source_document_id(
    evidence_ids: Iterable[str],
    *,
    evidence_items_by_id: Dict[str, Dict[str, Any]],
    source_document_id_by_url: Dict[str, str],
    fallback_source_document_id: Optional[str] = None,
) -> Optional[str]:
    for evidence_id in evidence_ids:
        item = evidence_items_by_id.get(str(evidence_id))
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if url and source_document_id_by_url.get(url):
            return source_document_id_by_url[url]
    return fallback_source_document_id


def _add_supported_by_edge(
    edges: Dict[str, Dict[str, Any]],
    *,
    graph_ref: str,
    from_id: str,
    source_document_id: Optional[str],
    evidence_tier: str,
    source_type: str,
    evidence_type: str,
    confidence: float,
) -> None:
    if not from_id or not source_document_id:
        return
    edge_id = _stable_id("supported_by", graph_ref, from_id, source_document_id)
    edges[edge_id] = _edge(
        edge_id,
        "SUPPORTED_BY",
        from_id,
        source_document_id,
        evidence_tier=evidence_tier,
        source_type=source_type,
        evidence_type=evidence_type,
        confidence=confidence,
        source_document_id=source_document_id,
        properties={"graph_ref": graph_ref},
    )


def _taxonomy_label(layer: str) -> Optional[str]:
    mapping = {
        "customer_archetype": "CustomerArchetype",
        "workflow": "Workflow",
        "capability": "Capability",
        "delivery_or_integration": "DeliveryIntegration",
    }
    return mapping.get(str(layer or "").strip())


def _normalize_taxonomy_identity(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return normalized


def _curated_primary_taxonomy_nodes(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    nodes = [
        item
        for item in (payload.get("taxonomy_nodes") or [])
        if isinstance(item, dict) and str(item.get("scope_status") or "in_scope") == "in_scope"
    ]
    if not nodes:
        return []

    brief = payload.get("sourcing_brief") or {}
    selected_ids = {
        str(node.get("id"))
        for key in (
            "customer_nodes",
            "workflow_nodes",
            "capability_nodes",
            "delivery_or_integration_nodes",
        )
        for node in (brief.get(key) or [])
        if isinstance(node, dict) and str(node.get("id") or "").strip()
    }
    selected_order = {node_id: idx for idx, node_id in enumerate(selected_ids)}

    by_layer: Dict[str, List[Dict[str, Any]]] = {}
    for item in nodes:
        layer = str(item.get("layer") or "").strip()
        by_layer.setdefault(layer, []).append(item)

    curated: List[Dict[str, Any]] = []
    for layer, limit in PRIMARY_GRAPH_LAYER_LIMITS.items():
        layer_nodes = by_layer.get(layer) or []
        seen_keys: set[str] = set()
        ranked = sorted(
            layer_nodes,
            key=lambda item: (
                0 if str(item.get("id") or "") in selected_ids else 1,
                selected_order.get(str(item.get("id") or ""), 9999),
                -float(item.get("confidence") or 0.0),
                str(item.get("phrase") or ""),
            ),
        )
        for item in ranked:
            phrase_key = _normalize_taxonomy_identity(item.get("phrase"))
            if not phrase_key or phrase_key in seen_keys:
                continue
            seen_keys.add(phrase_key)
            curated.append(item)
            if len([node for node in curated if str(node.get("layer") or "") == layer]) >= limit:
                break
    curated_ids = {str(item.get("id") or "") for item in curated}
    for item in nodes:
        node_id = str(item.get("id") or "")
        if node_id in selected_ids and node_id not in curated_ids:
            curated.append(item)
            curated_ids.add(node_id)
    return curated


def _primary_context_terms(primary_graph: Dict[str, Any]) -> List[str]:
    terms: List[str] = []
    for node in primary_graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        if node.get("label") not in {"CustomerArchetype", "Workflow", "Capability", "PartnerEntity", "CustomerEntity"}:
            continue
        phrase = _clean_phrase(node.get("name"))
        if phrase and phrase not in terms:
            terms.append(phrase)
    return terms[:24]


def _is_ambiguous_company_name(company_name: str, company_domain: str) -> bool:
    token = _normalize_taxonomy_identity(company_name).replace(" ", "")
    if not token:
        return False
    if any(ch.isdigit() for ch in token):
        return True
    if len(token) <= 5:
        return True
    domain_stem = (company_domain.split(".")[0] if company_domain else "").lower()
    return bool(domain_stem and len(domain_stem) <= 5)


def _company_identity_match_strength(
    *,
    company_name: str,
    company_domain: str,
    title: str,
    snippet: str,
    content: str,
    matched_context_terms: List[str],
) -> str:
    text_blob = " ".join([title, snippet, content]).lower()
    company_token = _clean_phrase(company_name).lower()
    domain_stem = (company_domain.split(".")[0] if company_domain else "").lower()
    if company_domain and company_domain.lower() in text_blob:
        return "strong"
    if company_token and company_token in text_blob and not _is_ambiguous_company_name(company_name, company_domain):
        return "strong"
    if domain_stem and domain_stem in text_blob and matched_context_terms:
        return "strong"
    if matched_context_terms and any(_clean_phrase(term).lower() in text_blob for term in matched_context_terms[:8]):
        return "contextual"
    return "weak"


def build_primary_company_graph_from_context(
    profile: CompanyProfile,
    primary_input: Dict[str, Any],
) -> Dict[str, Any]:
    payload = primary_input or {}
    source_company = (payload.get("sourcing_brief") or {}).get("source_company") or {}
    company_name = (
        source_company.get("name")
        or normalize_domain(profile.buyer_company_url or "")
        or "Source Company"
    )
    graph_ref = company_context_graph_ref(profile.workspace_id, company_name)
    graph = _empty_graph(graph_ref, profile.workspace_id)
    company_id = _stable_id("company", graph_ref, company_name)
    graph["company_node_id"] = company_id
    graph["sourcing_brief"] = deepcopy(payload.get("sourcing_brief") or {})

    company_node = _node(
        company_id,
        "Company",
        name=company_name,
        evidence_tier=PRIMARY_EVIDENCE_TIER,
        source_type=PRIMARY_EVIDENCE_TIER,
        evidence_type="company_profile",
        confidence=1.0,
        properties={
            "website": source_company.get("website") or profile.buyer_company_url,
            "workspace_id": profile.workspace_id,
            "graph_ref": graph_ref,
        },
    )
    nodes: Dict[str, Dict[str, Any]] = {company_node["id"]: company_node}
    edges: Dict[str, Dict[str, Any]] = {}
    graph["source_documents"] = _filter_source_company_documents(
        _first_party_source_documents(payload, graph_ref),
        profile=profile,
    )
    for document in graph["source_documents"]:
        nodes[document["id"]] = document

    default_source_document_id = graph["source_documents"][0]["id"] if graph["source_documents"] else None
    evidence_items_by_id = _first_party_evidence_items(payload)
    source_document_id_by_url = _source_document_id_by_url(graph["source_documents"])

    curated_taxonomy_nodes = _curated_primary_taxonomy_nodes(payload)
    curated_node_ids = {str(item.get("id") or "") for item in curated_taxonomy_nodes}

    for item in curated_taxonomy_nodes:
        if not isinstance(item, dict):
            continue
        label = _taxonomy_label(str(item.get("layer") or ""))
        phrase = _clean_phrase(item.get("phrase"))
        if not label or not phrase:
            continue
        node_id = str(item.get("id") or _stable_id(label.lower(), graph_ref, phrase))
        evidence_ids = [str(value) for value in (item.get("evidence_ids") or []) if str(value).strip()]
        source_document_id = _resolve_source_document_id(
            evidence_ids,
            evidence_items_by_id=evidence_items_by_id,
            source_document_id_by_url=source_document_id_by_url,
            fallback_source_document_id=default_source_document_id,
        )
        node_payload = _node(
            node_id,
            label,
            name=phrase,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="taxonomy",
            confidence=item.get("confidence") or 0.7,
            source_document_id=source_document_id,
            properties={
                "layer": item.get("layer"),
                "aliases": item.get("aliases") or [],
                "scope_status": item.get("scope_status") or "in_scope",
                "evidence_ids": evidence_ids,
                "graph_ref": graph_ref,
            },
        )
        nodes[node_id] = node_payload
        if label == "Capability":
            edge_type = "OFFERS_CAPABILITY"
        elif label == "CustomerArchetype":
            edge_type = "SERVES_CUSTOMER_ARCHETYPE"
        elif label == "Workflow":
            edge_type = "MENTIONS"
        else:
            edge_type = "MENTIONS"
        edge_id = _stable_id(edge_type.lower(), graph_ref, company_id, node_id)
        edges[edge_id] = _edge(
            edge_id,
            edge_type,
            company_id,
            node_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="taxonomy",
            confidence=item.get("confidence") or 0.7,
            source_document_id=source_document_id,
            properties={"graph_ref": graph_ref},
        )
        _add_supported_by_edge(
            edges,
            graph_ref=graph_ref,
            from_id=node_id,
            source_document_id=source_document_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="taxonomy",
            confidence=item.get("confidence") or 0.7,
        )

    for item in payload.get("taxonomy_edges") or []:
        if not isinstance(item, dict):
            continue
        relation_type = str(item.get("relation_type") or "")
        if relation_type == "supports_workflow":
            edge_type = "SUPPORTS_WORKFLOW"
        elif relation_type == "buys_capability":
            edge_type = "SERVES_CUSTOMER_ARCHETYPE"
        else:
            edge_type = "MENTIONS"
        from_id = str(item.get("from_node_id") or "").strip()
        to_id = str(item.get("to_node_id") or "").strip()
        if from_id not in curated_node_ids or to_id not in curated_node_ids or from_id not in nodes or to_id not in nodes:
            continue
        source_document_id = _resolve_source_document_id(
            [str(value) for value in (item.get("evidence_ids") or []) if str(value).strip()],
            evidence_items_by_id=evidence_items_by_id,
            source_document_id_by_url=source_document_id_by_url,
            fallback_source_document_id=default_source_document_id,
        )
        edge_id = _stable_id(edge_type.lower(), graph_ref, from_id, to_id)
        edges[edge_id] = _edge(
            edge_id,
            edge_type,
            from_id,
            to_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="taxonomy_edge",
            confidence=0.72,
            source_document_id=source_document_id,
            properties={"graph_ref": graph_ref},
        )

    brief = payload.get("sourcing_brief") or {}
    for customer in brief.get("named_customer_proof") or []:
        if not isinstance(customer, dict) or not customer.get("name"):
            continue
        customer_evidence_id = str(customer.get("evidence_id") or "").strip()
        source_document_id = _resolve_source_document_id(
            [customer_evidence_id] if customer_evidence_id else [],
            evidence_items_by_id=evidence_items_by_id,
            source_document_id_by_url=source_document_id_by_url,
            fallback_source_document_id=default_source_document_id,
        )
        if not source_document_id and str(customer.get("source_url") or "").strip():
            source_document = _source_document_node(
                graph_ref=graph_ref,
                url=str(customer.get("source_url") or profile.buyer_company_url or ""),
                title=str(customer.get("name")),
                source_type=PRIMARY_EVIDENCE_TIER,
                evidence_tier=PRIMARY_EVIDENCE_TIER,
                evidence_type=str(customer.get("evidence_type") or "customer_logo"),
                snippet=customer.get("context"),
            )
            nodes[source_document["id"]] = source_document
            source_document_id = source_document["id"]
        node_id = _stable_id("customer_entity", graph_ref, customer.get("name"))
        nodes[node_id] = _node(
            node_id,
            "CustomerEntity",
            name=str(customer.get("name")),
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type=str(customer.get("evidence_type") or "customer_logo"),
            confidence=0.68,
            source_document_id=source_document_id,
            properties={
                "context": customer.get("context"),
                "evidence_id": customer_evidence_id or None,
                "graph_ref": graph_ref,
            },
        )
        edge_id = _stable_id("serves_customer_entity", graph_ref, company_id, node_id)
        edges[edge_id] = _edge(
            edge_id,
            "SERVES_CUSTOMER_ENTITY",
            company_id,
            node_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type=str(customer.get("evidence_type") or "customer_logo"),
            confidence=0.68,
            source_document_id=source_document_id,
            properties={"graph_ref": graph_ref},
        )
        _add_supported_by_edge(
            edges,
            graph_ref=graph_ref,
            from_id=node_id,
            source_document_id=source_document_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type=str(customer.get("evidence_type") or "customer_logo"),
            confidence=0.68,
        )

    for partner in (brief.get("partner_integration_proof") or []):
        if not isinstance(partner, dict) or not partner.get("name"):
            continue
        partner_evidence_id = str(partner.get("evidence_id") or "").strip()
        source_document_id = _resolve_source_document_id(
            [partner_evidence_id] if partner_evidence_id else [],
            evidence_items_by_id=evidence_items_by_id,
            source_document_id_by_url=source_document_id_by_url,
            fallback_source_document_id=default_source_document_id,
        )
        if not source_document_id and str(partner.get("source_url") or "").strip():
            source_document = _source_document_node(
                graph_ref=graph_ref,
                url=str(partner.get("source_url") or profile.buyer_company_url or ""),
                title=str(partner.get("name")),
                source_type=PRIMARY_EVIDENCE_TIER,
                evidence_tier=PRIMARY_EVIDENCE_TIER,
                evidence_type="integration_partner",
            )
            nodes[source_document["id"]] = source_document
            source_document_id = source_document["id"]
        node_id = _stable_id("partner_entity", graph_ref, partner.get("name"))
        nodes[node_id] = _node(
            node_id,
            "PartnerEntity",
            name=str(partner.get("name")),
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="integration_partner",
            confidence=0.72,
            source_document_id=source_document_id,
            properties={"evidence_id": partner_evidence_id or None, "graph_ref": graph_ref},
        )
        edge_id = _stable_id("integrates_with", graph_ref, company_id, node_id)
        edges[edge_id] = _edge(
            edge_id,
            "INTEGRATES_WITH",
            company_id,
            node_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="integration_partner",
            confidence=0.72,
            source_document_id=source_document_id,
            properties={"graph_ref": graph_ref},
        )
        _add_supported_by_edge(
            edges,
            graph_ref=graph_ref,
            from_id=node_id,
            source_document_id=source_document_id,
            evidence_tier=PRIMARY_EVIDENCE_TIER,
            source_type=PRIMARY_EVIDENCE_TIER,
            evidence_type="integration_partner",
            confidence=0.72,
        )

    claim_source_document_id = default_source_document_id
    for hypothesis in brief.get("adjacency_hypotheses") or []:
        text = _clean_phrase(hypothesis.get("text"))
        if not text:
            continue
        node_id = _stable_id("claim", graph_ref, text)
        nodes[node_id] = _node(
            node_id,
            "Claim",
            name=text,
            evidence_tier=INFERRED_EVIDENCE_TIER,
            source_type=INFERRED_EVIDENCE_TIER,
            evidence_type="adjacency_hypothesis",
            confidence=hypothesis.get("confidence") or 0.55,
            source_document_id=claim_source_document_id,
            properties={
                "claim_type": "adjacency_hypothesis",
                "supporting_node_ids": hypothesis.get("supporting_node_ids") or [],
                "evidence_ids": hypothesis.get("evidence_ids") or [],
                "graph_ref": graph_ref,
            },
        )
        edge_id = _stable_id("mentions", graph_ref, company_id, node_id)
        edges[edge_id] = _edge(
            edge_id,
            "MENTIONS",
            company_id,
            node_id,
            evidence_tier=INFERRED_EVIDENCE_TIER,
            source_type=INFERRED_EVIDENCE_TIER,
            evidence_type="adjacency_hypothesis",
            confidence=hypothesis.get("confidence") or 0.55,
            source_document_id=claim_source_document_id,
            properties={"graph_ref": graph_ref},
        )
        _add_supported_by_edge(
            edges,
            graph_ref=graph_ref,
            from_id=node_id,
            source_document_id=claim_source_document_id,
            evidence_tier=INFERRED_EVIDENCE_TIER,
            source_type=INFERRED_EVIDENCE_TIER,
            evidence_type="adjacency_hypothesis",
            confidence=hypothesis.get("confidence") or 0.55,
        )

    graph["nodes"] = list(nodes.values())
    graph["edges"] = list(edges.values())
    graph["graph_stats"] = {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "source_document_count": len(graph["source_documents"]),
        "primary_evidence_count": len(graph["source_documents"]),
        "secondary_evidence_count": 0,
    }
    return _canonicalize_graph(graph)


def _secondary_query_text(company_name: str, phrase: Optional[str], *, domain: Optional[str] = None) -> str:
    parts = [company_name]
    if phrase:
        parts.append(f"\"{phrase}\"")
    if domain:
        parts.append(f"-site:{domain}")
    return " ".join([part for part in parts if part]).strip()


def _quoted_query_terms(*terms: Optional[str]) -> str:
    return " ".join([f"\"{value}\"" for value in [_clean_phrase(term) for term in terms] if value]).strip()


def _secondary_context_anchor_terms(primary_graph: Dict[str, Any], limit: int = 2) -> List[str]:
    return _top_graph_node_names(primary_graph, {"Capability", "Workflow"}, limit)


def _secondary_company_identity_terms(
    company_name: str,
    company_domain: str,
    context_terms: Iterable[str],
) -> List[str]:
    values: List[str] = []
    company_phrase = _clean_phrase(company_name)
    if company_phrase:
        values.append(company_phrase)
    if _is_ambiguous_company_name(company_name, company_domain):
        domain_phrase = _clean_phrase(company_domain)
        if domain_phrase and domain_phrase not in values:
            values.append(domain_phrase)
        domain_stem = _clean_phrase((company_domain.split(".")[0] if company_domain else ""))
        if domain_stem and domain_stem not in values:
            values.append(domain_stem)
        for term in context_terms:
            phrase = _clean_phrase(term)
            if phrase and phrase not in values:
                values.append(phrase)
                break
    return values


def _secondary_entity_query_text(
    company_name: str,
    entity_name: str,
    *,
    qualifier: Optional[str] = None,
    relationship_terms: Iterable[str] = (),
) -> str:
    parts: List[str] = []
    quoted = _quoted_query_terms(company_name, entity_name)
    if quoted:
        parts.append(quoted)
    qualifier_phrase = _clean_phrase(qualifier)
    if qualifier_phrase:
        parts.append(qualifier_phrase)
    for term in relationship_terms:
        token = _clean_phrase(term)
        if token:
            parts.append(token)
    return " ".join(parts).strip()


def _normalized_identity_tokens(company_name: str, company_domain: str) -> List[str]:
    values: List[str] = []
    company_token = _normalize_taxonomy_identity(company_name).replace(" ", "")
    if company_token:
        values.append(company_token)
    domain_phrase = _clean_phrase(company_domain).lower()
    if domain_phrase:
        values.append(re.sub(r"[^a-z0-9]+", "", domain_phrase))
    domain_stem = _clean_phrase((company_domain.split(".")[0] if company_domain else "")).lower()
    if domain_stem:
        values.append(re.sub(r"[^a-z0-9]+", "", domain_stem))
    unique: List[str] = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def _secondary_company_profile_identity_match(
    url: str,
    title: str,
    company_name: str,
    company_domain: str,
) -> bool:
    host = normalize_domain(url) or ""
    if not any(token in host for token in COMPANY_PROFILE_HOST_TOKENS):
        return False
    parsed = urlparse(url)
    path_blob = re.sub(r"[^a-z0-9]+", "", parsed.path.lower())
    title_blob = re.sub(r"[^a-z0-9]+", "", str(title or "").lower())
    for token in _normalized_identity_tokens(company_name, company_domain):
        if token and (token in path_blob or token in title_blob):
            return True
    return False


def _top_graph_node_names(primary_graph: Dict[str, Any], labels: set[str], limit: int) -> List[str]:
    values: List[str] = []
    for node in sorted(
        [
            node
            for node in (primary_graph.get("nodes") or [])
            if isinstance(node, dict) and str(node.get("label") or "") in labels
        ],
        key=lambda item: float(item.get("confidence") or 0),
        reverse=True,
    ):
        phrase = _clean_phrase(node.get("name"))
        if phrase and phrase not in values:
            values.append(phrase)
        if len(values) >= limit:
            break
    return values


def _is_counterparty_like_name(name: str) -> bool:
    phrase = _clean_phrase(name).lower()
    if not phrase:
        return False
    compact = re.sub(r"[^a-z0-9]+", " ", phrase).strip()
    if not compact:
        return False
    if compact in GENERIC_INTEGRATION_NAME_TOKENS:
        return False
    tokens = compact.split()
    if any(token in GENERIC_INTEGRATION_NAME_TOKENS for token in tokens):
        return False
    return True


def _build_secondary_queries(primary_graph: Dict[str, Any], comparator_urls: Iterable[str]) -> List[Dict[str, Any]]:
    settings = get_settings()
    company = next((node for node in primary_graph.get("nodes", []) if node.get("label") == "Company"), None)
    company_name = str((company or {}).get("name") or "Source Company")
    domain = normalize_domain((company or {}).get("website") or "")
    query_cap = max(8, int(settings.company_context_secondary_query_cap))
    top_phrases = _top_graph_node_names(primary_graph, {"Capability", "Workflow", "CustomerArchetype"}, 4)
    customer_names = _top_graph_node_names(primary_graph, {"CustomerEntity"}, 8)
    partner_names = [
        name
        for name in _top_graph_node_names(primary_graph, {"PartnerEntity"}, 8)
        if _is_counterparty_like_name(name)
    ][:4]
    qualifier_terms = _secondary_context_anchor_terms(primary_graph, 2)
    identity_terms = _secondary_company_identity_terms(company_name, domain, qualifier_terms)
    identity_query = _quoted_query_terms(*identity_terms[:2]) or _quoted_query_terms(company_name)
    company_profile_blocklist = list(COMPANY_PROFILE_HOSTS)
    queries: List[Dict[str, Any]] = []
    query_keys: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()

    def _append_query(query: Dict[str, Any]) -> None:
        key = (
            str(query.get("query_type") or ""),
            str(query.get("query_text") or ""),
            tuple(str(item) for item in (query.get("domain_allowlist") or [])),
            tuple(str(item) for item in (query.get("domain_blocklist") or [])),
        )
        if key in query_keys:
            return
        query_keys.add(key)
        queries.append(query)

    for customer_name in customer_names:
        customer_phrase = _secondary_entity_query_text(
            company_name,
            customer_name,
        )
        _append_query(
            {
                "query_id": _stable_id("secondary_query_customer", company_name, customer_name),
                "query_text": customer_phrase,
                "query_type": "customer_corroboration",
                "must_include_terms": [company_name, customer_name],
                "domain_blocklist": ([domain] if domain else []) + company_profile_blocklist,
            }
        )

    for partner_name in partner_names:
        partner_phrase = _secondary_entity_query_text(
            company_name,
            partner_name,
        )
        _append_query(
            {
                "query_id": _stable_id("secondary_query_partner", company_name, partner_name),
                "query_text": partner_phrase,
                "query_type": "partner_corroboration",
                "must_include_terms": [company_name, partner_name],
                "domain_blocklist": ([domain] if domain else []) + company_profile_blocklist,
            }
        )

    for host in DIRECTORY_PRIORITY_HOSTS[:2]:
        _append_query(
            {
                "query_id": _stable_id("secondary_query_directory_host", company_name, host),
                "query_text": identity_query,
                "query_type": "directory_category",
                "must_include_terms": identity_terms[:2] or [company_name],
                "domain_allowlist": [host],
                "domain_blocklist": ([domain] if domain else []) + company_profile_blocklist,
            }
        )

    _append_query(
        {
            "query_id": _stable_id("secondary_query_market_identity", company_name, domain),
            "query_text": identity_query,
            "query_type": "market_context",
            "must_include_terms": [company_name],
            "domain_blocklist": [domain] if domain else [],
        }
    )

    for phrase in top_phrases[:2]:
        _append_query(
            {
                "query_id": _stable_id("secondary_query_market", company_name, phrase),
                "query_text": _quoted_query_terms(company_name, phrase, qualifier_terms[0] if qualifier_terms else None),
                "query_type": "market_context",
                "must_include_terms": [company_name],
                "domain_blocklist": [domain] if domain else [],
            }
        )
    return queries[:query_cap]


def _classify_secondary_source(url: str, title: str, snippet: str, query_type: str | None = None) -> Tuple[str, str]:
    if query_type and query_type in SECONDARY_QUERY_TYPE_BUCKETS:
        return SECONDARY_QUERY_TYPE_BUCKETS[query_type]
    host = normalize_domain(url) or ""
    content = f"{title} {snippet}".lower()
    if any(token in host for token in DIRECTORY_HOST_TOKENS):
        return "directory", "category_positioning"
    if any(token in content for token in PARTNER_TOKENS):
        return "partner_page", "integration_announcement"
    if any(token in content for token in DEPLOYMENT_TOKENS):
        return "press_release", "deployment_announcement"
    if any(token in content for token in PRESS_RELEASE_TOKENS):
        return "press_release", "customer_quote"
    return "trade_publication", "workflow_description"


def _extract_category_name(url: str, title: str) -> Optional[str]:
    parsed = urlparse(url)
    path_bits = [bit for bit in parsed.path.split("/") if bit]
    if path_bits:
        candidate = path_bits[-1].replace("-", " ").replace("_", " ").strip()
        if candidate and len(candidate.split()) <= 8:
            return candidate.title()
    if "|" in title:
        return title.split("|", 1)[0].strip()
    return None


def _extract_mentions(text: str, candidates: Iterable[str]) -> List[str]:
    lowered = f" {text.lower()} "
    found: List[str] = []
    for candidate in candidates:
        phrase = _clean_phrase(candidate)
        if not phrase:
            continue
        if f" {phrase.lower()} " in lowered and phrase not in found:
            found.append(phrase)
    return found


def _secondary_signal_quality(
    *,
    url: str,
    title: str,
    snippet: str,
    content: str,
    company_name: str,
    source_type: str,
    claim_type: str,
    query_type: str,
    matched_customers: List[str],
    matched_partners: List[str],
    matched_context_terms: List[str],
    company_domain: str,
) -> str:
    host = normalize_domain(url) or ""
    text_blob = " ".join([title, snippet, content]).lower()
    company_token = str(company_name or "").strip().lower()
    if any(token in host for token in LOW_SIGNAL_HOST_TOKENS):
        return "drop"
    if any(token in text_blob for token in FALSE_POSITIVE_SCIENCE_TOKENS):
        return "drop"
    if query_type in {"customer_corroboration", "partner_corroboration"} and any(
        token in host for token in LOW_SIGNAL_CORROBORATION_HOST_TOKENS
    ):
        return "drop"
    if any(token in text_blob for token in LOW_SIGNAL_SECONDARY_TEXT_TOKENS):
        return "document_only"
    if _secondary_company_profile_identity_match(url, title, company_name, company_domain):
        if query_type in {"customer_corroboration", "partner_corroboration", "directory_category"}:
            return "document_only"
        return "strong"
    identity_strength = _company_identity_match_strength(
        company_name=company_name,
        company_domain=company_domain,
        title=title,
        snippet=snippet,
        content=content,
        matched_context_terms=matched_context_terms,
    )
    if identity_strength == "weak":
        return "drop"
    if company_token and company_token not in text_blob and (normalize_domain(url) or "") != company_token and identity_strength != "strong":
        return "document_only"
    if source_type == "directory":
        if not any(token in host for token in DIRECTORY_HOST_TOKENS):
            return "document_only"
        return "strong"
    if query_type == "customer_corroboration":
        if not matched_customers:
            return "drop"
        if not any(token in text_blob for token in (*DEPLOYMENT_TOKENS, *PRESS_RELEASE_TOKENS)):
            return "drop"
    if query_type == "partner_corroboration":
        if not matched_partners:
            return "drop"
        if not any(token in text_blob for token in (*PARTNER_TOKENS, *PRESS_RELEASE_TOKENS)):
            return "drop"
    if claim_type == "deployment_announcement" and not matched_customers:
        return "document_only"
    if claim_type == "integration_announcement" and not matched_partners:
        return "document_only"
    if len(text_blob.strip()) < 80:
        return "document_only"
    return "strong"


def _build_secondary_company_graph(
    profile: CompanyProfile,
    primary_graph: Dict[str, Any],
) -> Dict[str, Any]:
    settings = get_settings()
    graph = _empty_graph(primary_graph["graph_ref"], profile.workspace_id)
    queries = _build_secondary_queries(primary_graph, profile.comparator_seed_urls or [])
    if not queries:
        return graph
    provider_order = [
        token.strip().lower()
        for token in str(settings.company_context_secondary_provider_order or "").split(",")
        if token.strip()
    ]
    retrieval = run_external_search_queries(
        queries,
        provider_order=provider_order,
        per_query_cap=max(2, int(settings.company_context_secondary_per_query_cap)),
        total_cap=max(4, int(settings.company_context_secondary_result_cap)),
        per_domain_cap=max(1, int(settings.company_context_secondary_per_domain_cap)),
        max_seconds=max(5, int(settings.company_context_secondary_max_seconds)),
    )
    if not retrieval.get("results"):
        graph["graph_stats"]["secondary_evidence_count"] = 0
        graph["retrieval"] = retrieval
        return graph

    company = next((node for node in primary_graph.get("nodes", []) if node.get("label") == "Company"), None)
    company_id = str((company or {}).get("id") or "")
    company_name = str((company or {}).get("name") or "Source Company")
    candidate_customer_names = [
        str(node.get("name"))
        for node in primary_graph.get("nodes", [])
        if node.get("label") == "CustomerEntity"
    ]
    candidate_partner_names = [
        str(node.get("name"))
        for node in primary_graph.get("nodes", [])
        if node.get("label") == "PartnerEntity"
    ]
    context_terms = _primary_context_terms(primary_graph)
    company_domain = normalize_domain((company or {}).get("website") or profile.buyer_company_url or "")

    nodes: Dict[str, Dict[str, Any]] = {}
    edges: Dict[str, Dict[str, Any]] = {}
    secondary_evidence_proof: List[Dict[str, Any]] = []
    source_documents: List[Dict[str, Any]] = []

    for result in retrieval.get("results") or []:
        url = str(result.get("url") or "").strip()
        if not url:
            continue
        host = normalize_domain(url) or ""
        if company_domain and host == company_domain:
            continue
        if any(token in host for token in LOW_SIGNAL_HOST_TOKENS):
            continue
        title = _clean_phrase(result.get("title"))
        snippet = _clean_phrase(result.get("snippet"))
        page = fetch_page_fast(url)
        content = _clean_phrase(page.get("content"))[:5000]
        query_type = str(result.get("query_type") or "").strip()
        source_type, claim_type = _classify_secondary_source(url, title, snippet or content[:250], query_type)
        query_type = query_type or SOURCE_TYPE_TO_QUERY_TYPE.get(source_type, "market_context")
        evidence_snippet = snippet or content[:320]
        document = _source_document_node(
            graph_ref=graph["graph_ref"],
            url=url,
            title=title or host,
            source_type=source_type,
            evidence_tier=SECONDARY_EVIDENCE_TIER,
            evidence_type=claim_type,
            publisher=host,
            snippet=evidence_snippet,
        )
        document["publisher_channel"] = source_type
        document["publisher_type"] = (
            "customer_or_partner"
            if source_type in {"partner_page", "press_release"}
            else "directory"
            if source_type == "directory"
            else "third_party"
        )
        document["claim_scope"] = "about_subject_company"
        document["subject_company"] = company_name
        nodes[document["id"]] = document
        source_documents.append(document)

        text_blob = " ".join([title, snippet, content])
        matched_customers = _extract_mentions(text_blob, candidate_customer_names)
        matched_partners = _extract_mentions(text_blob, candidate_partner_names)
        quality = _secondary_signal_quality(
            url=url,
            title=title,
            snippet=snippet,
            content=content,
            company_name=company_name,
            source_type=source_type,
            claim_type=claim_type,
            query_type=query_type,
            matched_customers=matched_customers,
            matched_partners=matched_partners,
            matched_context_terms=context_terms,
            company_domain=company_domain,
        )

        claim_text = evidence_snippet or title
        if not claim_text or quality == "drop":
            continue
        if quality == "document_only":
            continue
        claim_id = _stable_id("claim", graph["graph_ref"], url, claim_type)
        nodes[claim_id] = _node(
            claim_id,
            "Claim",
            name=claim_text,
            evidence_tier=SECONDARY_EVIDENCE_TIER,
            source_type=source_type,
            evidence_type=claim_type,
            confidence=0.66 if source_type == "directory" else 0.74,
            source_document_id=document["id"],
            properties={"claim_type": claim_type, "graph_ref": graph["graph_ref"]},
        )
        edge_id = _stable_id("supported_by", graph["graph_ref"], claim_id, document["id"])
        edges[edge_id] = _edge(
            edge_id,
            "SUPPORTED_BY",
            claim_id,
            document["id"],
            evidence_tier=SECONDARY_EVIDENCE_TIER,
            source_type=source_type,
            evidence_type=claim_type,
            confidence=0.8,
            source_document_id=document["id"],
            properties={"graph_ref": graph["graph_ref"]},
        )
        mention_edge_id = _stable_id("mentions", graph["graph_ref"], company_id, claim_id)
        edges[mention_edge_id] = _edge(
            mention_edge_id,
            "MENTIONS",
            company_id,
            claim_id,
            evidence_tier=SECONDARY_EVIDENCE_TIER,
            source_type=source_type,
            evidence_type=claim_type,
            confidence=0.7,
            source_document_id=document["id"],
            properties={"graph_ref": graph["graph_ref"]},
        )

        if source_type == "directory":
            category_name = _extract_category_name(url, title)
            if category_name and len(category_name) >= 4 and category_name.lower() not in {"email", "companies", "company"}:
                category_id = _stable_id("category", graph["graph_ref"], category_name)
                nodes[category_id] = _node(
                    category_id,
                    "Category",
                    name=category_name,
                    evidence_tier=SECONDARY_EVIDENCE_TIER,
                    source_type=source_type,
                    evidence_type="category_positioning",
                    confidence=0.7,
                    source_document_id=document["id"],
                    properties={"graph_ref": graph["graph_ref"]},
                )
                list_edge_id = _stable_id("listed_in_category", graph["graph_ref"], company_id, category_id)
                edges[list_edge_id] = _edge(
                    list_edge_id,
                    "LISTED_IN_CATEGORY",
                    company_id,
                    category_id,
                    evidence_tier=SECONDARY_EVIDENCE_TIER,
                    source_type=source_type,
                    evidence_type="category_positioning",
                    confidence=0.7,
                    source_document_id=document["id"],
                    properties={"graph_ref": graph["graph_ref"]},
                )

        if claim_type == "deployment_announcement":
            for customer_name in matched_customers[:2]:
                customer_id = _stable_id("customer_entity", graph["graph_ref"], customer_name)
                if customer_id not in nodes:
                    nodes[customer_id] = _node(
                        customer_id,
                        "CustomerEntity",
                        name=customer_name,
                        evidence_tier=SECONDARY_EVIDENCE_TIER,
                        source_type=source_type,
                        evidence_type=claim_type,
                        confidence=0.74,
                        source_document_id=document["id"],
                        properties={"graph_ref": graph["graph_ref"]},
                    )
                announce_edge_id = _stable_id("announced_by_customer", graph["graph_ref"], customer_id, company_id)
                edges[announce_edge_id] = _edge(
                    announce_edge_id,
                    "ANNOUNCED_BY_CUSTOMER",
                    customer_id,
                    company_id,
                    evidence_tier=SECONDARY_EVIDENCE_TIER,
                    source_type=source_type,
                    evidence_type=claim_type,
                    confidence=0.74,
                    source_document_id=document["id"],
                    properties={"graph_ref": graph["graph_ref"]},
                )
        if claim_type == "integration_announcement":
            for partner_name in matched_partners[:2]:
                partner_id = _stable_id("partner_entity", graph["graph_ref"], partner_name)
                if partner_id not in nodes:
                    nodes[partner_id] = _node(
                        partner_id,
                        "PartnerEntity",
                        name=partner_name,
                        evidence_tier=SECONDARY_EVIDENCE_TIER,
                        source_type=source_type,
                        evidence_type=claim_type,
                        confidence=0.74,
                        source_document_id=document["id"],
                        properties={"graph_ref": graph["graph_ref"]},
                    )
                integration_edge_id = _stable_id("integrates_with", graph["graph_ref"], company_id, partner_id)
                edges[integration_edge_id] = _edge(
                    integration_edge_id,
                    "INTEGRATES_WITH",
                    company_id,
                    partner_id,
                    evidence_tier=SECONDARY_EVIDENCE_TIER,
                    source_type=source_type,
                    evidence_type=claim_type,
                    confidence=0.74,
                    source_document_id=document["id"],
                    properties={"graph_ref": graph["graph_ref"]},
                )

        secondary_evidence_proof.append(
            {
                "id": _stable_id("external_proof", graph["graph_ref"], url, claim_type),
                "query_type": query_type,
                "publisher_channel": source_type,
                "publisher_type": (
                    "customer_or_partner"
                    if source_type in {"partner_page", "press_release"}
                    else "directory"
                    if source_type == "directory"
                    else "third_party"
                ),
                "claim_scope": "about_subject_company",
                "subject_company": company_name,
                "claim_type": claim_type,
                "publisher": host,
                "published_at": None,
                "claim_text": claim_text[:500],
                "evidence_snippet": evidence_snippet[:500],
                "url": url,
                "title": title or host,
                "entity_mentions": [company_name, *matched_customers[:2], *matched_partners[:2]],
                "supports_node_ids": [claim_id],
                "supports_edge_ids": [edge_id],
                "confidence": 0.66 if source_type == "directory" else 0.74,
                "freshness": "unknown",
                "evidence_tier": SECONDARY_EVIDENCE_TIER,
            }
        )

    graph["nodes"] = list(nodes.values())
    graph["edges"] = list(edges.values())
    graph["source_documents"] = source_documents
    graph["secondary_evidence_proof"] = secondary_evidence_proof[:8]
    graph["retrieval"] = retrieval
    graph["graph_stats"] = {
        "node_count": len(graph["nodes"]),
        "edge_count": len(graph["edges"]),
        "source_document_count": len(source_documents),
        "primary_evidence_count": 0,
        "secondary_evidence_count": len(secondary_evidence_proof),
        "provider_mix": retrieval.get("provider_mix") or {},
    }
    return graph


def merge_company_graphs(primary_graph: Dict[str, Any], secondary_graph: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(primary_graph)
    node_map = {node["id"]: deepcopy(node) for node in primary_graph.get("nodes", []) if isinstance(node, dict)}
    edge_map = {edge["id"]: deepcopy(edge) for edge in primary_graph.get("edges", []) if isinstance(edge, dict)}
    for node in secondary_graph.get("nodes", []) or []:
        if isinstance(node, dict):
            node_map.setdefault(node["id"], deepcopy(node))
    for edge in secondary_graph.get("edges", []) or []:
        if isinstance(edge, dict):
            edge_map.setdefault(edge["id"], deepcopy(edge))
    source_docs = {
        item["id"]: deepcopy(item)
        for item in (primary_graph.get("source_documents") or [])
        if isinstance(item, dict) and item.get("id")
    }
    for item in secondary_graph.get("source_documents") or []:
        if isinstance(item, dict) and item.get("id"):
            source_docs.setdefault(item["id"], deepcopy(item))

    merged["nodes"] = list(node_map.values())
    merged["edges"] = list(edge_map.values())
    merged["source_documents"] = list(source_docs.values())
    merged["secondary_evidence_proof"] = deepcopy(secondary_graph.get("secondary_evidence_proof") or [])
    primary_stats = primary_graph.get("graph_stats") or {}
    secondary_stats = secondary_graph.get("graph_stats") or {}
    merged["graph_stats"] = {
        "node_count": len(merged["nodes"]),
        "edge_count": len(merged["edges"]),
        "source_document_count": len(merged["source_documents"]),
        "primary_evidence_count": int(primary_stats.get("primary_evidence_count") or 0),
        "secondary_evidence_count": int(secondary_stats.get("secondary_evidence_count") or 0),
        "provider_mix": secondary_stats.get("provider_mix") or {},
    }
    return _canonicalize_graph(merged)


def generate_sourcing_brief_from_graph(
    graph: Dict[str, Any],
    *,
    base_brief: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    brief = deepcopy(base_brief or graph.get("sourcing_brief") or {})
    nodes = [node for node in graph.get("nodes", []) if isinstance(node, dict)]

    def _top_nodes(label: str, limit: int) -> List[Dict[str, Any]]:
        return [
            {
                "id": node.get("id"),
                "layer": {
                    "CustomerArchetype": "customer_archetype",
                    "Workflow": "workflow",
                    "Capability": "capability",
                    "DeliveryIntegration": "delivery_or_integration",
                }.get(label, label.lower()),
                "phrase": node.get("name"),
                "aliases": node.get("aliases") or [],
                "confidence": _confidence(node.get("confidence")),
                "evidence_ids": [node.get("source_document_id")] if node.get("source_document_id") else [],
                "scope_status": node.get("scope_status") or "in_scope",
            }
            for node in sorted(
                [node for node in nodes if node.get("label") == label and node.get("scope_status") != "removed"],
                key=lambda item: float(item.get("confidence") or 0),
                reverse=True,
            )[:limit]
        ]

    brief["customer_nodes"] = _top_nodes("CustomerArchetype", 4)
    brief["workflow_nodes"] = _top_nodes("Workflow", 4)
    brief["capability_nodes"] = _top_nodes("Capability", 6)
    brief["delivery_or_integration_nodes"] = _top_nodes("DeliveryIntegration", 4)
    brief["named_customer_proof"] = sorted(
        [
            {
                "name": node.get("name"),
                "source_url": next(
                    (
                        doc.get("url")
                        for doc in graph.get("source_documents", [])
                        if doc.get("id") == node.get("source_document_id")
                    ),
                    None,
                ),
                "context": node.get("context"),
                "evidence_type": node.get("evidence_type"),
                "evidence_id": node.get("source_document_id"),
                "evidence_tier": node.get("evidence_tier"),
            }
            for node in nodes
            if node.get("label") == "CustomerEntity"
        ],
        key=lambda item: (item.get("evidence_tier") != PRIMARY_EVIDENCE_TIER, item.get("name") or ""),
    )[:10]
    brief["partner_integration_proof"] = [
        {
            "name": node.get("name"),
            "source_url": next(
                (
                    doc.get("url")
                    for doc in graph.get("source_documents", [])
                    if doc.get("id") == node.get("source_document_id")
                ),
                None,
            ),
            "evidence_id": node.get("source_document_id"),
            "evidence_tier": node.get("evidence_tier"),
        }
        for node in nodes
        if node.get("label") == "PartnerEntity"
    ][:10]
    secondary_evidence = deepcopy(graph.get("secondary_evidence_proof") or [])
    brief["secondary_evidence_proof"] = secondary_evidence
    brief["customer_partner_corroboration"] = [
        item
        for item in secondary_evidence
        if str(item.get("publisher_type") or "") == "customer_or_partner"
    ][:8]
    brief["directory_category_context"] = [
        item
        for item in secondary_evidence
        if str(item.get("publisher_type") or "") == "directory"
    ][:8]
    brief["other_secondary_context"] = [
        item
        for item in secondary_evidence
        if str(item.get("publisher_type") or "") not in {"customer_or_partner", "directory"}
    ][:8]
    brief["strongest_evidence_buckets"] = [
        {"label": "First-party source documents", "count": int(graph.get("graph_stats", {}).get("primary_evidence_count") or 0)},
        {"label": "Secondary public evidence", "count": int(graph.get("graph_stats", {}).get("secondary_evidence_count") or 0)},
    ]
    unknowns = []
    if not brief["secondary_evidence_proof"]:
        unknowns.append("Secondary public deployment, category, or partner corroboration remains limited.")
    if not brief.get("named_customer_proof"):
        unknowns.append("No named customers are publicly evidenced yet from first-party or external sources.")
    unknowns.extend(
        [
            "Ownership, dealability, and customer concentration usually remain non-public at this stage.",
            "Commercial win/loss dynamics and implementation depth generally require customer or employee conversations.",
        ]
    )
    brief["unknowns_not_publicly_resolvable"] = unknowns[:4]
    brief.setdefault("source_summary", (base_brief or {}).get("source_summary"))
    brief.setdefault("confidence_gaps", (base_brief or {}).get("confidence_gaps") or [])
    brief.setdefault("open_questions", (base_brief or {}).get("open_questions") or [])
    brief.setdefault("active_lenses", (base_brief or {}).get("active_lenses") or [])
    brief.setdefault("adjacency_hypotheses", (base_brief or {}).get("adjacency_hypotheses") or [])
    brief["crawl_coverage"] = (base_brief or {}).get("crawl_coverage") or {}
    return brief


def build_deep_research_handoff(
    graph: Dict[str, Any],
    *,
    brief: Optional[Dict[str, Any]] = None,
    expansion_inputs: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    resolved_brief = deepcopy(brief or graph.get("sourcing_brief") or {})
    source_company = deepcopy(resolved_brief.get("source_company") or {})
    source_documents = [
        item
        for item in (graph.get("source_documents") or [])
        if isinstance(item, dict)
    ]

    def _document_stub(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": item.get("id"),
            "name": item.get("name"),
            "url": item.get("url"),
            "publisher": item.get("publisher"),
            "publisher_channel": item.get("publisher_channel") or item.get("source_type"),
            "publisher_type": item.get("publisher_type"),
            "evidence_tier": item.get("evidence_tier"),
            "evidence_type": item.get("evidence_type"),
            "snippet": item.get("snippet"),
        }

    def _expansion_input_stub(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": item.get("name"),
            "website": item.get("website"),
            "source_summary": item.get("source_summary"),
            "top_capabilities": [
                node.get("phrase")
                for node in (item.get("taxonomy_nodes") or [])
                if isinstance(node, dict) and node.get("layer") == "capability" and node.get("phrase")
            ][:6],
            "top_customer_segments": [
                node.get("phrase")
                for node in (item.get("taxonomy_nodes") or [])
                if isinstance(node, dict) and node.get("layer") == "customer_archetype" and node.get("phrase")
            ][:6],
            "named_customer_count": len(item.get("named_customer_proof") or []),
            "partner_count": len(item.get("partner_integration_proof") or []),
        }

    primary_documents = [
        _document_stub(item)
        for item in source_documents
        if str(item.get("evidence_tier") or "") == PRIMARY_EVIDENCE_TIER
    ][:12]
    secondary_documents = [
        _document_stub(item)
        for item in source_documents
        if str(item.get("evidence_tier") or "") == SECONDARY_EVIDENCE_TIER
    ][:12]

    return {
        "graph_ref": graph.get("graph_ref"),
        "workspace_id": graph.get("workspace_id"),
        "graph_stats": deepcopy(graph.get("graph_stats") or {}),
        "source_company_truth": {
            "source_company": source_company,
            "source_summary": resolved_brief.get("source_summary"),
            "customer_nodes": deepcopy(resolved_brief.get("customer_nodes") or []),
            "workflow_nodes": deepcopy(resolved_brief.get("workflow_nodes") or []),
            "capability_nodes": deepcopy(resolved_brief.get("capability_nodes") or []),
            "delivery_or_integration_nodes": deepcopy(resolved_brief.get("delivery_or_integration_nodes") or []),
            "named_customer_proof": deepcopy(resolved_brief.get("named_customer_proof") or []),
            "partner_integration_proof": deepcopy(resolved_brief.get("partner_integration_proof") or []),
            "primary_source_documents": primary_documents,
        },
        "secondary_context_about_source_company": {
            "customer_partner_corroboration": deepcopy(resolved_brief.get("customer_partner_corroboration") or []),
            "directory_category_context": deepcopy(resolved_brief.get("directory_category_context") or []),
            "other_secondary_context": deepcopy(resolved_brief.get("other_secondary_context") or []),
            "secondary_source_documents": secondary_documents,
        },
        "adjacent_market_inputs": [
            _expansion_input_stub(item)
            for item in (expansion_inputs or [])[:6]
            if isinstance(item, dict)
        ],
        "active_lenses": deepcopy(resolved_brief.get("active_lenses") or []),
        "adjacency_hypotheses": deepcopy(resolved_brief.get("adjacency_hypotheses") or []),
        "confidence_gaps": deepcopy(resolved_brief.get("confidence_gaps") or []),
        "open_questions": deepcopy(resolved_brief.get("open_questions") or []),
        "unknowns_not_publicly_resolvable": deepcopy(
            resolved_brief.get("unknowns_not_publicly_resolvable") or []
        ),
    }


def build_company_context_graph(
    profile: CompanyProfile,
    *,
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    primary_input = payload or build_company_context_artifacts(profile)
    primary_graph = build_primary_company_graph_from_context(profile, primary_input)
    secondary_graph = _build_secondary_company_graph(profile, primary_graph)
    merged_graph = merge_company_graphs(primary_graph, secondary_graph)
    merged_graph["sourcing_brief"] = generate_sourcing_brief_from_graph(
        merged_graph,
        base_brief=(primary_input or {}).get("sourcing_brief") or primary_graph.get("sourcing_brief") or {},
    )
    return _canonicalize_graph(merged_graph)


class Neo4jCompanyContextGraphStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._driver = None

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.neo4j_uri
            and self.settings.neo4j_username
            and self.settings.neo4j_password
        )

    def _driver_or_none(self):
        if not self.configured:
            return None
        if GraphDatabase is None:
            return None
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password),
            )
        return self._driver

    def sync_graph(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        driver = self._driver_or_none()
        if driver is None:
            return {
                "status": "not_configured",
                "error": "Neo4j is not configured",
                "graph_ref": graph.get("graph_ref"),
            }
        graph_ref = str(graph.get("graph_ref") or "")
        database = self.settings.neo4j_database or None
        try:
            with driver.session(database=database) as session:
                session.run("MATCH (n {graph_ref: $graph_ref}) DETACH DELETE n", graph_ref=graph_ref)
                nodes_by_label: Dict[str, List[Dict[str, Any]]] = {}
                for node in graph.get("nodes", []) or []:
                    label = str(node.get("label") or "").strip()
                    if label not in GRAPH_NODE_LABELS:
                        continue
                    payload = deepcopy(node)
                    payload["graph_ref"] = graph_ref
                    nodes_by_label.setdefault(label, []).append(payload)
                for label, rows in nodes_by_label.items():
                    session.run(
                        f"UNWIND $rows AS row CREATE (n:CompanyContext:{label}) SET n = row",
                        rows=rows,
                    )
                edges_by_type: Dict[str, List[Dict[str, Any]]] = {}
                for edge in graph.get("edges", []) or []:
                    edge_type = str(edge.get("type") or "").strip()
                    if edge_type not in GRAPH_EDGE_TYPES:
                        continue
                    edges_by_type.setdefault(edge_type, []).append(deepcopy(edge))
                for edge_type, rows in edges_by_type.items():
                    session.run(
                        f"""
                        UNWIND $rows AS row
                        MATCH (from:CompanyContext {{id: row.from_id, graph_ref: $graph_ref}})
                        MATCH (to:CompanyContext {{id: row.to_id, graph_ref: $graph_ref}})
                        CREATE (from)-[rel:{edge_type}]->(to)
                        SET rel = row
                        """,
                        rows=rows,
                        graph_ref=graph_ref,
                    )
            return {"status": "success", "error": None, "graph_ref": graph_ref}
        except Exception as exc:
            return {"status": "failed", "error": str(exc), "graph_ref": graph_ref}


def sync_company_context_pack_graph(
    company_context_pack: Any,
    profile: CompanyProfile,
    *,
    payload_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = (
        build_company_context_payload(payload_override, profile)
        if isinstance(payload_override, dict)
        else build_company_context_payload(company_context_pack, profile)
    )
    graph_payload = payload.get("company_context_graph") or {}
    sync_result = Neo4jCompanyContextGraphStore().sync_graph(graph_payload)
    sync_status = str(sync_result.get("status") or "failed")
    graph_ref = (
        payload.get("company_context_graph_ref")
        or graph_payload.get("graph_ref")
        or sync_result.get("graph_ref")
    )

    graph_cache = deepcopy(graph_payload)
    graph_cache["deep_research_handoff"] = (
        payload.get("deep_research_handoff")
        if isinstance(payload.get("deep_research_handoff"), dict)
        else {}
    )
    graph_cache["graph_derived_packet"] = (
        payload.get("graph_derived_packet")
        if isinstance(payload.get("graph_derived_packet"), dict)
        else graph_cache.get("graph_derived_packet") or {}
    )
    graph_cache["source_documents"] = payload.get("source_documents") or graph_payload.get("source_documents") or []

    company_context_pack.company_context_graph_ref = graph_ref
    company_context_pack.company_context_graph_cache_json = graph_cache
    company_context_pack.graph_stats_json = payload.get("graph_stats") or {}
    company_context_pack.graph_sync_status = sync_status
    company_context_pack.graph_sync_error = sync_result.get("error")
    company_context_pack.graph_synced_at = datetime.utcnow()
    company_context_pack.sourcing_brief_json = (
        payload.get("sourcing_brief")
        or company_context_pack.sourcing_brief_json
        or {}
    )
    payload["graph_status"] = sync_status
    payload["graph_warning"] = sync_result.get("error")
    payload["graph_synced_at"] = company_context_pack.graph_synced_at
    payload["company_context_graph_ref"] = graph_ref
    payload["source_documents"] = graph_cache.get("source_documents") or []
    return payload


def build_company_context_payload(
    company_context_pack: Any,
    profile: CompanyProfile,
) -> Dict[str, Any]:
    expansion_inputs = build_expansion_inputs(
        build_context_pack_v2(profile.context_pack_json or {}),
        comparator_seed_urls=profile.comparator_seed_urls or [],
        buyer_url=profile.buyer_company_url,
    )
    if hasattr(company_context_pack, "workspace_id"):
        sourcing_brief_override = (
            deepcopy(company_context_pack.sourcing_brief_json)
            if isinstance(getattr(company_context_pack, "sourcing_brief_json", None), dict)
            else {}
        )
        summary_override = str(sourcing_brief_override.get("source_summary") or "").strip() or None
        open_questions_override = sourcing_brief_override.get("open_questions") or None
        override_nodes = (
            deepcopy(company_context_pack.taxonomy_nodes_json)
            if isinstance(company_context_pack.taxonomy_nodes_json, list) and company_context_pack.taxonomy_nodes_json
            else None
        )
        base_payload = build_company_context_artifacts(
            profile,
            override_nodes=override_nodes,
            source_summary_override=summary_override,
            open_questions_override=open_questions_override,
            confirmed_at=company_context_pack.confirmed_at,
        )
    else:
        base_payload = deepcopy(company_context_pack or {})
    base_payload["expansion_inputs"] = expansion_inputs
    graph = build_company_context_graph(profile, payload=base_payload)
    brief = generate_sourcing_brief_from_graph(
        graph,
        base_brief=base_payload.get("sourcing_brief") or {},
    )
    graph["sourcing_brief"] = brief
    deep_research_handoff = build_deep_research_handoff(
        graph,
        brief=brief,
        expansion_inputs=base_payload.get("expansion_inputs") or [],
    )
    return {
        **base_payload,
        "company_context_graph_ref": graph.get("graph_ref"),
        "company_context_graph": graph,
        "deep_research_handoff": deep_research_handoff,
        "graph_status": "ready",
        "graph_freshness": graph.get("generated_at"),
        "graph_stats": graph.get("graph_stats") or {},
        "sourcing_brief": brief,
        "expansion_inputs": base_payload.get("expansion_inputs") or [],
    }
