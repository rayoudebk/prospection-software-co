"""Workspace API routes - Full CRUD and workflow endpoints."""
import asyncio
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import math

from app.models.base import async_session_maker, get_db
from app.models.workspace import Workspace, CompanyProfile
from app.models.company_context import CompanyContextPack
from app.models.company import Company, CompanyDossier, CompanyStatus
from app.models.job import (
    DB_ACTIVE_JOB_STATES,
    RUNTIME_ACTIVE_JOB_STATES,
    Job,
    JobType,
    JobState,
    JobProvider,
)
from app.models.source_evidence import SourceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, CompanyFact
from app.models.claims_graph import ClaimGraphNode, ClaimGraphEdge, ClaimGraphEdgeEvidence
from app.models.workspace_feedback import WorkspaceFeedbackEvent
from app.models.evaluation import EvaluationRun, EvaluationSampleResult
from app.models.intelligence import (
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    ComparatorSourceRun,
    RegistryQueryLog,
    CompanyMention,
    CompanyScreening,
    CompanyClaim,
)
from app.services.decision_catalog import get_catalog_payload, reason_text
from app.services.evidence_policy import (
    DEFAULT_EVIDENCE_POLICY,
    claim_group_for_dimension,
    is_fresh,
    normalize_policy,
)
from app.services.decision_engine import evaluate_decision
from app.services.reporting import (
    RELIABLE_FILINGS_COUNTRIES,
    classify_size_bucket,
    estimate_size_from_signals,
    extract_customers_and_integrations,
    normalize_domain,
    is_trusted_source_url,
    modules_with_evidence,
    normalize_country,
    source_label_for_url,
)
from app.services.quality_audit import normalize_quality_audit_v1
from app.services.company_context_graph import (
    Neo4jCompanyContextGraphStore,
    build_company_context_payload,
)
from app.services.company_context import (
    assess_buyer_evidence,
    apply_scope_review_decisions,
    build_expansion_report_artifact,
    build_sourcing_report_artifact,
    build_context_pack_v2,
    build_expansion_inputs,
    build_company_context_artifacts,
    derive_discovery_scope_hints,
    derive_scope_review_payload,
    normalize_expansion_brief,
    normalize_lens_seeds,
    normalize_open_questions,
    normalize_taxonomy_edges,
    normalize_taxonomy_nodes,
    _build_sourcing_brief_artifacts,
    _derive_source_pills_from_profile,
)

router = APIRouter()

DIRECTORY_HOST_TOKENS = (
    "thewealthmosaic.com",
    "thewealthmosaic.co.uk",
    "crunchbase.com",
    "g2.com",
    "capterra.com",
)


# ============================================================================
# Pydantic Schemas
# ============================================================================

class WorkspaceCreate(BaseModel):
    name: str
    region_scope: str = "EU+UK"


class WorkspaceUpdate(BaseModel):
    name: Optional[str] = None
    region_scope: Optional[str] = None


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    region_scope: str
    created_at: datetime
    company_count: int = 0
    has_context_pack: bool = False
    has_confirmed_scope_review: bool = False

    class Config:
        from_attributes = True


class GeoScope(BaseModel):
    region: str = "EU+UK"
    include_countries: List[str] = Field(default_factory=list)
    exclude_countries: List[str] = Field(default_factory=list)


class CompanyProfileUpdate(BaseModel):
    buyer_company_url: Optional[str] = None
    comparator_seed_urls: Optional[List[str]] = None
    supporting_evidence_urls: Optional[List[str]] = None
    geo_scope: Optional[GeoScope] = None


class CompanyProfileResponse(BaseModel):
    id: int
    workspace_id: int
    buyer_company_url: Optional[str]
    comparator_seed_urls: List[str]
    supporting_evidence_urls: List[str]
    comparator_seed_summaries: Dict[str, str]
    geo_scope: Dict[str, Any]
    context_pack_markdown: Optional[str]
    context_pack_generated_at: Optional[datetime]
    product_pages_found: int
    context_pack_json: Optional[Dict[str, Any]] = None  # Full structured data

    class Config:
        from_attributes = True


class BuyerEvidenceDiagnosticsResponse(BaseModel):
    mode: str
    status: str
    score: int
    used_for_inference: bool
    warning: Optional[str] = None
    metrics: Dict[str, int] = Field(default_factory=dict)


class TaxonomyNodeResponse(BaseModel):
    id: str
    layer: str
    phrase: str
    aliases: List[str] = Field(default_factory=list)
    confidence: float
    evidence_ids: List[str] = Field(default_factory=list)
    scope_status: str = "in_scope"


class TaxonomyNodeInput(BaseModel):
    id: Optional[str] = None
    layer: str
    phrase: str
    aliases: List[str] = Field(default_factory=list)
    confidence: float = 0.68
    evidence_ids: List[str] = Field(default_factory=list)
    scope_status: str = "in_scope"


class TaxonomyEdgeResponse(BaseModel):
    from_node_id: str
    to_node_id: str
    relation_type: str
    evidence_ids: List[str] = Field(default_factory=list)


class LensSeedResponse(BaseModel):
    id: str
    lens_type: str
    label: str
    query_phrase: Optional[str] = None
    rationale: str
    supporting_node_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float


class SourcingBriefResponse(BaseModel):
    source_company: Dict[str, Any] = Field(default_factory=dict)
    source_summary: Optional[str] = None
    reasoning_status: str = "not_run"
    reasoning_warning: Optional[str] = None
    reasoning_provider: Optional[str] = None
    reasoning_model: Optional[str] = None
    customer_nodes: List[TaxonomyNodeResponse] = Field(default_factory=list)
    workflow_nodes: List[TaxonomyNodeResponse] = Field(default_factory=list)
    capability_nodes: List[TaxonomyNodeResponse] = Field(default_factory=list)
    delivery_or_integration_nodes: List[TaxonomyNodeResponse] = Field(default_factory=list)
    named_customer_proof: List[Dict[str, Any]] = Field(default_factory=list)
    partner_integration_proof: List[Dict[str, Any]] = Field(default_factory=list)
    secondary_evidence_proof: List[Dict[str, Any]] = Field(default_factory=list)
    customer_partner_corroboration: List[Dict[str, Any]] = Field(default_factory=list)
    directory_category_context: List[Dict[str, Any]] = Field(default_factory=list)
    other_secondary_context: List[Dict[str, Any]] = Field(default_factory=list)
    active_lenses: List[LensSeedResponse] = Field(default_factory=list)
    adjacency_hypotheses: List[Dict[str, Any]] = Field(default_factory=list)
    strongest_evidence_buckets: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_gaps: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    unknowns_not_publicly_resolvable: List[str] = Field(default_factory=list)
    crawl_coverage: Dict[str, Any] = Field(default_factory=dict)
    confirmed_at: Optional[Any] = None


class ExpansionBriefItemResponse(BaseModel):
    id: str
    label: str
    expansion_type: str
    status: str
    confidence: float
    why_it_matters: Optional[str] = None
    evidence_urls: List[str] = Field(default_factory=list)
    supporting_node_ids: List[str] = Field(default_factory=list)
    source_entity_names: List[str] = Field(default_factory=list)
    market_importance: str = "medium"
    operational_centrality: str = "meaningful"
    workflow_criticality: str = "medium"
    daily_operator_usage: str = "medium"
    switching_cost_intensity: str = "medium"
    priority_tier: str = "meaningful_adjacent"


class ExpansionBriefResponse(BaseModel):
    reasoning_status: str = "not_run"
    reasoning_warning: Optional[str] = None
    reasoning_provider: Optional[str] = None
    reasoning_model: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    adjacent_capabilities: List[ExpansionBriefItemResponse] = Field(default_factory=list)
    adjacent_customer_segments: List[ExpansionBriefItemResponse] = Field(default_factory=list)
    named_account_anchors: List[ExpansionBriefItemResponse] = Field(default_factory=list)
    geography_expansions: List[ExpansionBriefItemResponse] = Field(default_factory=list)


class ScopeReviewItemResponse(BaseModel):
    id: str
    label: str
    scope_item_type: str
    origin: str
    status: str
    confidence: float
    evidence_ids: List[str] = Field(default_factory=list)
    evidence_urls: List[str] = Field(default_factory=list)
    supporting_node_ids: List[str] = Field(default_factory=list)
    source_entity_names: List[str] = Field(default_factory=list)
    why_it_matters: Optional[str] = None
    priority_tier: Optional[str] = None
    market_importance: Optional[str] = None
    operational_centrality: Optional[str] = None
    workflow_criticality: Optional[str] = None
    daily_operator_usage: Optional[str] = None
    switching_cost_intensity: Optional[str] = None


class ScopeReviewDecisionInput(BaseModel):
    id: str
    status: str


class ScopeReviewUpdate(BaseModel):
    decisions: List[ScopeReviewDecisionInput] = Field(default_factory=list)


class SourceDocumentResponse(BaseModel):
    id: str
    name: str
    url: Optional[str] = None
    publisher: Optional[str] = None
    snippet: Optional[str] = None
    publisher_channel: str
    publisher_type: Optional[str] = None
    claim_scope: Optional[str] = None
    subject_company: Optional[str] = None
    evidence_tier: str
    evidence_type: str


class ReportArtifactSourcePillResponse(BaseModel):
    id: str
    label: str
    url: str
    publisher: Optional[str] = None
    publisher_channel: str
    publisher_type: Optional[str] = None
    source_tier: str
    source_kind: str
    evidence_type: str
    claim_scope: Optional[str] = None
    published_at: Optional[str] = None
    captured_at: Optional[str] = None


class ReportArtifactSentenceResponse(BaseModel):
    id: str
    text: str
    citation_pill_ids: List[str] = Field(default_factory=list)


class ReportArtifactParagraphBlockResponse(BaseModel):
    type: str = "paragraph"
    sentences: List[ReportArtifactSentenceResponse] = Field(default_factory=list)


class ReportArtifactBulletListBlockResponse(BaseModel):
    type: str = "bullet_list"
    items: List[ReportArtifactSentenceResponse] = Field(default_factory=list)


class ReportArtifactCalloutBlockResponse(BaseModel):
    type: str = "callout"
    tone: str = "neutral"
    title: Optional[str] = None
    sentences: List[ReportArtifactSentenceResponse] = Field(default_factory=list)


class ReportArtifactKeyValueItemResponse(BaseModel):
    id: str
    key: str
    value: str
    citation_pill_ids: List[str] = Field(default_factory=list)


class ReportArtifactKeyValueBlockResponse(BaseModel):
    type: str = "key_value"
    items: List[ReportArtifactKeyValueItemResponse] = Field(default_factory=list)


class ReportArtifactSectionResponse(BaseModel):
    id: str
    heading: Optional[str] = None
    blocks: List[Dict[str, Any]] = Field(default_factory=list)


class ReportArtifactResponse(BaseModel):
    artifact_type: str = "report_artifact"
    report_kind: str
    version: str = "v1"
    status: str
    generated_at: Optional[str] = None
    confirmed_at: Optional[str] = None
    reasoning_status: str = "not_run"
    reasoning_warning: Optional[str] = None
    reasoning_provider: Optional[str] = None
    reasoning_model: Optional[str] = None
    title: str
    summary: Optional[str] = None
    sections: List[ReportArtifactSectionResponse] = Field(default_factory=list)
    sources: List[ReportArtifactSourcePillResponse] = Field(default_factory=list)
    footer_actions: List[str] = Field(default_factory=list)


class CompanyContextPackResponse(BaseModel):
    id: int
    workspace_id: int
    company_context_graph_ref: Optional[str] = None
    graph_status: str = "not_synced"
    graph_warning: Optional[str] = None
    graph_synced_at: Optional[datetime] = None
    graph_stats: Dict[str, Any] = Field(default_factory=dict)
    company_context_graph: Optional[Dict[str, Any]] = None
    deep_research_handoff: Dict[str, Any] = Field(default_factory=dict)
    buyer_evidence: Optional[BuyerEvidenceDiagnosticsResponse] = None
    context_pack_v2: Optional[Dict[str, Any]] = None
    source_documents: List[SourceDocumentResponse] = Field(default_factory=list)
    expansion_inputs: List[Dict[str, Any]] = Field(default_factory=list)
    taxonomy_nodes: List[TaxonomyNodeResponse] = Field(default_factory=list)
    taxonomy_edges: List[TaxonomyEdgeResponse] = Field(default_factory=list)
    lens_seeds: List[LensSeedResponse] = Field(default_factory=list)
    sourcing_brief: Optional[SourcingBriefResponse] = None
    expansion_brief: Optional[ExpansionBriefResponse] = None
    sourcing_report: Optional[ReportArtifactResponse] = None
    expansion_report: Optional[ReportArtifactResponse] = None
    generated_at: Optional[datetime]
    confirmed_at: Optional[datetime]


class CompanyContextPackUpdate(BaseModel):
    source_summary: Optional[str] = None
    taxonomy_nodes: Optional[List[TaxonomyNodeInput]] = None
    confirmed: Optional[bool] = None


class ScopeReviewResponse(BaseModel):
    workspace_id: int
    workspace_geo_scope: Dict[str, Any] = Field(default_factory=dict)
    confirmed_at: Optional[datetime] = None
    source_capabilities: List[ScopeReviewItemResponse] = Field(default_factory=list)
    source_customer_segments: List[ScopeReviewItemResponse] = Field(default_factory=list)
    source_workflows: List[ScopeReviewItemResponse] = Field(default_factory=list)
    source_delivery_or_integration: List[ScopeReviewItemResponse] = Field(default_factory=list)
    adjacent_capabilities: List[ScopeReviewItemResponse] = Field(default_factory=list)
    adjacent_customer_segments: List[ScopeReviewItemResponse] = Field(default_factory=list)
    named_account_anchors: List[ScopeReviewItemResponse] = Field(default_factory=list)
    geography_expansions: List[ScopeReviewItemResponse] = Field(default_factory=list)


class CompanyCreate(BaseModel):
    name: str
    website: Optional[str] = None
    hq_country: Optional[str] = None


class CompanyUpdate(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    hq_country: Optional[str] = None
    operating_countries: Optional[List[str]] = None
    tags_custom: Optional[List[str]] = None
    status: Optional[str] = None


class CompanyResponse(BaseModel):
    id: int
    workspace_id: int
    name: str
    website: Optional[str]
    official_website_url: Optional[str] = None
    discovery_url: Optional[str] = None
    entity_type: Optional[str] = None
    hq_country: Optional[str]
    operating_countries: List[str]
    tags_custom: List[str]
    status: str
    why_relevant: List[Dict[str, Any]]
    is_manual: bool
    created_at: datetime
    evidence_count: int = 0
    decision_classification: Optional[str] = None
    evidence_sufficiency: Optional[str] = None
    reason_codes: Dict[str, List[str]] = Field(default_factory=lambda: {"positive": [], "caution": [], "reject": []})
    rationale_summary: Optional[str] = None
    top_claim: Dict[str, Any] = Field(default_factory=dict)
    citation_summary_v1: Optional["CitationSummaryV1"] = None
    registry_neighbors_with_first_party_website_count: int = 0
    registry_neighbors_dropped_missing_official_website_count: int = 0
    registry_origin_screening_counts: Dict[str, int] = Field(default_factory=dict)
    first_party_hint_urls_used_count: int = 0
    first_party_hint_pages_crawled_total: int = 0
    unresolved_contradictions_count: int = 0
    why_fit_bullets: List[Dict[str, Any]] = Field(default_factory=list)
    business_model_signal: Optional[str] = None
    customer_proof: List[str] = Field(default_factory=list)
    employee_signal: Optional[str] = None
    open_questions: List[str] = Field(default_factory=list)

    class Config:
        from_attributes = True


class CompanyDossierResponse(BaseModel):
    id: int
    company_id: int
    dossier_json: Dict[str, Any]
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


class EnrichBatchRequest(BaseModel):
    company_ids: List[int]
    job_types: List[str] = Field(default=["enrich_full"])


class JobResponse(BaseModel):
    id: int
    workspace_id: int
    company_id: Optional[int]
    job_type: str
    state: str
    provider: str
    progress: float
    progress_message: Optional[str]
    result_json: Optional[Dict[str, Any]]
    error_message: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]

    class Config:
        from_attributes = True


class GatesResponse(BaseModel):
    context_pack: bool
    scope_review: bool
    universe: bool
    segmentation: bool
    enrichment: bool
    missing_items: Dict[str, List[str]]


class ReportGenerateRequest(BaseModel):
    name: Optional[str] = None
    include_unknown_size: bool = False
    include_outside_sme: bool = False


class SourcePill(BaseModel):
    label: str
    url: str
    document_id: Optional[str] = None
    captured_at: Optional[str] = None


class CitationSentence(BaseModel):
    id: str
    text: str
    citation_pill_ids: List[str] = Field(default_factory=list)


class CitationSourcePill(BaseModel):
    pill_id: str
    label: str
    url: str
    source_tier: str
    source_kind: str
    captured_at: Optional[str] = None
    claim_group: str


class CitationSummaryV1(BaseModel):
    version: str
    sentences: List[CitationSentence] = Field(default_factory=list)
    source_pills: List[CitationSourcePill] = Field(default_factory=list)


class SourcedValue(BaseModel):
    value: str
    unit: Optional[str] = None
    period: Optional[str] = None
    confidence: str
    source: SourcePill


class ReportClaim(BaseModel):
    text: str
    confidence: str
    rendering: str = "fact"  # fact | hypothesis
    source: Optional[SourcePill] = None


class ReportCard(BaseModel):
    company_id: int
    name: str
    website: Optional[str] = None
    hq_country: Optional[str] = None
    legal_status: Optional[str] = None
    size_bucket: str
    size_estimate: Optional[int] = None
    size_range_low: Optional[int] = None
    size_range_high: Optional[int] = None
    fit_score: float
    evidence_score: float
    workflow_profile: List[ReportClaim] = Field(default_factory=list)
    customer_profile: List[ReportClaim] = Field(default_factory=list)
    business_model_profile: List[ReportClaim] = Field(default_factory=list)
    ownership_profile: List[ReportClaim] = Field(default_factory=list)
    transaction_profile: List[ReportClaim] = Field(default_factory=list)
    filing_metrics: Dict[str, SourcedValue] = Field(default_factory=dict)
    source_pills: List[SourcePill] = Field(default_factory=list)
    coverage_note: Optional[str] = None
    next_validation_questions: List[str] = Field(default_factory=list)
    decision_classification: Optional[str] = None
    reason_highlights: List[str] = Field(default_factory=list)
    evidence_quality_summary: Dict[str, Any] = Field(default_factory=dict)
    known_unknowns: List[str] = Field(default_factory=list)


class UniverseTopCandidateResponse(BaseModel):
    company_id: Optional[int] = None
    candidate_entity_id: Optional[int] = None
    company_name: str
    official_website_url: Optional[str] = None
    discovery_sources: List[str] = Field(default_factory=list)
    entity_type: str = "company"
    decision_classification: str
    evidence_sufficiency: str
    reason_codes: Dict[str, List[str]] = Field(default_factory=lambda: {"positive": [], "caution": [], "reject": []})
    rationale_summary: Optional[str] = None
    top_claim: Dict[str, Any] = Field(default_factory=dict)
    citation_summary_v1: Optional[CitationSummaryV1] = None
    registry_neighbors_with_first_party_website_count: int = 0
    registry_neighbors_dropped_missing_official_website_count: int = 0
    registry_origin_screening_counts: Dict[str, int] = Field(default_factory=dict)
    first_party_hint_urls_used_count: int = 0
    first_party_hint_pages_crawled_total: int = 0
    missing_claim_groups: List[str] = Field(default_factory=list)
    unresolved_contradictions_count: int = 0
    ranking_eligible: bool = False
    run_quality_tier: str = "degraded"
    quality_gate_passed: bool = False
    quality_audit_passed: bool = False
    degraded_reasons: List[str] = Field(default_factory=list)


class QualityAuditPattern(BaseModel):
    pattern_key: str
    count: int
    sample_screening_ids: List[int] = Field(default_factory=list)
    sample_candidate_names: List[str] = Field(default_factory=list)


class QualityAuditImpactedCandidate(BaseModel):
    screening_id: int
    candidate_name: str
    reasons: List[str] = Field(default_factory=list)


class QualityAuditV1(BaseModel):
    run_id: str
    pass_: bool = Field(alias="pass")
    patterns: List[QualityAuditPattern] = Field(default_factory=list)
    thresholds: Dict[str, int] = Field(default_factory=dict)
    top_impacted_candidates: List[QualityAuditImpactedCandidate] = Field(default_factory=list)
    generated_at: Optional[str] = None


class ReportSnapshotResponse(BaseModel):
    id: int
    workspace_id: int
    name: str
    status: str
    generated_at: datetime
    filters_json: Dict[str, Any]
    coverage_json: Dict[str, Any]
    item_count: int = 0

    class Config:
        from_attributes = True


class CompanyDecisionResponse(BaseModel):
    company_id: int
    workspace_id: int
    classification: str
    evidence_sufficiency: str
    positive_reason_codes: List[str] = Field(default_factory=list)
    caution_reason_codes: List[str] = Field(default_factory=list)
    reject_reason_codes: List[str] = Field(default_factory=list)
    missing_claim_groups: List[str] = Field(default_factory=list)
    unresolved_contradictions_count: int = 0
    rationale_summary: Optional[str] = None
    rationale_markdown: Optional[str] = None
    decision_engine_version: Optional[str] = None
    gating_passed: bool = False
    generated_at: str


class EvidencePolicyUpdate(BaseModel):
    policy: Dict[str, Any]


class MonitoringRunRequest(BaseModel):
    max_companies: int = 80
    stale_only: bool = False
    classifications: List[str] = Field(
        default_factory=lambda: ["borderline_watchlist", "insufficient_evidence"]
    )


class WorkspaceFeedbackCreate(BaseModel):
    company_id: Optional[int] = None
    company_screening_id: Optional[int] = None
    feedback_type: str = "classification_override"
    previous_classification: Optional[str] = None
    new_classification: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    comment: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str] = None


class WorkspaceFeedbackResponse(BaseModel):
    id: int
    workspace_id: int
    company_id: Optional[int]
    company_screening_id: Optional[int]
    feedback_type: str
    previous_classification: Optional[str]
    new_classification: Optional[str]
    reason_codes: List[str] = Field(default_factory=list)
    comment: Optional[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: Optional[str]
    created_at: datetime


class ClaimsGraphSummaryResponse(BaseModel):
    workspace_id: int
    nodes_count: int
    edges_count: int
    edge_evidence_count: int
    relation_type_distribution: Dict[str, int] = Field(default_factory=dict)
    node_type_distribution: Dict[str, int] = Field(default_factory=dict)
    generated_at: str


class EvaluationReplayRequest(BaseModel):
    model_version: str = "decision_engine_v1"
    samples: List[Dict[str, Any]] = Field(default_factory=list)


class EvaluationReplayResponse(BaseModel):
    workspace_id: int
    run_id: int
    metrics: Dict[str, Any]
    created_at: str


# ============================================================================
# Internal Helpers
# ============================================================================

def _to_job_response(job: Job) -> JobResponse:
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        company_id=job.company_id,
        job_type=job.job_type.value,
        state=job.state.value,
        provider=job.provider.value,
        progress=job.progress,
        progress_message=job.progress_message,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


def _safe_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        from urllib.parse import urlparse

        normalized = url if url.startswith(("http://", "https://")) else f"https://{url}"
        parsed = urlparse(normalized)
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None


def _normalize_http_url(url: Optional[str]) -> Optional[str]:
    raw = str(url or "").strip()
    if not raw:
        return None
    try:
        from urllib.parse import urlparse

        has_scheme = "://" in raw
        normalized = raw if has_scheme else f"https://{raw}"
        parsed = urlparse(normalized)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"}:
            return None
        if not parsed.netloc:
            return None
        if parsed.username or parsed.password:
            return None
        query = f"?{parsed.query}" if parsed.query else ""
        path = parsed.path or ""
        return f"{scheme}://{parsed.netloc}{path}{query}"
    except Exception:
        return None


def _clean_url_list(
    values: Optional[List[str]],
    *,
    max_items: int = 50,
    require_path: bool = False,
) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for value in values or []:
        normalized = _normalize_http_url(value)
        if not normalized:
            continue
        if require_path:
            try:
                from urllib.parse import urlparse

                parsed = urlparse(normalized)
                path = str(parsed.path or "").strip()
                if not path or path == "/":
                    continue
            except Exception:
                continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _citation_summary_from_meta(screening_meta: Dict[str, Any]) -> Optional[CitationSummaryV1]:
    payload = screening_meta.get("citation_summary_v1")
    if not isinstance(payload, dict):
        return None
    try:
        summary = CitationSummaryV1.model_validate(payload)
    except Exception:
        return None
    if summary.version != "v1":
        return None
    if not summary.sentences or not summary.source_pills:
        return None
    return summary


def _screening_diagnostics_from_meta(screening_meta: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "registry_neighbors_with_first_party_website_count": int(
            screening_meta.get("registry_neighbors_with_first_party_website_count", 0)
        ),
        "registry_neighbors_dropped_missing_official_website_count": int(
            screening_meta.get("registry_neighbors_dropped_missing_official_website_count", 0)
        ),
        "registry_origin_screening_counts": (
            screening_meta.get("registry_origin_screening_counts")
            if isinstance(screening_meta.get("registry_origin_screening_counts"), dict)
            else {}
        ),
        "first_party_hint_urls_used_count": int(screening_meta.get("first_party_hint_urls_used_count", 0)),
        "first_party_hint_pages_crawled_total": int(screening_meta.get("first_party_hint_pages_crawled_total", 0)),
    }


def _quality_audit_from_job_result(result_json: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    payload = result_json if isinstance(result_json, dict) else {}
    normalized = normalize_quality_audit_v1(payload.get("quality_audit_v1"))
    if not normalized:
        return None
    try:
        model = QualityAuditV1.model_validate(normalized)
    except Exception:
        return None
    return model.model_dump(by_alias=True)


def _quality_payload_from_job_result(result_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = result_json if isinstance(result_json, dict) else {}
    screening_run_id = str(payload.get("screening_run_id") or "").strip() or None
    run_quality_tier = str(payload.get("run_quality_tier") or "").strip().lower()
    if run_quality_tier not in {"high_quality", "degraded"}:
        fallback_mode = bool(payload.get("fallback_mode"))
        run_quality_tier = "degraded" if fallback_mode else "high_quality"
    degraded_reasons = [
        str(item).strip()
        for item in (payload.get("degraded_reasons") or [])
        if str(item).strip()
    ]
    quality_gate_passed = bool(payload.get("quality_gate_passed"))
    if "quality_gate_passed" not in payload:
        quality_gate_passed = run_quality_tier == "high_quality"
    quality_audit_v1 = _quality_audit_from_job_result(payload)
    if quality_audit_v1 and screening_run_id and str(quality_audit_v1.get("run_id") or "").strip() != screening_run_id:
        quality_audit_v1 = None
    quality_audit_passed = bool(payload.get("quality_audit_passed"))
    if quality_audit_v1 is not None:
        quality_audit_passed = bool(quality_audit_v1.get("pass"))
    quality_validation_ready = bool(payload.get("quality_validation_ready"))
    if "quality_validation_ready" not in payload and quality_audit_v1 is not None:
        quality_validation_ready = bool(quality_audit_v1.get("pass"))
    return {
        "run_quality_tier": run_quality_tier,
        "quality_gate_passed": quality_gate_passed,
        "quality_audit_v1": quality_audit_v1,
        "quality_audit_passed": quality_audit_passed,
        "quality_validation_ready": quality_validation_ready,
        "quality_validation_blocked_reasons": [
            str(item).strip()
            for item in (payload.get("quality_validation_blocked_reasons") or [])
            if str(item).strip()
        ],
        "pre_rerun_quality_audit_v1": normalize_quality_audit_v1(payload.get("pre_rerun_quality_audit_v1")),
        "pre_rerun_quality_audit_run_id": str(payload.get("pre_rerun_quality_audit_run_id") or "").strip() or None,
        "pre_rerun_quality_validation_ready": bool(payload.get("pre_rerun_quality_validation_ready", False)),
        "pre_rerun_quality_validation_blocked_reasons": [
            str(item).strip()
            for item in (payload.get("pre_rerun_quality_validation_blocked_reasons") or [])
            if str(item).strip()
        ],
        "degraded_reasons": degraded_reasons,
        "model_attempt_trace": payload.get("model_attempt_trace") if isinstance(payload.get("model_attempt_trace"), list) else [],
        "stage_time_ms": payload.get("stage_time_ms") if isinstance(payload.get("stage_time_ms"), dict) else {},
        "timeout_events": payload.get("timeout_events") if isinstance(payload.get("timeout_events"), list) else [],
        "queue_wait_ms_by_stage": (
            payload.get("queue_wait_ms_by_stage")
            if isinstance(payload.get("queue_wait_ms_by_stage"), dict)
            else {}
        ),
        "stage_retry_counts": (
            payload.get("stage_retry_counts")
            if isinstance(payload.get("stage_retry_counts"), dict)
            else {}
        ),
        "cache_hit_rates": (
            payload.get("cache_hit_rates")
            if isinstance(payload.get("cache_hit_rates"), dict)
            else {}
        ),
        "candidate_dropoff_funnel_v1": (
            payload.get("candidate_dropoff_funnel_v1")
            if isinstance(payload.get("candidate_dropoff_funnel_v1"), dict)
            else {}
        ),
        "stage_execution_mode": str(payload.get("stage_execution_mode") or "hybrid_preflight_monolith"),
        "screening_run_id": screening_run_id,
    }


def _safe_metric_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _compute_variance_hotspots_from_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metric_extractors = {
        "first_party_hint_urls_used_count": lambda row: row.get("first_party_hint_urls_used_count"),
        "first_party_hint_pages_crawled_total": lambda row: row.get("first_party_hint_pages_crawled_total"),
        "first_party_crawl_pages_total": lambda row: row.get("first_party_crawl_pages_total"),
        "stage_llm_discovery_fanout.llm_ms": lambda row: (
            (row.get("stage_time_ms") or {}).get("stage_llm_discovery_fanout")
            if isinstance(row.get("stage_time_ms"), dict)
            else None
        ),
        "ranking_eligible_count": lambda row: row.get("ranking_eligible_count"),
    }

    hotspots: List[Dict[str, Any]] = []
    for metric_name, extractor in metric_extractors.items():
        values: List[float] = []
        for row in results:
            if not isinstance(row, dict):
                continue
            value = _safe_metric_number(extractor(row))
            if value is None:
                continue
            values.append(float(value))
        if not values:
            continue
        run_count = len(values)
        mean = sum(values) / run_count
        variance = sum((value - mean) ** 2 for value in values) / max(1, run_count)
        hotspots.append(
            {
                "metric": metric_name,
                "min": min(values),
                "max": max(values),
                "avg": round(mean, 4),
                "stddev": round(math.sqrt(variance), 4),
                "run_count": run_count,
            }
        )

    hotspots.sort(key=lambda row: float(row.get("stddev") or 0.0), reverse=True)
    return hotspots


async def _variance_hotspots_for_workspace(
    db: AsyncSession,
    workspace_id: int,
    limit_runs: int = 25,
) -> List[Dict[str, Any]]:
    result = await db.execute(
        select(Job)
        .where(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.discovery_universe,
            Job.state == JobState.completed,
        )
        .order_by(Job.finished_at.desc(), Job.created_at.desc())
        .limit(max(3, int(limit_runs)))
    )
    rows = result.scalars().all()
    payloads = [
        row.result_json
        for row in rows
        if isinstance(row.result_json, dict)
    ]
    return _compute_variance_hotspots_from_results(payloads)


async def _latest_completed_discovery_job(
    db: AsyncSession,
    workspace_id: int,
) -> Optional[Job]:
    result = await db.execute(
        select(Job)
        .where(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.discovery_universe,
            Job.state == JobState.completed,
        )
        .order_by(Job.finished_at.desc(), Job.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _company_context_response_from_payload(payload: Dict[str, Any]) -> CompanyContextPackResponse:
    sourcing_brief = payload.get("sourcing_brief")
    if isinstance(sourcing_brief, dict):
        sourcing_brief = {
            **sourcing_brief,
            "partner_integration_proof": sourcing_brief.get("partner_integration_proof") or [],
            "secondary_evidence_proof": sourcing_brief.get("secondary_evidence_proof") or [],
            "customer_partner_corroboration": sourcing_brief.get("customer_partner_corroboration") or [],
            "directory_category_context": sourcing_brief.get("directory_category_context") or [],
            "other_secondary_context": sourcing_brief.get("other_secondary_context") or [],
        }
    expansion_brief = normalize_expansion_brief(payload.get("expansion_brief") or {})
    source_documents = []
    for item in (payload.get("source_documents") or []):
        if not isinstance(item, dict):
            continue
        source_documents.append(
            {
                **item,
                "publisher_channel": item.get("publisher_channel") or item.get("source_type") or "company_website",
                "publisher_type": item.get("publisher_type"),
                "claim_scope": item.get("claim_scope"),
                "subject_company": item.get("subject_company"),
            }
        )
    sourcing_report = payload.get("sourcing_report")
    if not isinstance(sourcing_report, dict):
        sourcing_report = build_sourcing_report_artifact(
            sourcing_brief=sourcing_brief if isinstance(sourcing_brief, dict) else {},
            source_documents=source_documents,
            context_pack_v2=payload.get("context_pack_v2") if isinstance(payload.get("context_pack_v2"), dict) else {},
            confirmed_at=payload.get("confirmed_at"),
        )
    expansion_report = payload.get("expansion_report")
    if not isinstance(expansion_report, dict):
        expansion_report = build_expansion_report_artifact(
            source_company=(sourcing_brief.get("source_company") if isinstance(sourcing_brief, dict) else {}) or {},
            expansion_brief=expansion_brief,
            source_documents=source_documents,
            context_pack_v2=payload.get("context_pack_v2") if isinstance(payload.get("context_pack_v2"), dict) else {},
            confirmed_at=(
                expansion_brief.get("confirmed_at")
                if isinstance(expansion_brief, dict)
                else payload.get("confirmed_at")
            ),
        )
    return CompanyContextPackResponse(
        id=int(payload.get("id") or 0),
        workspace_id=int(payload.get("workspace_id") or 0),
        company_context_graph_ref=payload.get("company_context_graph_ref"),
        graph_status=str(payload.get("graph_status") or "not_synced"),
        graph_warning=payload.get("graph_warning"),
        graph_synced_at=payload.get("graph_synced_at"),
        graph_stats=payload.get("graph_stats") if isinstance(payload.get("graph_stats"), dict) else {},
        company_context_graph=payload.get("company_context_graph") if isinstance(payload.get("company_context_graph"), dict) else None,
        deep_research_handoff=(
            payload.get("deep_research_handoff")
            if isinstance(payload.get("deep_research_handoff"), dict)
            else (
                payload.get("graph_derived_packet")
                if isinstance(payload.get("graph_derived_packet"), dict)
                else {}
            )
        ),
        source_documents=[SourceDocumentResponse.model_validate(item) for item in source_documents],
        buyer_evidence=(
            BuyerEvidenceDiagnosticsResponse.model_validate(payload.get("buyer_evidence"))
            if isinstance(payload.get("buyer_evidence"), dict)
            else None
        ),
        context_pack_v2=payload.get("context_pack_v2") if isinstance(payload.get("context_pack_v2"), dict) else None,
        expansion_inputs=[
            item
            for item in (payload.get("expansion_inputs") or [])
            if isinstance(item, dict)
        ],
        taxonomy_nodes=[
            TaxonomyNodeResponse.model_validate(item)
            for item in (payload.get("taxonomy_nodes") or [])
            if isinstance(item, dict)
        ],
        taxonomy_edges=[
            TaxonomyEdgeResponse.model_validate(item)
            for item in (payload.get("taxonomy_edges") or [])
            if isinstance(item, dict)
        ],
        lens_seeds=[
            LensSeedResponse.model_validate(item)
            for item in (payload.get("lens_seeds") or [])
            if isinstance(item, dict)
        ],
        sourcing_brief=(
            SourcingBriefResponse.model_validate(sourcing_brief)
            if isinstance(sourcing_brief, dict)
            else None
        ),
        expansion_brief=ExpansionBriefResponse.model_validate(expansion_brief),
        sourcing_report=ReportArtifactResponse.model_validate(sourcing_report),
        expansion_report=ReportArtifactResponse.model_validate(expansion_report),
        generated_at=payload.get("generated_at"),
        confirmed_at=payload.get("confirmed_at"),
    )


def _company_context_refresh_response_from_payload(payload: Dict[str, Any]) -> CompanyContextPackResponse:
    return CompanyContextPackResponse(
        id=int(payload.get("id") or 0),
        workspace_id=int(payload.get("workspace_id") or 0),
        company_context_graph_ref=payload.get("company_context_graph_ref"),
        graph_status=str(payload.get("graph_status") or "not_synced"),
        graph_warning=payload.get("graph_warning"),
        graph_synced_at=payload.get("graph_synced_at"),
        graph_stats=payload.get("graph_stats") if isinstance(payload.get("graph_stats"), dict) else {},
        company_context_graph=None,
        deep_research_handoff={},
        buyer_evidence=(
            BuyerEvidenceDiagnosticsResponse.model_validate(payload.get("buyer_evidence"))
            if isinstance(payload.get("buyer_evidence"), dict)
            else None
        ),
        context_pack_v2=None,
        source_documents=[],
        expansion_inputs=[],
        taxonomy_nodes=[],
        taxonomy_edges=[],
        lens_seeds=[],
        sourcing_brief=None,
        expansion_brief=None,
        sourcing_report=None,
        expansion_report=None,
        generated_at=payload.get("generated_at"),
        confirmed_at=payload.get("confirmed_at"),
    )


def _scope_review_response_from_payload(
    workspace_id: int,
    payload: Dict[str, Any],
) -> ScopeReviewResponse:
    defaults_by_key = {
        "adjacent_capabilities": {"scope_item_type": "adjacent_capability", "origin": "expansion_brief"},
        "adjacent_customer_segments": {"scope_item_type": "adjacent_customer_segment", "origin": "expansion_brief"},
        "named_account_anchors": {"scope_item_type": "named_account_anchor", "origin": "expansion_brief"},
        "geography_expansions": {"scope_item_type": "geography_expansion", "origin": "expansion_brief"},
    }

    def _items(key: str) -> List[ScopeReviewItemResponse]:
        return [
            ScopeReviewItemResponse.model_validate({**defaults_by_key.get(key, {}), **item})
            for item in (payload.get(key) or [])
            if isinstance(item, dict)
        ]

    return ScopeReviewResponse(
        workspace_id=workspace_id,
        workspace_geo_scope=payload.get("workspace_geo_scope") or {},
        confirmed_at=payload.get("confirmed_at"),
        source_capabilities=_items("source_capabilities"),
        source_customer_segments=_items("source_customer_segments"),
        source_workflows=_items("source_workflows"),
        source_delivery_or_integration=_items("source_delivery_or_integration"),
        adjacent_capabilities=_items("adjacent_capabilities"),
        adjacent_customer_segments=_items("adjacent_customer_segments"),
        named_account_anchors=_items("named_account_anchors"),
        geography_expansions=_items("geography_expansions"),
    )


def _company_context_payload_from_pack(
    company_context_pack: CompanyContextPack,
    *,
    profile: CompanyProfile,
) -> Dict[str, Any]:
    context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    graph_cache = (
        company_context_pack.company_context_graph_cache_json
        if isinstance(company_context_pack.company_context_graph_cache_json, dict)
        else {}
    )
    graph_ref = (
        company_context_pack.company_context_graph_ref
        or graph_cache.get("graph_ref")
    )
    payload = {
        "id": company_context_pack.id,
        "workspace_id": company_context_pack.workspace_id,
        "buyer_evidence": assess_buyer_evidence(profile),
        "context_pack_v2": context_pack_v2,
        "taxonomy_nodes": normalize_taxonomy_nodes(company_context_pack.taxonomy_nodes_json or []),
        "taxonomy_edges": normalize_taxonomy_edges(company_context_pack.taxonomy_edges_json or []),
        "lens_seeds": normalize_lens_seeds(company_context_pack.lens_seeds_json or []),
        "sourcing_brief": company_context_pack.sourcing_brief_json or {},
        "expansion_brief": normalize_expansion_brief(company_context_pack.expansion_brief_json or {}),
        "generated_at": company_context_pack.generated_at,
        "confirmed_at": company_context_pack.confirmed_at,
    }
    payload["company_context_graph_ref"] = graph_ref
    payload["company_context_graph"] = graph_cache or None
    payload["graph_status"] = company_context_pack.graph_sync_status or "not_synced"
    payload["graph_warning"] = company_context_pack.graph_sync_error
    payload["graph_synced_at"] = company_context_pack.graph_synced_at
    payload["graph_stats"] = company_context_pack.graph_stats_json or {}
    payload["deep_research_handoff"] = (
        graph_cache.get("deep_research_handoff")
        if isinstance(graph_cache.get("deep_research_handoff"), dict)
        else (
            graph_cache.get("graph_derived_packet")
            if isinstance(graph_cache.get("graph_derived_packet"), dict)
            else {}
        )
    )
    payload["source_documents"] = (
        graph_cache.get("source_documents") or []
    )
    payload["expansion_inputs"] = build_expansion_inputs(
        context_pack_v2,
        comparator_seed_urls=[],
        buyer_url=profile.buyer_company_url,
    )
    return payload


async def _get_company_profile(db: AsyncSession, workspace_id: int) -> Optional[CompanyProfile]:
    result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    return result.scalar_one_or_none()


async def _sync_company_context_graph(
    company_context_pack: CompanyContextPack,
    profile: CompanyProfile,
) -> Dict[str, Any]:
    payload = build_company_context_payload(company_context_pack, profile)
    graph_payload = payload.get("company_context_graph") or {}
    sync_result = Neo4jCompanyContextGraphStore().sync_graph(graph_payload)
    sync_status = str(sync_result.get("status") or "failed")
    graph_ref = (
        payload.get("company_context_graph_ref")
        or graph_payload.get("graph_ref")
        or sync_result.get("graph_ref")
    )
    company_context_pack.company_context_graph_ref = graph_ref
    company_context_pack.company_context_graph_cache_json = graph_payload
    company_context_pack.graph_stats_json = payload.get("graph_stats") or {}
    company_context_pack.graph_sync_status = sync_status
    company_context_pack.graph_sync_error = sync_result.get("error")
    company_context_pack.graph_synced_at = datetime.utcnow()
    company_context_pack.sourcing_brief_json = payload.get("sourcing_brief") or company_context_pack.sourcing_brief_json or {}
    company_context_pack.expansion_brief_json = payload.get("expansion_brief") or company_context_pack.expansion_brief_json or {}
    payload["graph_status"] = sync_status
    payload["graph_warning"] = sync_result.get("error")
    payload["graph_synced_at"] = company_context_pack.graph_synced_at
    payload["company_context_graph_ref"] = graph_ref
    payload["source_documents"] = graph_payload.get("source_documents") or []
    return payload


async def _ensure_company_context_pack(
    db: AsyncSession,
    workspace_id: int,
    *,
    profile: Optional[CompanyProfile] = None,
) -> CompanyContextPack:
    result = await db.execute(
        select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
    )
    company_context_pack = result.scalar_one_or_none()
    if company_context_pack:
        return company_context_pack

    profile = profile or await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_payload = build_company_context_artifacts(profile)
    company_context_pack = CompanyContextPack(
        workspace_id=workspace_id,
        sourcing_brief_json=company_context_payload.get("sourcing_brief") or {},
        expansion_brief_json=company_context_payload.get("expansion_brief") or {},
        taxonomy_nodes_json=company_context_payload.get("taxonomy_nodes") or [],
        taxonomy_edges_json=company_context_payload.get("taxonomy_edges") or [],
        lens_seeds_json=company_context_payload.get("lens_seeds") or [],
        generated_at=company_context_payload.get("generated_at"),
        confirmed_at=company_context_payload.get("confirmed_at"),
    )
    db.add(company_context_pack)
    await db.flush()
    return company_context_pack


async def _run_context_pack_inline(job_id: int) -> None:
    from app.workers.workspace_tasks import generate_context_pack_v2

    await asyncio.to_thread(generate_context_pack_v2, job_id)


async def _run_company_context_refresh_inline(workspace_id: int) -> None:
    async with async_session_maker() as db:
        profile = await _get_company_profile(db, workspace_id)
        if not profile:
            return
        company_context_pack = await _ensure_company_context_pack(
            db,
            workspace_id,
            profile=profile,
        )
        try:
            refreshed = build_company_context_artifacts(profile)
            company_context_pack.sourcing_brief_json = refreshed.get("sourcing_brief") or {}
            company_context_pack.expansion_brief_json = refreshed.get("expansion_brief") or {}
            company_context_pack.taxonomy_nodes_json = refreshed.get("taxonomy_nodes") or []
            company_context_pack.taxonomy_edges_json = refreshed.get("taxonomy_edges") or []
            company_context_pack.lens_seeds_json = refreshed.get("lens_seeds") or []
            company_context_pack.generated_at = refreshed.get("generated_at")
            company_context_pack.confirmed_at = None
            company_context_pack.updated_at = datetime.utcnow()
            await _sync_company_context_graph(company_context_pack, profile)
        except Exception as exc:
            company_context_pack.graph_sync_status = "failed"
            company_context_pack.graph_sync_error = str(exc)[:1000]
            company_context_pack.graph_synced_at = datetime.utcnow()
            company_context_pack.updated_at = datetime.utcnow()
        await db.commit()


def _source_from_evidence(evidence: Optional[SourceEvidence]) -> Optional[SourcePill]:
    if not evidence or not evidence.source_url:
        return None
    if not is_trusted_source_url(evidence.source_url):
        return None
    return SourcePill(
        label=source_label_for_url(evidence.source_url),
        url=evidence.source_url,
        document_id=str(evidence.id),
        captured_at=evidence.captured_at.isoformat() if evidence.captured_at else None,
    )


def _build_claim(
    text: str,
    source_url: Optional[str],
    evidence_by_url: Dict[str, SourceEvidence],
    confidence: str = "medium",
) -> ReportClaim:
    source: Optional[SourcePill] = None
    if source_url and is_trusted_source_url(source_url):
        source = _source_from_evidence(evidence_by_url.get(source_url))
        if not source:
            source = SourcePill(
                label=source_label_for_url(source_url),
                url=source_url,
                document_id=None,
                captured_at=None,
            )
    rendering = "fact" if source else "hypothesis"
    return ReportClaim(text=text, confidence=confidence, rendering=rendering, source=source)


def _coverage_note_for_country(country: Optional[str]) -> Optional[str]:
    normalized = normalize_country(country)
    if normalized in RELIABLE_FILINGS_COUNTRIES:
        return None
    return "Not available in current reliable filings coverage"


def _pick_fact_confidence(confidence: Optional[str]) -> str:
    if confidence in {"high", "medium", "low"}:
        return confidence
    return "medium"


def _parse_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    text = str(value).strip()
    if not text:
        return None
    numbers = "".join(ch for ch in text if ch.isdigit())
    if not numbers:
        return None
    try:
        return int(numbers)
    except Exception:
        return None


def _size_range_from_claims(
    size_estimate: Optional[int],
    facts: List[CompanyFact],
    claims: List[CompanyClaim],
) -> tuple[Optional[int], Optional[int]]:
    values: List[int] = []
    if size_estimate is not None:
        values.append(size_estimate)
    for fact in facts:
        if fact.fact_key != "employees":
            continue
        parsed = _parse_int(fact.fact_value)
        if parsed is not None:
            values.append(parsed)
    for claim in claims:
        if claim.claim_key != "employees":
            continue
        parsed = _parse_int(claim.numeric_value)
        if parsed is not None:
            values.append(parsed)
    if not values:
        return None, None
    return min(values), max(values)


def _dedupe_source_pills(pills: List[SourcePill]) -> List[SourcePill]:
    deduped: List[SourcePill] = []
    seen: set[str] = set()
    for pill in pills:
        key = pill.url.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(pill)
    return deduped


def _collect_bucket_claims(
    bucket: Any,
    *,
    evidence_by_url: Dict[str, SourceEvidence],
    fallback_confidence: str = "medium",
) -> List[ReportClaim]:
    claims: List[ReportClaim] = []
    if not isinstance(bucket, list):
        return claims
    for entry in bucket:
        if not isinstance(entry, dict):
            continue
        text = str(entry.get("text") or "").strip()
        source_url = entry.get("evidence_url")
        if not text:
            continue
        claims.append(
            _build_claim(
                text=text,
                source_url=source_url,
                evidence_by_url=evidence_by_url,
                confidence=fallback_confidence,
            )
        )
    return claims


def _collect_workflow_profile(
    dossier_json: Dict[str, Any],
    *,
    evidence_by_url: Dict[str, SourceEvidence],
    fallback_capabilities: List[str],
    fallback_evidence_urls: List[str],
) -> List[ReportClaim]:
    canonical = _collect_bucket_claims(dossier_json.get("workflow"), evidence_by_url=evidence_by_url)
    if canonical:
        return canonical
    claims: List[ReportClaim] = []
    modules = modules_with_evidence(
        dossier_json,
        fallback_capabilities=fallback_capabilities,
        fallback_evidence_urls=fallback_evidence_urls,
    )
    customers, integrations = extract_customers_and_integrations(dossier_json)
    for module in modules[:6]:
        claims.append(
            _build_claim(
                text=f"Workflow capability: {module['name']}",
                source_url=(module["evidence_urls"][0] if module.get("has_evidence") else None),
                evidence_by_url=evidence_by_url,
                confidence="high" if module.get("has_evidence") else "low",
            )
        )
    for integration in integrations[:4]:
        claims.append(
            _build_claim(
                text=f"Integration surface: {integration['name']}",
                source_url=integration.get("source_url"),
                evidence_by_url=evidence_by_url,
                confidence="high" if integration.get("has_evidence") else "low",
            )
        )
    return claims


def _collect_customer_profile(
    dossier_json: Dict[str, Any],
    *,
    evidence_by_url: Dict[str, SourceEvidence],
    why_relevant: List[Dict[str, Any]],
) -> List[ReportClaim]:
    canonical = _collect_bucket_claims(dossier_json.get("customer"), evidence_by_url=evidence_by_url)
    if canonical:
        return canonical
    claims: List[ReportClaim] = []
    customers, _integrations = extract_customers_and_integrations(dossier_json)
    for customer in customers[:6]:
        claims.append(
            _build_claim(
                text=f"Customer proof: {customer['name']}",
                source_url=customer.get("source_url"),
                evidence_by_url=evidence_by_url,
                confidence="high" if customer.get("has_evidence") else "low",
            )
        )
    for reason in why_relevant[:4]:
        if not isinstance(reason, dict):
            continue
        text = str(reason.get("text") or "").strip()
        if not text:
            continue
        claims.append(
            _build_claim(
                text=f"Discovery rationale: {text}",
                source_url=reason.get("citation_url"),
                evidence_by_url=evidence_by_url,
                confidence="medium",
            )
        )
    return claims


def _collect_business_model_profile(
    dossier_json: Dict[str, Any],
    *,
    evidence_by_url: Dict[str, SourceEvidence],
) -> List[ReportClaim]:
    canonical = _collect_bucket_claims(dossier_json.get("business_model"), evidence_by_url=evidence_by_url)
    if canonical:
        return canonical
    hiring = (dossier_json or {}).get("hiring") or {}
    mix_summary = hiring.get("mix_summary") if isinstance(hiring, dict) else {}
    notes = str((mix_summary or {}).get("notes") or "").strip()
    if not notes:
        return []
    return [
        _build_claim(
            text=notes,
            source_url=None,
            evidence_by_url=evidence_by_url,
            confidence="low",
            rendering="hypothesis",
        )
    ]


def _is_directory_host_url(url: Optional[str]) -> bool:
    domain = normalize_domain(url)
    if not domain:
        return False
    return any(domain == token or domain.endswith(f".{token}") for token in DIRECTORY_HOST_TOKENS)


def _reason_codes_payload(screening: Optional[CompanyScreening]) -> Dict[str, List[str]]:
    if not screening:
        return {"positive": [], "caution": [], "reject": []}
    return {
        "positive": [str(code) for code in (screening.positive_reason_codes_json or []) if str(code)],
        "caution": [str(code) for code in (screening.caution_reason_codes_json or []) if str(code)],
        "reject": [str(code) for code in (screening.reject_reason_codes_json or []) if str(code)],
    }


async def _latest_screening_for_company(
    db: AsyncSession,
    workspace_id: int,
    company_id: int,
) -> Optional[CompanyScreening]:
    screening_result = await db.execute(
        select(CompanyScreening)
        .where(
            CompanyScreening.workspace_id == workspace_id,
            CompanyScreening.company_id == company_id,
        )
        .order_by(CompanyScreening.created_at.desc())
        .limit(1)
    )
    return screening_result.scalar_one_or_none()


def _company_claims_to_fit_card(
    company: Company,
    screening: Optional[CompanyScreening],
    screening_meta: Dict[str, Any],
    citation_summary_v1: Optional[CitationSummaryV1],
    company_claims: List[CompanyClaim],
) -> Dict[str, Any]:
    why_fit_bullets: List[Dict[str, Any]] = []
    business_model_signal: Optional[str] = None
    customer_proof: List[str] = []
    employee_signal: Optional[str] = None

    if citation_summary_v1:
        pill_by_id = {pill.pill_id: pill for pill in citation_summary_v1.source_pills}
        for sentence in citation_summary_v1.sentences[:4]:
            citation_url = None
            for pill_id in sentence.citation_pill_ids:
                pill = pill_by_id.get(pill_id)
                if pill:
                    citation_url = pill.url
                    break
            why_fit_bullets.append(
                {
                    "text": sentence.text,
                    "citation_url": citation_url,
                }
            )
    if not why_fit_bullets:
        top_claim = screening.top_claim_json if screening and isinstance(screening.top_claim_json, dict) else {}
        if top_claim.get("text"):
            why_fit_bullets.append(
                {
                    "text": str(top_claim.get("text"))[:260],
                    "citation_url": top_claim.get("source_url"),
                }
            )
    if not why_fit_bullets:
        for item in (company.why_relevant or [])[:4]:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            why_fit_bullets.append(
                {
                    "text": text[:260],
                    "citation_url": item.get("citation_url"),
                }
            )

    business_model_tokens = ("saas", "subscription", "license", "services", "implementation", "managed service", "contract")
    customer_dimensions = {"customer", "customers", "case_study"}
    for claim in company_claims:
        claim_text = str(claim.claim_text or "").strip()
        lowered = claim_text.lower()
        if not business_model_signal and any(token in lowered for token in business_model_tokens):
            business_model_signal = claim_text[:220]
        if (claim.dimension or "").lower() in customer_dimensions and claim_text:
            customer_proof.append(claim_text[:220])
        if not employee_signal and claim.claim_key == "employees":
            parsed = _parse_int(claim.numeric_value or claim.claim_text)
            if parsed is not None:
                employee_signal = f"Employee estimate: {parsed}"

    first_party_enrichment = (
        screening_meta.get("first_party_enrichment")
        if isinstance(screening_meta.get("first_party_enrichment"), dict)
        else {}
    )
    if not customer_proof and int(first_party_enrichment.get("customer_evidence_count") or 0) > 0:
        customer_proof.append(
            f"Customer proof signals detected on first-party site ({int(first_party_enrichment.get('customer_evidence_count') or 0)})."
        )
    if not employee_signal:
        candidate_size = _parse_int(screening_meta.get("candidate_employee_estimate"))
        if candidate_size is not None:
            employee_signal = f"Employee estimate: {candidate_size}"
        else:
            for tag in company.tags_custom or []:
                if not str(tag).startswith("employee_estimate:"):
                    continue
                parsed = _parse_int(str(tag).split(":", 1)[1])
                if parsed is not None:
                    employee_signal = f"Employee estimate: {parsed}"
                    break

    open_questions: List[str] = []
    if not business_model_signal:
        open_questions.append("Confirm the revenue model and services mix.")
    if not customer_proof:
        open_questions.append("Validate named customers or buyer-side proof.")
    if not employee_signal:
        open_questions.append("Confirm employee range or company-size proxy.")

    return {
        "why_fit_bullets": why_fit_bullets[:4],
        "business_model_signal": business_model_signal,
        "customer_proof": customer_proof[:4],
        "employee_signal": employee_signal,
        "open_questions": open_questions[:3],
    }


async def _company_response_from_row(db: AsyncSession, company: Company) -> CompanyResponse:
    evidence_count_result = await db.execute(
        select(func.count(SourceEvidence.id)).where(SourceEvidence.company_id == company.id)
    )
    evidence_count = evidence_count_result.scalar() or 0
    screening = await _latest_screening_for_company(db, company.workspace_id, company.id)
    screening_meta = screening.screening_meta_json if screening and isinstance(screening.screening_meta_json, dict) else {}
    citation_summary_v1 = _citation_summary_from_meta(screening_meta)
    diagnostics = _screening_diagnostics_from_meta(screening_meta)
    company_claims_result = await db.execute(
        select(CompanyClaim)
        .where(
            CompanyClaim.workspace_id == company.workspace_id,
            CompanyClaim.company_id == company.id,
        )
        .order_by(CompanyClaim.created_at.desc(), CompanyClaim.id.desc())
        .limit(24)
    )
    company_claims = company_claims_result.scalars().all()
    fit_card = _company_claims_to_fit_card(company, screening, screening_meta, citation_summary_v1, company_claims)
    return CompanyResponse(
        id=company.id,
        workspace_id=company.workspace_id,
        name=company.name,
        website=company.website,
        official_website_url=(
            screening.candidate_official_website
            if screening and screening.candidate_official_website
            else company.website
        ),
        discovery_url=screening.candidate_discovery_url if screening else None,
        entity_type=str(screening_meta.get("entity_type") or "company") if screening else "company",
        hq_country=company.hq_country,
        operating_countries=company.operating_countries or [],
        tags_custom=company.tags_custom or [],
        status=company.status.value,
        why_relevant=company.why_relevant or [],
        is_manual=company.is_manual,
        created_at=company.created_at,
        evidence_count=evidence_count,
        decision_classification=screening.decision_classification if screening else None,
        evidence_sufficiency=screening.evidence_sufficiency if screening else None,
        reason_codes=_reason_codes_payload(screening),
        rationale_summary=screening.rationale_summary if screening else None,
        top_claim=screening.top_claim_json if screening and isinstance(screening.top_claim_json, dict) else {},
        citation_summary_v1=citation_summary_v1,
        registry_neighbors_with_first_party_website_count=int(diagnostics["registry_neighbors_with_first_party_website_count"]),
        registry_neighbors_dropped_missing_official_website_count=int(
            diagnostics["registry_neighbors_dropped_missing_official_website_count"]
        ),
        registry_origin_screening_counts=diagnostics["registry_origin_screening_counts"],
        first_party_hint_urls_used_count=int(diagnostics["first_party_hint_urls_used_count"]),
        first_party_hint_pages_crawled_total=int(diagnostics["first_party_hint_pages_crawled_total"]),
        unresolved_contradictions_count=screening.unresolved_contradictions_count if screening else 0,
        why_fit_bullets=fit_card["why_fit_bullets"],
        business_model_signal=fit_card["business_model_signal"],
        customer_proof=fit_card["customer_proof"],
        employee_signal=fit_card["employee_signal"],
        open_questions=fit_card["open_questions"],
    )


def _to_workspace_feedback_response(event: WorkspaceFeedbackEvent) -> WorkspaceFeedbackResponse:
    return WorkspaceFeedbackResponse(
        id=event.id,
        workspace_id=event.workspace_id,
        company_id=event.company_id,
        company_screening_id=event.company_screening_id,
        feedback_type=event.feedback_type,
        previous_classification=event.previous_classification,
        new_classification=event.new_classification,
        reason_codes=[str(code) for code in (event.reason_codes_json or []) if str(code)],
        comment=event.comment,
        metadata=event.metadata_json or {},
        created_by=event.created_by,
        created_at=event.created_at,
    )


# ============================================================================
# Workspace CRUD
# ============================================================================

@router.post("", response_model=WorkspaceResponse)
async def create_workspace(
    data: WorkspaceCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create a new workspace with company-context scaffolding."""
    workspace = Workspace(
        name=data.name,
        region_scope=data.region_scope,
        decision_policy_json=normalize_policy(DEFAULT_EVIDENCE_POLICY),
    )
    db.add(workspace)
    await db.flush()
    
    # Create company profile
    profile = CompanyProfile(
        workspace_id=workspace.id,
        geo_scope={"region": data.region_scope, "include_countries": [], "exclude_countries": []}
    )
    db.add(profile)

    await db.commit()
    await db.refresh(workspace)
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        region_scope=workspace.region_scope,
        created_at=workspace.created_at,
        company_count=0,
        has_context_pack=False,
        has_confirmed_scope_review=False
    )


@router.get("", response_model=List[WorkspaceResponse])
async def list_workspaces(db: AsyncSession = Depends(get_db)):
    """List all workspaces with summary stats."""
    result = await db.execute(
        select(Workspace).order_by(Workspace.created_at.desc())
    )
    workspaces = result.scalars().all()
    
    responses = []
    for ws in workspaces:
        # Get company count
        company_count_result = await db.execute(
            select(func.count(Company.id)).where(Company.workspace_id == ws.id)
        )
        company_count = company_count_result.scalar() or 0
        
        # Check context pack
        profile_result = await db.execute(
            select(CompanyProfile).where(CompanyProfile.workspace_id == ws.id)
        )
        profile = profile_result.scalar_one_or_none()
        company_context_result = await db.execute(
            select(CompanyContextPack).where(CompanyContextPack.workspace_id == ws.id)
        )
        company_context_pack = company_context_result.scalar_one_or_none()
        has_context_pack = bool(
            (company_context_pack and (company_context_pack.sourcing_brief_json or company_context_pack.company_context_graph_ref))
            or (profile and profile.context_pack_markdown)
        )

        has_confirmed_scope_review = bool(
            company_context_pack
            and normalize_expansion_brief(company_context_pack.expansion_brief_json or {}).get("confirmed_at")
        )
        
        responses.append(WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            region_scope=ws.region_scope,
            created_at=ws.created_at,
            company_count=company_count,
            has_context_pack=has_context_pack,
            has_confirmed_scope_review=has_confirmed_scope_review
        ))
    
    return responses


@router.get("/{workspace_id}", response_model=WorkspaceResponse)
async def get_workspace(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get a single workspace."""
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    company_count_result = await db.execute(
        select(func.count(Company.id)).where(Company.workspace_id == workspace_id)
    )
    company_count = company_count_result.scalar() or 0
    
    profile_result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = profile_result.scalar_one_or_none()
    company_context_result = await db.execute(
        select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
    )
    company_context_pack = company_context_result.scalar_one_or_none()
    has_context_pack = bool(
        (company_context_pack and (company_context_pack.sourcing_brief_json or company_context_pack.company_context_graph_ref))
        or (profile and profile.context_pack_markdown)
    )

    has_confirmed_scope_review = bool(
        company_context_pack
        and normalize_expansion_brief(company_context_pack.expansion_brief_json or {}).get("confirmed_at")
    )
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        region_scope=workspace.region_scope,
        created_at=workspace.created_at,
        company_count=company_count,
        has_context_pack=has_context_pack,
        has_confirmed_scope_review=has_confirmed_scope_review
    )


@router.patch("/{workspace_id}", response_model=WorkspaceResponse)
async def update_workspace(
    workspace_id: int,
    data: WorkspaceUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update workspace name or region."""
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    if data.name is not None:
        workspace.name = data.name
    if data.region_scope is not None:
        profile_result = await db.execute(
            select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
        )
        profile = profile_result.scalar_one_or_none()
        workspace.region_scope = data.region_scope
        if profile:
            geo_scope = profile.geo_scope if isinstance(profile.geo_scope, dict) else {}
            profile.geo_scope = {
                **geo_scope,
                "region": data.region_scope,
                "include_countries": geo_scope.get("include_countries") or [],
                "exclude_countries": geo_scope.get("exclude_countries") or [],
            }

    await db.commit()
    await db.refresh(workspace)
    
    return await get_workspace(workspace_id, db)


@router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a workspace and all associated data."""
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    await db.delete(workspace)
    await db.commit()
    return {"deleted": True}


# ============================================================================
# Context Pack
# ============================================================================

@router.get("/{workspace_id}/context-pack", response_model=CompanyProfileResponse)
async def get_context_pack(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get the context pack / company profile for a workspace."""
    result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    
    return CompanyProfileResponse(
        id=profile.id,
        workspace_id=profile.workspace_id,
        buyer_company_url=profile.buyer_company_url,
        comparator_seed_urls=profile.comparator_seed_urls or [],
        supporting_evidence_urls=profile.supporting_evidence_urls or [],
        comparator_seed_summaries=profile.comparator_seed_summaries or {},
        geo_scope=profile.geo_scope or {},
        context_pack_markdown=profile.context_pack_markdown,
        context_pack_generated_at=profile.context_pack_generated_at,
        product_pages_found=profile.product_pages_found or 0,
        context_pack_json=profile.context_pack_json
    )


class ExportRequest(BaseModel):
    include_markdown: bool = True


class ExportResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None


@router.post("/{workspace_id}/context-pack:export", response_model=ExportResponse)
async def export_context_pack(
    workspace_id: int,
    request: ExportRequest = None,
    db: AsyncSession = Depends(get_db)
):
    """Export context pack as API payload only (no server-side file writes)."""
    result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    
    if not profile.context_pack_json:
        raise HTTPException(status_code=400, detail="Context pack not generated yet. Run refresh first.")
    
    include_markdown = True if request is None else bool(request.include_markdown)

    export_data = {
        "workspace_id": workspace_id,
        "buyer_company_url": profile.buyer_company_url,
        "comparator_seed_urls": profile.comparator_seed_urls or [],
        "supporting_evidence_urls": profile.supporting_evidence_urls or [],
        "comparator_seed_summaries": profile.comparator_seed_summaries or {},
        "geo_scope": profile.geo_scope or {},
        "product_pages_found": profile.product_pages_found or 0,
        "context_pack": profile.context_pack_json,
    }
    if include_markdown:
        export_data["context_pack_markdown"] = profile.context_pack_markdown

    return ExportResponse(success=True, data=export_data)


@router.patch("/{workspace_id}/context-pack", response_model=CompanyProfileResponse)
async def update_context_pack(
    workspace_id: int,
    data: CompanyProfileUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update context pack inputs."""
    result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    
    if data.buyer_company_url is not None:
        profile.buyer_company_url = data.buyer_company_url
    if data.comparator_seed_urls is not None:
        profile.comparator_seed_urls = _clean_url_list(data.comparator_seed_urls, max_items=10)
    if data.supporting_evidence_urls is not None:
        profile.supporting_evidence_urls = _clean_url_list(data.supporting_evidence_urls, max_items=50)
    if data.geo_scope is not None:
        profile.geo_scope = data.geo_scope.model_dump()
    
    profile.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(profile)
    
    return await get_context_pack(workspace_id, db)


@router.post("/{workspace_id}/context-pack:refresh", response_model=JobResponse)
async def refresh_context_pack(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Trigger context pack generation job."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    active_jobs_result = await db.execute(
        select(Job).where(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.context_pack,
            Job.state.in_(DB_ACTIVE_JOB_STATES),
        )
    )
    active_jobs = active_jobs_result.scalars().all()
    for active_job in active_jobs:
        active_job.state = JobState.failed
        active_job.error_message = "Superseded by newer sourcing brief run"
        active_job.progress_message = "Superseded by newer sourcing brief run"
        active_job.finished_at = datetime.utcnow()
        if active_job.interaction_id:
            try:
                from app.workers.celery_app import celery_app

                celery_app.control.revoke(active_job.interaction_id, terminate=False)
            except Exception:
                pass
    if active_jobs:
        await db.commit()
    
    # Create job
    job = Job(
        workspace_id=workspace_id,
        job_type=JobType.context_pack,
        state=JobState.queued,
        provider=JobProvider.crawler
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Trigger async task (Celery) and fall back to inline execution if the queue path is unavailable.
    from app.workers.workspace_tasks import generate_context_pack_v2
    try:
        task_result = generate_context_pack_v2.delay(job.id)
        job.interaction_id = str(task_result.id)
        await db.commit()
        await db.refresh(job)
    except Exception as exc:
        job.progress_message = "Background crawl worker unavailable, running inline"
        await db.commit()
        try:
            await _run_context_pack_inline(job.id)
        except Exception as inline_exc:
            job.state = JobState.failed
            job.error_message = "Context-pack generation failed after queue fallback"
            job.progress_message = "Failed to start context-pack generation"
            job.finished_at = datetime.utcnow()
            await db.commit()
            raise HTTPException(
                status_code=503,
                detail=(
                    "Could not start website crawl. "
                    f"Background crawl worker unavailable ({exc.__class__.__name__}); "
                    f"inline fallback failed ({inline_exc.__class__.__name__})."
                ),
            ) from inline_exc
        await db.refresh(job)
    
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        company_id=job.company_id,
        job_type=job.job_type.value,
        state=job.state.value,
        provider=job.provider.value,
        progress=job.progress,
        progress_message=job.progress_message,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at
    )


# ============================================================================
# Company Context
# ============================================================================

@router.get("/{workspace_id}/company-context", response_model=CompanyContextPackResponse)
async def get_company_context(workspace_id: int, db: AsyncSession = Depends(get_db)):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    full_context_pack_v2 = build_context_pack_v2(profile.context_pack_json or {})
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )
    await db.commit()
    await db.refresh(company_context_pack)
    has_graph_cache = bool(
        company_context_pack.company_context_graph_ref
        and isinstance(company_context_pack.company_context_graph_cache_json, dict)
        and company_context_pack.company_context_graph_cache_json
    )
    if has_graph_cache:
        payload = _company_context_payload_from_pack(company_context_pack, profile=profile)
    else:
        payload = await _sync_company_context_graph(company_context_pack, profile)
        await db.commit()
        await db.refresh(company_context_pack)
    payload["context_pack_v2"] = full_context_pack_v2
    payload["expansion_inputs"] = build_expansion_inputs(
        payload.get("context_pack_v2") or {},
        comparator_seed_urls=[],
        buyer_url=profile.buyer_company_url or ((payload.get("sourcing_brief") or {}).get("source_company") or {}).get("website"),
    )
    payload["workspace_id"] = workspace_id
    payload["id"] = company_context_pack.id
    return _company_context_response_from_payload(payload)


@router.patch("/{workspace_id}/company-context", response_model=CompanyContextPackResponse)
async def update_company_context(
    workspace_id: int,
    data: CompanyContextPackUpdate,
    db: AsyncSession = Depends(get_db),
):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )
    if data.source_summary is not None:
        sourcing_brief = company_context_pack.sourcing_brief_json if isinstance(company_context_pack.sourcing_brief_json, dict) else {}
        sourcing_brief["source_summary"] = str(data.source_summary or "").strip()[:8000] or None
        company_context_pack.sourcing_brief_json = sourcing_brief
    if data.taxonomy_nodes is not None:
        normalized_nodes = normalize_taxonomy_nodes([item.model_dump() for item in data.taxonomy_nodes])
        (
            _context_pack_v2,
            rebuilt_nodes,
            rebuilt_edges,
            rebuilt_lens_seeds,
            rebuilt_open_questions,
            rebuilt_sourcing_brief,
        ) = _build_sourcing_brief_artifacts(
            profile,
            source_pills=_derive_source_pills_from_profile(profile),
            override_nodes=normalized_nodes,
        )
        company_context_pack.taxonomy_nodes_json = rebuilt_nodes
        company_context_pack.taxonomy_edges_json = rebuilt_edges
        company_context_pack.lens_seeds_json = rebuilt_lens_seeds
        company_context_pack.sourcing_brief_json = {
            **rebuilt_sourcing_brief,
            "source_summary": (
                ((company_context_pack.sourcing_brief_json or {}).get("source_summary"))
                if isinstance(company_context_pack.sourcing_brief_json, dict)
                else None
            ) or rebuilt_sourcing_brief.get("source_summary"),
            "confirmed_at": company_context_pack.confirmed_at.isoformat() if company_context_pack.confirmed_at else None,
        }
        company_context_pack.sourcing_brief_json["open_questions"] = normalize_open_questions(rebuilt_open_questions)
    if data.confirmed is not None:
        company_context_pack.confirmed_at = datetime.utcnow() if data.confirmed else None
    sourcing_brief = company_context_pack.sourcing_brief_json if isinstance(company_context_pack.sourcing_brief_json, dict) else {}
    sourcing_brief["confirmed_at"] = company_context_pack.confirmed_at.isoformat() if company_context_pack.confirmed_at else None
    company_context_pack.sourcing_brief_json = sourcing_brief
    company_context_pack.updated_at = datetime.utcnow()
    payload = await _sync_company_context_graph(company_context_pack, profile)
    await db.commit()
    await db.refresh(company_context_pack)
    payload["expansion_inputs"] = build_expansion_inputs(
        payload.get("context_pack_v2") or {},
        comparator_seed_urls=[],
        buyer_url=profile.buyer_company_url or ((payload.get("sourcing_brief") or {}).get("source_company") or {}).get("website"),
    )
    payload["workspace_id"] = workspace_id
    payload["id"] = company_context_pack.id
    return _company_context_response_from_payload(payload)


@router.post("/{workspace_id}/company-context:refresh", response_model=CompanyContextPackResponse)
async def refresh_company_context(
    workspace_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )
    if company_context_pack.graph_sync_status == "refreshing":
        payload = _company_context_payload_from_pack(company_context_pack, profile=profile)
        payload["workspace_id"] = workspace_id
        payload["id"] = company_context_pack.id
        return _company_context_refresh_response_from_payload(payload)

    company_context_pack.graph_sync_status = "refreshing"
    company_context_pack.graph_sync_error = None
    company_context_pack.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(company_context_pack)
    background_tasks.add_task(_run_company_context_refresh_inline, workspace_id)
    payload = _company_context_payload_from_pack(company_context_pack, profile=profile)
    payload["workspace_id"] = workspace_id
    payload["id"] = company_context_pack.id
    return _company_context_refresh_response_from_payload(payload)

# ============================================================================
# Scope Review
# ============================================================================

@router.get("/{workspace_id}/scope-review", response_model=ScopeReviewResponse)
async def get_scope_review(workspace_id: int, db: AsyncSession = Depends(get_db)):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )
    scope_payload = derive_scope_review_payload(company_context_pack, profile)
    return _scope_review_response_from_payload(workspace_id, scope_payload)


@router.patch("/{workspace_id}/scope-review", response_model=ScopeReviewResponse)
async def update_scope_review(
    workspace_id: int,
    data: ScopeReviewUpdate,
    db: AsyncSession = Depends(get_db),
):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )

    adjusted = apply_scope_review_decisions(
        company_context_pack,
        [item.model_dump() for item in data.decisions],
    )
    company_context_pack.taxonomy_nodes_json = adjusted.get("taxonomy_nodes") or company_context_pack.taxonomy_nodes_json
    updated_expansion_brief = adjusted.get("expansion_brief") or company_context_pack.expansion_brief_json or {}
    if isinstance(updated_expansion_brief, dict):
        updated_expansion_brief["confirmed_at"] = None
    company_context_pack.expansion_brief_json = updated_expansion_brief
    company_context_pack.updated_at = datetime.utcnow()
    await _sync_company_context_graph(company_context_pack, profile)
    await db.commit()
    await db.refresh(company_context_pack)

    scope_payload = derive_scope_review_payload(company_context_pack, profile)
    return _scope_review_response_from_payload(workspace_id, scope_payload)


@router.post("/{workspace_id}/scope-review:confirm", response_model=ScopeReviewResponse)
async def confirm_scope_review(workspace_id: int, db: AsyncSession = Depends(get_db)):
    profile = await _get_company_profile(db, workspace_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Context pack not found")
    company_context_pack = await _ensure_company_context_pack(
        db,
        workspace_id,
        profile=profile,
    )

    confirmed_at = datetime.utcnow()
    expansion_brief = normalize_expansion_brief(company_context_pack.expansion_brief_json or {})
    expansion_brief["confirmed_at"] = confirmed_at.isoformat()
    company_context_pack.expansion_brief_json = expansion_brief
    company_context_pack.updated_at = confirmed_at
    await _sync_company_context_graph(company_context_pack, profile)
    await db.commit()
    await db.refresh(company_context_pack)

    scope_payload = derive_scope_review_payload(company_context_pack, profile)
    return _scope_review_response_from_payload(workspace_id, scope_payload)


# ============================================================================
# Discovery & Companies
# ============================================================================

@router.post("/{workspace_id}/discovery:run", response_model=JobResponse)
async def run_discovery(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Run discovery to find candidate universe."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Create job
    job = Job(
        workspace_id=workspace_id,
        job_type=JobType.discovery_universe,
        state=JobState.queued,
        provider=JobProvider.gemini_flash
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)
    
    # Trigger async task
    from app.workers.workspace_tasks import run_discovery_universe
    task_result = run_discovery_universe.delay(job.id)
    job.interaction_id = str(task_result.id)
    await db.commit()
    await db.refresh(job)
    
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        company_id=job.company_id,
        job_type=job.job_type.value,
        state=job.state.value,
        provider=job.provider.value,
        progress=job.progress,
        progress_message=job.progress_message,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at
    )


@router.get("/{workspace_id}/discovery:diagnostics")
async def get_discovery_diagnostics(
    workspace_id: int,
    include_quality_audit: bool = Query(True),
    db: AsyncSession = Depends(get_db),
):
    """Return screening diagnostics for the latest discovery run."""
    workspace_result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    latest_discovery_job = await _latest_completed_discovery_job(db, workspace_id)
    latest_discovery_job_result: Dict[str, Any] = (
        latest_discovery_job.result_json
        if latest_discovery_job and isinstance(latest_discovery_job.result_json, dict)
        else {}
    )
    quality_payload = _quality_payload_from_job_result(latest_discovery_job_result)
    latest_run_id = quality_payload.get("screening_run_id")
    variance_hotspots_v1 = await _variance_hotspots_for_workspace(db, workspace_id, limit_runs=30)

    screenings_query = (
        select(CompanyScreening)
        .where(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(1000)
    )
    screenings_result = await db.execute(screenings_query)
    screenings = screenings_result.scalars().all()

    if latest_run_id:
        screenings = [
            screening for screening in screenings
            if (screening.screening_meta_json or {}).get("screening_run_id") == latest_run_id
        ]
    elif screenings:
        fallback_run_id = (screenings[0].screening_meta_json or {}).get("screening_run_id")
        if fallback_run_id:
            latest_run_id = fallback_run_id
            screenings = [
                screening for screening in screenings
                if (screening.screening_meta_json or {}).get("screening_run_id") == fallback_run_id
            ]

    source_runs_result = await db.execute(
        select(ComparatorSourceRun)
        .where(ComparatorSourceRun.workspace_id == workspace_id)
        .order_by(ComparatorSourceRun.captured_at.desc())
        .limit(10)
    )
    source_runs = source_runs_result.scalars().all()

    source_coverage = [
        {
            "source_name": run.source_name,
            "source_url": run.source_url,
            "status": run.status,
            "mentions_found": run.mentions_found,
            "pages_crawled": (run.metadata_json or {}).get("pages_crawled", 0),
            "errors": (run.metadata_json or {}).get("errors", []),
            "captured_at": run.captured_at.isoformat() if run.captured_at else None,
        }
        for run in source_runs
    ]

    registry_logs: List[RegistryQueryLog] = []
    if latest_run_id:
        registry_log_result = await db.execute(
            select(RegistryQueryLog).where(
                RegistryQueryLog.workspace_id == workspace_id,
                RegistryQueryLog.run_id == latest_run_id,
            )
        )
        registry_logs = registry_log_result.scalars().all()

    registry_queries_by_country: Dict[str, int] = {}
    registry_raw_hits_by_country: Dict[str, int] = {}
    registry_reject_reason_breakdown: Dict[str, int] = {}
    registry_neighbors_kept_pre_dedupe = 0
    for log in registry_logs:
        country_key = log.country or "UNKNOWN"
        registry_queries_by_country[country_key] = registry_queries_by_country.get(country_key, 0) + 1
        registry_raw_hits_by_country[country_key] = registry_raw_hits_by_country.get(country_key, 0) + int(log.raw_hits or 0)
        registry_neighbors_kept_pre_dedupe += int(log.kept_hits or 0)
        for reason, count in (log.reject_reasons_json or {}).items():
            key = str(reason)
            registry_reject_reason_breakdown[key] = registry_reject_reason_breakdown.get(key, 0) + int(count or 0)

    entity_result = await db.execute(
        select(CandidateEntity).where(CandidateEntity.workspace_id == workspace_id)
    )
    candidate_entities = entity_result.scalars().all()
    entity_ids = [entity.id for entity in candidate_entities]
    entity_aliases: List[CandidateEntityAlias] = []
    entity_origins: List[CandidateOriginEdge] = []
    if entity_ids:
        alias_result = await db.execute(
            select(CandidateEntityAlias).where(CandidateEntityAlias.entity_id.in_(entity_ids))
        )
        entity_aliases = alias_result.scalars().all()
        origin_result = await db.execute(
            select(CandidateOriginEdge).where(CandidateOriginEdge.entity_id.in_(entity_ids))
        )
        entity_origins = origin_result.scalars().all()

    alias_count_by_entity: Dict[int, int] = {}
    for alias in entity_aliases:
        alias_count_by_entity[alias.entity_id] = alias_count_by_entity.get(alias.entity_id, 0) + 1
    alias_clusters_count = len(
        [entity_id for entity_id, count in alias_count_by_entity.items() if count > 1]
    )

    origin_mix_distribution: Dict[str, int] = {}
    for edge in entity_origins:
        origin_type = edge.origin_type or "unknown"
        origin_mix_distribution[origin_type] = origin_mix_distribution.get(origin_type, 0) + 1

    screening_ids = [row.id for row in screenings]
    claims: List[CompanyClaim] = []
    if screening_ids:
        claims_result = await db.execute(
            select(CompanyClaim).where(CompanyClaim.company_screening_id.in_(screening_ids))
        )
        claims = claims_result.scalars().all()

    kept_count = len([row for row in screenings if row.screening_status == "kept"])
    review_count = len([row for row in screenings if row.screening_status == "review"])
    rejected_count = len([row for row in screenings if row.screening_status == "rejected"])
    ranking_eligible_count = len([row for row in screenings if bool(row.ranking_eligible)])
    directory_only_count = len(
        [
            row for row in screenings
            if not str(row.candidate_official_website or "").strip()
            or _is_directory_host_url(row.candidate_official_website)
        ]
    )
    solution_entity_count = len(
        [
            row for row in screenings
            if str(
                ((row.screening_meta_json or {}).get("entity_type") if isinstance(row.screening_meta_json, dict) else "")
                or ""
            ).strip().lower() == "solution"
        ]
    )
    official_website_resolution_rate = round(
        (
            len(
                [
                    row for row in screenings
                    if str(row.candidate_official_website or "").strip()
                    and not _is_directory_host_url(row.candidate_official_website)
                ]
            )
            / max(1, len(screenings))
        ),
        4,
    )

    reject_reason_counts: Dict[str, int] = {}
    for screening in screenings:
        for reason in (screening.reject_reasons_json or []):
            key = str(reason).strip() or "unknown_reason"
            reject_reason_counts[key] = reject_reason_counts.get(key, 0) + 1

    source_quality_distribution: Dict[str, int] = {}
    evidence_claims_by_screening: Dict[int, int] = {}
    dimension_coverage = {"icp": 0, "product_services": 0, "customers": 0, "size": 0, "moat": 0}
    seen_dimensions_per_screening: Dict[int, set[str]] = {}
    for claim in claims:
        source_type = claim.source_type or "unknown"
        source_quality_distribution[source_type] = source_quality_distribution.get(source_type, 0) + 1
        if not claim.company_screening_id:
            continue
        evidence_claims_by_screening[claim.company_screening_id] = (
            evidence_claims_by_screening.get(claim.company_screening_id, 0) + 1
        )
        seen_dimensions_per_screening.setdefault(claim.company_screening_id, set()).add((claim.dimension or "").lower())

    for dimensions in seen_dimensions_per_screening.values():
        if any(d in dimensions for d in {"icp", "target_customer"}):
            dimension_coverage["icp"] += 1
        if any(d in dimensions for d in {"capability", "product", "services", "directory_context", "evidence"}):
            dimension_coverage["product_services"] += 1
        if any(d in dimensions for d in {"customers", "customer", "case_study"}):
            dimension_coverage["customers"] += 1
        if "employees" in dimensions or "size" in dimensions:
            dimension_coverage["size"] += 1
        if "moat" in dimensions:
            dimension_coverage["moat"] += 1

    evidence_coverage_stats = {
        "screenings_with_any_claim": len([sid for sid, count in evidence_claims_by_screening.items() if count > 0]),
        "avg_claims_per_screening": round(
            (sum(evidence_claims_by_screening.values()) / max(1, len(evidence_claims_by_screening))),
            2,
        ),
        "dimension_coverage": dimension_coverage,
    }

    return {
        "workspace_id": workspace_id,
        "screening_run_id": latest_run_id,
        "screening_totals": {
            "screenings": len(screenings),
            "kept": kept_count,
            "review": review_count,
            "rejected": rejected_count,
        },
        "filter_reason_counts": reject_reason_counts,
        "evidence_coverage_stats": evidence_coverage_stats,
        "source_quality_distribution": source_quality_distribution,
        "source_coverage": source_coverage,
        "funnel_metrics": {
            "seed_directory_count": int(latest_discovery_job_result.get("seed_directory_count", 0)),
            "seed_reference_count": int(latest_discovery_job_result.get("seed_reference_count", 0)),
            "seed_llm_count": int(latest_discovery_job_result.get("seed_llm_count", 0)),
            "identity_resolved_count": int(latest_discovery_job_result.get("identity_resolved_count", 0)),
            "registry_identity_candidates_count": int(latest_discovery_job_result.get("registry_identity_candidates_count", 0)),
            "registry_identity_mapped_count": int(latest_discovery_job_result.get("registry_identity_mapped_count", 0)),
            "registry_identity_country_breakdown": latest_discovery_job_result.get("registry_identity_country_breakdown", {}),
            "alias_clusters_count": int(
                latest_discovery_job_result.get("alias_clusters_count", alias_clusters_count)
            ),
            "duplicates_collapsed_count": int(latest_discovery_job_result.get("duplicates_collapsed_count", 0)),
            "registry_queries_count": int(
                latest_discovery_job_result.get("registry_queries_count", sum(registry_queries_by_country.values()))
            ),
            "registry_neighbors_kept_count": int(
                latest_discovery_job_result.get("registry_neighbors_kept_count", 0)
            ),
            "registry_neighbors_with_first_party_website_count": int(
                latest_discovery_job_result.get("registry_neighbors_with_first_party_website_count", 0)
            ),
            "registry_neighbors_dropped_missing_official_website_count": int(
                latest_discovery_job_result.get("registry_neighbors_dropped_missing_official_website_count", 0)
            ),
            "registry_origin_screening_counts": latest_discovery_job_result.get(
                "registry_origin_screening_counts",
                {},
            ),
            "registry_queries_by_country": latest_discovery_job_result.get("registry_queries_by_country", registry_queries_by_country),
            "registry_raw_hits_by_country": latest_discovery_job_result.get("registry_raw_hits_by_country", registry_raw_hits_by_country),
            "registry_neighbors_kept_pre_dedupe": int(
                latest_discovery_job_result.get("registry_neighbors_kept_pre_dedupe", registry_neighbors_kept_pre_dedupe)
            ),
            "registry_neighbors_unique_post_dedupe": int(
                latest_discovery_job_result.get("registry_neighbors_unique_post_dedupe", 0)
            ),
            "registry_reject_reason_breakdown": latest_discovery_job_result.get(
                "registry_reject_reason_breakdown",
                registry_reject_reason_breakdown,
            ),
            "final_universe_count": int(
                latest_discovery_job_result.get("final_universe_count", len(candidate_entities))
            ),
            "first_party_crawl_attempted_count": int(
                latest_discovery_job_result.get("first_party_crawl_attempted_count", 0)
            ),
            "first_party_crawl_success_count": int(
                latest_discovery_job_result.get("first_party_crawl_success_count", 0)
            ),
            "first_party_crawl_failed_count": int(
                latest_discovery_job_result.get("first_party_crawl_failed_count", 0)
            ),
            "first_party_crawl_deep_count": int(
                latest_discovery_job_result.get("first_party_crawl_deep_count", 0)
            ),
            "first_party_crawl_light_count": int(
                latest_discovery_job_result.get("first_party_crawl_light_count", 0)
            ),
            "first_party_crawl_pages_total": int(
                latest_discovery_job_result.get("first_party_crawl_pages_total", 0)
            ),
            "first_party_crawl_fallback_count": int(
                latest_discovery_job_result.get("first_party_crawl_fallback_count", 0)
            ),
            "first_party_hint_urls_used_count": int(
                latest_discovery_job_result.get("first_party_hint_urls_used_count", 0)
            ),
            "first_party_hint_pages_crawled_total": int(
                latest_discovery_job_result.get("first_party_hint_pages_crawled_total", 0)
            ),
            "first_party_hint_domain_stats": (
                latest_discovery_job_result.get("first_party_hint_domain_stats")
                if isinstance(latest_discovery_job_result.get("first_party_hint_domain_stats"), dict)
                else {}
            ),
        },
        "origin_mix_distribution": latest_discovery_job_result.get("origin_mix_distribution", origin_mix_distribution),
        "dedupe_quality_metrics": latest_discovery_job_result.get("dedupe_quality_metrics", {}),
        "registry_expansion_yield": latest_discovery_job_result.get("registry_expansion_yield", {}),
        "ranking_eligibility": {
            "ranking_eligible_count": ranking_eligible_count,
            "directory_only_count": directory_only_count,
            "solution_entity_count": solution_entity_count,
            "official_website_resolution_rate": official_website_resolution_rate,
        },
        "ranking_eligible_count": ranking_eligible_count,
        "directory_only_count": directory_only_count,
        "solution_entity_count": solution_entity_count,
        "official_website_resolution_rate": official_website_resolution_rate,
        "run_quality_tier": quality_payload["run_quality_tier"],
        "quality_gate_passed": bool(quality_payload["quality_gate_passed"]),
        "quality_audit_passed": bool(quality_payload["quality_audit_passed"]),
        "quality_validation_ready": bool(quality_payload["quality_validation_ready"]),
        "quality_validation_blocked_reasons": quality_payload["quality_validation_blocked_reasons"],
        "pre_rerun_quality_audit_run_id": quality_payload["pre_rerun_quality_audit_run_id"],
        "pre_rerun_quality_validation_ready": bool(quality_payload["pre_rerun_quality_validation_ready"]),
        "pre_rerun_quality_validation_blocked_reasons": quality_payload[
            "pre_rerun_quality_validation_blocked_reasons"
        ],
        "degraded_reasons": quality_payload["degraded_reasons"],
        "model_attempt_trace": quality_payload["model_attempt_trace"],
        "stage_time_ms": quality_payload["stage_time_ms"],
        "timeout_events": quality_payload["timeout_events"],
        "queue_wait_ms_by_stage": quality_payload["queue_wait_ms_by_stage"],
        "stage_retry_counts": quality_payload["stage_retry_counts"],
        "cache_hit_rates": quality_payload["cache_hit_rates"],
        "candidate_dropoff_funnel_v1": quality_payload["candidate_dropoff_funnel_v1"],
        "stage_execution_mode": quality_payload["stage_execution_mode"],
        "variance_hotspots_v1": variance_hotspots_v1,
        "generated_at": datetime.utcnow().isoformat(),
    }
    if include_quality_audit:
        response_payload["quality_audit_v1"] = quality_payload["quality_audit_v1"]
        response_payload["pre_rerun_quality_audit_v1"] = quality_payload["pre_rerun_quality_audit_v1"]
    return response_payload


@router.get("/{workspace_id}/companies", response_model=List[CompanyResponse])
async def list_companies(
    workspace_id: int,
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List companies in workspace with optional status filter."""
    query = select(Company).where(Company.workspace_id == workspace_id)
    
    if status:
        query = query.where(Company.status == CompanyStatus(status))
    
    query = query.order_by(Company.created_at.desc())
    result = await db.execute(query)
    companies = result.scalars().all()

    responses: List[CompanyResponse] = []
    for row in companies:
        responses.append(await _company_response_from_row(db, row))
    return responses


@router.get("/{workspace_id}/universe/top-candidates", response_model=List[UniverseTopCandidateResponse])
async def list_top_candidates(
    workspace_id: int,
    limit: int = Query(25, ge=1, le=200),
    allow_degraded: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    latest_discovery_job = await _latest_completed_discovery_job(db, workspace_id)
    latest_discovery_job_result: Dict[str, Any] = (
        latest_discovery_job.result_json
        if latest_discovery_job and isinstance(latest_discovery_job.result_json, dict)
        else {}
    )
    quality_payload = _quality_payload_from_job_result(latest_discovery_job_result)

    screenings_result = await db.execute(
        select(CompanyScreening)
        .where(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(4000)
    )
    screenings_all = screenings_result.scalars().all()
    if not screenings_all:
        return []

    latest_run_id = quality_payload.get("screening_run_id")
    if latest_run_id:
        screenings = [
            row for row in screenings_all
            if (row.screening_meta_json or {}).get("screening_run_id") == latest_run_id
        ]
    else:
        fallback_run_id = (screenings_all[0].screening_meta_json or {}).get("screening_run_id")
        if fallback_run_id:
            latest_run_id = fallback_run_id
            screenings = [
                row for row in screenings_all
                if (row.screening_meta_json or {}).get("screening_run_id") == latest_run_id
            ]
        else:
            screenings = screenings_all

    if quality_payload["run_quality_tier"] == "degraded" and not allow_degraded:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "degraded_run_blocked",
                "message": "Latest completed discovery run is degraded. Pass allow_degraded=true to inspect it.",
                "screening_run_id": latest_run_id,
                "run_quality_tier": quality_payload["run_quality_tier"],
                "quality_gate_passed": bool(quality_payload["quality_gate_passed"]),
                "quality_audit_passed": bool(quality_payload["quality_audit_passed"]),
                "degraded_reasons": quality_payload["degraded_reasons"],
            },
        )

    candidates = [row for row in screenings if bool(row.ranking_eligible)]
    if not candidates:
        return []

    fallback_diagnostics = {
        "registry_neighbors_with_first_party_website_count": int(
            latest_discovery_job_result.get("registry_neighbors_with_first_party_website_count", 0)
        ),
        "registry_neighbors_dropped_missing_official_website_count": int(
            latest_discovery_job_result.get("registry_neighbors_dropped_missing_official_website_count", 0)
        ),
        "registry_origin_screening_counts": (
            latest_discovery_job_result.get("registry_origin_screening_counts")
            if isinstance(latest_discovery_job_result.get("registry_origin_screening_counts"), dict)
            else {}
        ),
        "first_party_hint_urls_used_count": int(
            latest_discovery_job_result.get("first_party_hint_urls_used_count", 0)
        ),
        "first_party_hint_pages_crawled_total": int(
            latest_discovery_job_result.get("first_party_hint_pages_crawled_total", 0)
        ),
    }

    company_ids = [row.company_id for row in candidates if row.company_id]
    entity_ids = [row.candidate_entity_id for row in candidates if row.candidate_entity_id]

    company_map: Dict[int, Company] = {}
    if company_ids:
        company_result = await db.execute(select(Company).where(Company.id.in_(company_ids)))
        company_map = {row.id: row for row in company_result.scalars().all()}

    entity_map: Dict[int, CandidateEntity] = {}
    origins_by_entity: Dict[int, List[CandidateOriginEdge]] = {}
    if entity_ids:
        entity_result = await db.execute(select(CandidateEntity).where(CandidateEntity.id.in_(entity_ids)))
        entities = entity_result.scalars().all()
        entity_map = {row.id: row for row in entities}
        origin_result = await db.execute(
            select(CandidateOriginEdge).where(CandidateOriginEdge.entity_id.in_(entity_ids))
        )
        for origin in origin_result.scalars().all():
            origins_by_entity.setdefault(origin.entity_id, []).append(origin)

    class_rank = {
        "good_target": 0,
        "borderline_watchlist": 1,
        "insufficient_evidence": 2,
        "not_good_target": 3,
    }
    candidates.sort(
        key=lambda row: (
            class_rank.get(str(row.decision_classification or "insufficient_evidence"), 99),
            -float(row.total_score or 0.0),
            row.id,
        )
    )

    output: List[UniverseTopCandidateResponse] = []
    for screening in candidates[:limit]:
        company = company_map.get(screening.company_id) if screening.company_id else None
        entity = entity_map.get(screening.candidate_entity_id) if screening.candidate_entity_id else None
        meta = screening.screening_meta_json if isinstance(screening.screening_meta_json, dict) else {}
        diagnostics = _screening_diagnostics_from_meta(meta)
        citation_summary_v1 = _citation_summary_from_meta(meta)
        if (
            diagnostics["registry_neighbors_with_first_party_website_count"] == 0
            and diagnostics["registry_neighbors_dropped_missing_official_website_count"] == 0
            and not diagnostics["registry_origin_screening_counts"]
        ):
            diagnostics = fallback_diagnostics
        discovery_sources: List[str] = []
        if screening.candidate_discovery_url:
            discovery_sources.append(screening.candidate_discovery_url)
        if entity:
            for origin in origins_by_entity.get(entity.id, []):
                if origin.origin_url:
                    discovery_sources.append(origin.origin_url)
        # Deduplicate while preserving order.
        seen_sources: set[str] = set()
        deduped_sources: List[str] = []
        for source in discovery_sources:
            key = str(source or "").strip()
            if not key:
                continue
            lower = key.lower()
            if lower in seen_sources:
                continue
            seen_sources.add(lower)
            deduped_sources.append(key)

        top_claim = screening.top_claim_json if isinstance(screening.top_claim_json, dict) else {}
        if not top_claim.get("source_url") or not top_claim.get("source_tier"):
            top_claim = {}

        output.append(
            UniverseTopCandidateResponse(
                company_id=screening.company_id,
                candidate_entity_id=screening.candidate_entity_id,
                company_name=(company.name if company else screening.candidate_name),
                official_website_url=(
                    screening.candidate_official_website
                    or (company.website if company else None)
                    or (entity.canonical_website if entity else None)
                ),
                discovery_sources=deduped_sources[:12],
                entity_type=(
                    str(entity.entity_type or "company")
                    if entity
                    else str(meta.get("entity_type") or "company")
                ),
                decision_classification=str(screening.decision_classification or "insufficient_evidence"),
                evidence_sufficiency=str(screening.evidence_sufficiency or "insufficient"),
                reason_codes=_reason_codes_payload(screening),
                rationale_summary=screening.rationale_summary,
                top_claim=top_claim,
                citation_summary_v1=citation_summary_v1,
                registry_neighbors_with_first_party_website_count=int(
                    diagnostics["registry_neighbors_with_first_party_website_count"]
                ),
                registry_neighbors_dropped_missing_official_website_count=int(
                    diagnostics["registry_neighbors_dropped_missing_official_website_count"]
                ),
                registry_origin_screening_counts=diagnostics["registry_origin_screening_counts"],
                first_party_hint_urls_used_count=int(diagnostics["first_party_hint_urls_used_count"]),
                first_party_hint_pages_crawled_total=int(diagnostics["first_party_hint_pages_crawled_total"]),
                missing_claim_groups=[str(item) for item in (screening.missing_claim_groups_json or []) if str(item)],
                unresolved_contradictions_count=int(screening.unresolved_contradictions_count or 0),
                ranking_eligible=bool(screening.ranking_eligible),
                run_quality_tier=quality_payload["run_quality_tier"],
                quality_gate_passed=bool(quality_payload["quality_gate_passed"]),
                quality_audit_passed=bool(quality_payload["quality_audit_passed"]),
                degraded_reasons=quality_payload["degraded_reasons"],
            )
        )

    return output


@router.post("/{workspace_id}/companies", response_model=CompanyResponse)
async def create_company(
    workspace_id: int,
    data: CompanyCreate,
    db: AsyncSession = Depends(get_db)
):
    """Manually add a company to the workspace."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    company = Company(
        workspace_id=workspace_id,
        name=data.name,
        website=data.website,
        hq_country=data.hq_country,
        status=CompanyStatus.candidate,
        is_manual=True
    )
    db.add(company)
    await db.commit()
    await db.refresh(company)
    
    return await _company_response_from_row(db, company)


@router.patch("/{workspace_id}/companies/{company_id}", response_model=CompanyResponse)
async def update_company(
    workspace_id: int,
    company_id: int,
    data: CompanyUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update company details or status (keep/remove)."""
    result = await db.execute(
        select(Company).where(
            Company.id == company_id,
            Company.workspace_id == workspace_id
        )
    )
    company = result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    
    if data.name is not None:
        company.name = data.name
    if data.website is not None:
        company.website = data.website
    if data.hq_country is not None:
        company.hq_country = data.hq_country
    if data.operating_countries is not None:
        company.operating_countries = data.operating_countries
    if data.tags_custom is not None:
        company.tags_custom = data.tags_custom
    if data.status is not None:
        company.status = CompanyStatus(data.status)
    
    company.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(company)
    
    return await _company_response_from_row(db, company)


# ============================================================================
# Enrichment
# ============================================================================

@router.post("/{workspace_id}/companies:enrich", response_model=List[JobResponse])
async def enrich_companies_batch(
    workspace_id: int,
    data: EnrichBatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Batch enrich multiple companies."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    jobs = []
    for company_id in data.company_ids:
        # Verify company exists
        company_result = await db.execute(
            select(Company).where(Company.id == company_id, Company.workspace_id == workspace_id)
        )
        company = company_result.scalar_one_or_none()
        if not company:
            continue
        
        for job_type_str in data.job_types:
            job_type = JobType(job_type_str)
            job = Job(
                workspace_id=workspace_id,
                company_id=company_id,
                job_type=job_type,
                state=JobState.queued,
                provider=JobProvider.gemini_flash
            )
            db.add(job)
            await db.flush()
            jobs.append(job)
    
    await db.commit()
    
    # Trigger async tasks
    from app.workers.workspace_tasks import run_enrich_company
    for job in jobs:
        task_result = run_enrich_company.delay(job.id)
        job.interaction_id = str(task_result.id)
    await db.commit()
    
    return [
        JobResponse(
            id=job.id,
            workspace_id=job.workspace_id,
            company_id=job.company_id,
            job_type=job.job_type.value,
            state=job.state.value,
            provider=job.provider.value,
            progress=job.progress,
            progress_message=job.progress_message,
            result_json=job.result_json,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at
        )
        for job in jobs
    ]


@router.get("/{workspace_id}/companies/{company_id}/dossier", response_model=Optional[CompanyDossierResponse])
async def get_company_dossier(
    workspace_id: int,
    company_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get the latest dossier for a company."""
    result = await db.execute(
        select(CompanyDossier)
        .where(CompanyDossier.company_id == company_id)
        .order_by(CompanyDossier.version.desc())
        .limit(1)
    )
    dossier = result.scalar_one_or_none()
    
    if not dossier:
        return None
    
    return CompanyDossierResponse(
        id=dossier.id,
        company_id=dossier.company_id,
        dossier_json=dossier.dossier_json,
        version=dossier.version,
        created_at=dossier.created_at
    )


# ============================================================================
# Static Reports (Snapshot-based)
# ============================================================================

@router.post("/{workspace_id}/reports:generate", response_model=JobResponse)
async def generate_report_snapshot(
    workspace_id: int,
    data: ReportGenerateRequest = ReportGenerateRequest(),
    db: AsyncSession = Depends(get_db),
):
    """Generate an immutable static report snapshot for the workspace."""
    workspace_result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    job = Job(
        workspace_id=workspace_id,
        job_type=JobType.generate_report_snapshot,
        state=JobState.queued,
        provider=JobProvider.crawler,
        result_json={
            "filters": {
                "name": data.name,
                "include_unknown_size": data.include_unknown_size,
                "include_outside_sme": data.include_outside_sme,
            }
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from app.workers.workspace_tasks import generate_static_report

    task_result = generate_static_report.delay(
        job.id,
        {
            "name": data.name,
            "include_unknown_size": data.include_unknown_size,
            "include_outside_sme": data.include_outside_sme,
        },
    )
    job.interaction_id = str(task_result.id)
    await db.commit()
    await db.refresh(job)
    return _to_job_response(job)


@router.get("/{workspace_id}/reports", response_model=List[ReportSnapshotResponse])
async def list_report_snapshots(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ReportSnapshot)
        .where(ReportSnapshot.workspace_id == workspace_id)
        .order_by(ReportSnapshot.generated_at.desc())
    )
    snapshots = result.scalars().all()

    responses: List[ReportSnapshotResponse] = []
    for snapshot in snapshots:
        count_result = await db.execute(
            select(func.count(ReportSnapshotItem.id)).where(
                ReportSnapshotItem.report_id == snapshot.id
            )
        )
        responses.append(
            ReportSnapshotResponse(
                id=snapshot.id,
                workspace_id=snapshot.workspace_id,
                name=snapshot.name,
                status=snapshot.status,
                generated_at=snapshot.generated_at,
                filters_json=snapshot.filters_json or {},
                coverage_json=snapshot.coverage_json or {},
                item_count=count_result.scalar() or 0,
            )
        )
    return responses


@router.get("/{workspace_id}/reports/{report_id}", response_model=ReportSnapshotResponse)
async def get_report_snapshot(
    workspace_id: int,
    report_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ReportSnapshot).where(
            ReportSnapshot.id == report_id,
            ReportSnapshot.workspace_id == workspace_id,
        )
    )
    snapshot = result.scalar_one_or_none()
    if not snapshot:
        raise HTTPException(status_code=404, detail="Report snapshot not found")

    count_result = await db.execute(
        select(func.count(ReportSnapshotItem.id)).where(
            ReportSnapshotItem.report_id == snapshot.id
        )
    )
    return ReportSnapshotResponse(
        id=snapshot.id,
        workspace_id=snapshot.workspace_id,
        name=snapshot.name,
        status=snapshot.status,
        generated_at=snapshot.generated_at,
        filters_json=snapshot.filters_json or {},
        coverage_json=snapshot.coverage_json or {},
        item_count=count_result.scalar() or 0,
    )


@router.get("/{workspace_id}/reports/{report_id}/cards", response_model=List[ReportCard])
async def list_report_cards(
    workspace_id: int,
    report_id: int,
    size_bucket: Optional[str] = Query(
        None,
        description="Optional bucket filter: sme_in_range|unknown|outside_sme_range",
    ),
    include_outside_range: bool = Query(
        False,
        description="Include outside_sme_range cards in unfiltered responses.",
    ),
    db: AsyncSession = Depends(get_db),
):
    snapshot_result = await db.execute(
        select(ReportSnapshot).where(
            ReportSnapshot.id == report_id,
            ReportSnapshot.workspace_id == workspace_id,
        )
    )
    snapshot = snapshot_result.scalar_one_or_none()
    if not snapshot:
        raise HTTPException(status_code=404, detail="Report snapshot not found")

    items_result = await db.execute(
        select(ReportSnapshotItem)
        .where(ReportSnapshotItem.report_id == report_id)
        .order_by(ReportSnapshotItem.compete_score.desc())
    )
    items = items_result.scalars().all()

    cards: List[ReportCard] = []
    for item in items:
        company_result = await db.execute(
            select(Company).where(Company.id == item.company_id, Company.workspace_id == workspace_id)
        )
        company = company_result.scalar_one_or_none()
        if not company:
            continue

        dossier_result = await db.execute(
            select(CompanyDossier)
            .where(CompanyDossier.company_id == company.id)
            .order_by(CompanyDossier.version.desc())
            .limit(1)
        )
        dossier = dossier_result.scalar_one_or_none()
        dossier_json = dossier.dossier_json if dossier else {}

        evidence_result = await db.execute(
            select(SourceEvidence).where(SourceEvidence.company_id == company.id)
        )
        evidence_items = evidence_result.scalars().all()
        evidence_by_url = {e.source_url: e for e in evidence_items if e.source_url}
        evidence_by_id = {e.id: e for e in evidence_items}
        fallback_capabilities = [
            tag.split(":", 1)[1].strip()
            for tag in (company.tags_custom or [])
            if isinstance(tag, str) and tag.startswith("capability:")
        ]
        fallback_evidence_urls = [
            evidence.source_url
            for evidence in evidence_items
            if evidence.source_url and is_trusted_source_url(evidence.source_url)
        ]

        workflow_profile = _collect_workflow_profile(
            dossier_json,
            evidence_by_url=evidence_by_url,
            fallback_capabilities=fallback_capabilities,
            fallback_evidence_urls=fallback_evidence_urls,
        )
        customer_profile = _collect_customer_profile(
            dossier_json,
            evidence_by_url=evidence_by_url,
            why_relevant=company.why_relevant or [],
        )
        business_model_profile = _collect_business_model_profile(
            dossier_json,
            evidence_by_url=evidence_by_url,
        )
        ownership_profile = _collect_bucket_claims(
            dossier_json.get("ownership"),
            evidence_by_url=evidence_by_url,
        )
        transaction_profile = _collect_bucket_claims(
            dossier_json.get("transaction_feasibility"),
            evidence_by_url=evidence_by_url,
        )

        facts_result = await db.execute(
            select(CompanyFact).where(CompanyFact.company_id == company.id)
        )
        facts = facts_result.scalars().all()
        claims_result = await db.execute(
            select(CompanyClaim).where(CompanyClaim.company_id == company.id)
        )
        claims = claims_result.scalars().all()
        screening = await _latest_screening_for_company(db, workspace_id, company.id)

        estimate = item.lens_breakdown_json.get("size_estimate")
        if estimate is None:
            estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts,
                evidence_items=evidence_items,
                tags_custom=company.tags_custom or [],
                why_relevant=company.why_relevant or [],
            )
        size_range_low, size_range_high = _size_range_from_claims(estimate, facts, claims)
        bucket = item.lens_breakdown_json.get("size_bucket") or classify_size_bucket(estimate)
        if size_bucket and bucket != size_bucket:
            continue
        if size_bucket is None and bucket == "outside_sme_range" and not include_outside_range:
            continue

        filing_metrics: Dict[str, SourcedValue] = {}
        for fact in facts:
            source_evidence = evidence_by_id.get(fact.source_evidence_id)
            if not source_evidence and fact.source_evidence_id:
                source_result = await db.execute(
                    select(SourceEvidence).where(SourceEvidence.id == fact.source_evidence_id)
                )
                source_evidence = source_result.scalar_one_or_none()
            if not source_evidence or not source_evidence.source_url:
                continue
            if not is_trusted_source_url(source_evidence.source_url):
                continue
            filing_metrics[fact.fact_key] = SourcedValue(
                value=fact.fact_value,
                unit=fact.fact_unit,
                period=fact.period,
                confidence=_pick_fact_confidence(fact.confidence),
                source=SourcePill(
                    label=source_label_for_url(source_evidence.source_url),
                    url=source_evidence.source_url,
                    document_id=str(source_evidence.id),
                    captured_at=source_evidence.captured_at.isoformat() if source_evidence.captured_at else None,
                ),
            )

        for metric_name, metric_payload in (dossier_json.get("kpis") or {}).items():
            if metric_name in filing_metrics or not isinstance(metric_payload, dict):
                continue
            source_url = metric_payload.get("evidence_url")
            value = str(metric_payload.get("value") or "").strip()
            if not source_url or not value or not is_trusted_source_url(source_url):
                continue
            filing_metrics[metric_name] = SourcedValue(
                value=value,
                unit=(str(metric_payload.get("unit") or "").strip() or None),
                period=(str(metric_payload.get("period") or "").strip() or None),
                confidence=_pick_fact_confidence(metric_payload.get("confidence")),
                source=SourcePill(
                    label=source_label_for_url(source_url),
                    url=source_url,
                    captured_at=None,
                ),
            )

        source_pills: List[SourcePill] = []
        for claim in workflow_profile + customer_profile + business_model_profile + ownership_profile + transaction_profile:
            if claim.source:
                source_pills.append(claim.source)
        for metric in filing_metrics.values():
            source_pills.append(metric.source)
        for claim in claims:
            if not claim.source_url or not is_trusted_source_url(claim.source_url):
                continue
            source_pills.append(
                SourcePill(
                    label=source_label_for_url(claim.source_url),
                    url=claim.source_url,
                    document_id=str(claim.id),
                    captured_at=claim.created_at.isoformat() if claim.created_at else None,
                )
            )

        legal_status = None
        for tag in company.tags_custom or []:
            if isinstance(tag, str) and tag.startswith("legal_status:"):
                legal_status = tag.split(":", 1)[1].strip()
                break

        reason_codes_payload = item.reason_codes_json if isinstance(item.reason_codes_json, dict) else {}
        if not reason_codes_payload and screening:
            reason_codes_payload = _reason_codes_payload(screening)
        reason_code_list: List[str] = []
        for key in ["positive", "caution", "reject"]:
            reason_code_list.extend([str(code) for code in (reason_codes_payload.get(key) or []) if str(code)])

        tier_counts: Dict[str, int] = {}
        fresh_count = 0
        for evidence in evidence_items:
            tier = str(evidence.source_tier or "tier3_third_party")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            if is_fresh(evidence.valid_through):
                fresh_count += 1
        evidence_quality_summary = {
            "tier_counts": tier_counts,
            "freshness_ratio": round(fresh_count / max(1, len(evidence_items)), 4),
            "total_evidence": len(evidence_items),
        }
        known_unknowns = []
        if screening:
            known_unknowns = [f"Missing claim group: {group}" for group in (screening.missing_claim_groups_json or [])]

        card = ReportCard(
            company_id=company.id,
            name=company.name,
            website=company.website,
            hq_country=company.hq_country,
            legal_status=legal_status,
            size_bucket=bucket,
            size_estimate=estimate,
            size_range_low=size_range_low,
            size_range_high=size_range_high,
            fit_score=item.compete_score,
            evidence_score=item.complement_score,
            workflow_profile=workflow_profile,
            customer_profile=customer_profile,
            business_model_profile=business_model_profile,
            ownership_profile=ownership_profile,
            transaction_profile=transaction_profile,
            filing_metrics=filing_metrics,
            source_pills=_dedupe_source_pills(source_pills),
            coverage_note=_coverage_note_for_country(company.hq_country),
            next_validation_questions=[
                "Can we verify customer concentration from at least two independent sources?",
                "What ownership and control signals still need confirmation?",
                "Do filing and commercial signals agree on company scale?",
            ],
            decision_classification=(
                item.decision_classification
                or (screening.decision_classification if screening else None)
            ),
            reason_highlights=[reason_text(code) for code in reason_code_list[:6]],
            evidence_quality_summary=(
                item.evidence_summary_json if isinstance(item.evidence_summary_json, dict) and item.evidence_summary_json else evidence_quality_summary
            ),
            known_unknowns=known_unknowns[:8],
        )
        cards.append(card)

    return cards


@router.get("/{workspace_id}/reports/{report_id}/export")
async def export_report_snapshot(
    workspace_id: int,
    report_id: int,
    format: str = Query("default", pattern="^(default|rich_json)$"),
    db: AsyncSession = Depends(get_db),
):
    """Return report export payload (no arbitrary file write)."""
    snapshot = await get_report_snapshot(workspace_id, report_id, db)
    cards = await list_report_cards(workspace_id, report_id, None, True, db)
    compete = await get_report_lens(workspace_id, report_id, "compete", db)
    complement = await get_report_lens(workspace_id, report_id, "complement", db)

    if format != "rich_json":
        return {
            "report": snapshot.model_dump(),
            "cards": [card.model_dump() for card in cards],
            "lenses": {
                "compete": compete.model_dump(),
                "complement": complement.model_dump(),
            },
            "exported_at": datetime.utcnow().isoformat(),
        }

    report_items_result = await db.execute(
        select(ReportSnapshotItem).where(ReportSnapshotItem.report_id == report_id)
    )
    report_items = report_items_result.scalars().all()
    item_by_company: Dict[int, ReportSnapshotItem] = {item.company_id: item for item in report_items}

    screenings_result = await db.execute(
        select(CompanyScreening)
        .where(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(2000)
    )
    screenings_all = screenings_result.scalars().all()

    screening_run_id = None
    if screenings_all:
        screening_run_id = (screenings_all[0].screening_meta_json or {}).get("screening_run_id")
    screenings = screenings_all
    if screening_run_id:
        screenings = [
            screening
            for screening in screenings_all
            if (screening.screening_meta_json or {}).get("screening_run_id") == screening_run_id
        ]

    screening_ids = [row.id for row in screenings]
    claims: List[CompanyClaim] = []
    if screening_ids:
        claims_result = await db.execute(
            select(CompanyClaim).where(CompanyClaim.company_screening_id.in_(screening_ids))
        )
        claims = claims_result.scalars().all()
    claims_by_screening: Dict[int, List[CompanyClaim]] = {}
    for claim in claims:
        if claim.company_screening_id:
            claims_by_screening.setdefault(claim.company_screening_id, []).append(claim)

    entity_ids = [row.candidate_entity_id for row in screenings if row.candidate_entity_id]
    entity_map: Dict[int, CandidateEntity] = {}
    aliases_by_entity: Dict[int, List[CandidateEntityAlias]] = {}
    origins_by_entity: Dict[int, List[CandidateOriginEdge]] = {}
    if entity_ids:
        entity_result = await db.execute(
            select(CandidateEntity).where(CandidateEntity.id.in_(entity_ids))
        )
        entities = entity_result.scalars().all()
        entity_map = {entity.id: entity for entity in entities}

        alias_result = await db.execute(
            select(CandidateEntityAlias).where(CandidateEntityAlias.entity_id.in_(entity_ids))
        )
        for alias in alias_result.scalars().all():
            aliases_by_entity.setdefault(alias.entity_id, []).append(alias)

        origin_result = await db.execute(
            select(CandidateOriginEdge).where(CandidateOriginEdge.entity_id.in_(entity_ids))
        )
        for origin in origin_result.scalars().all():
            origins_by_entity.setdefault(origin.entity_id, []).append(origin)

    company_ids = [row.company_id for row in screenings if row.company_id]
    company_map: Dict[int, Company] = {}
    dossier_map: Dict[int, CompanyDossier] = {}
    facts_by_company: Dict[int, List[CompanyFact]] = {}
    evidence_by_company: Dict[int, List[SourceEvidence]] = {}
    if company_ids:
        company_result = await db.execute(
            select(Company).where(Company.id.in_(company_ids), Company.workspace_id == workspace_id)
        )
        companies = company_result.scalars().all()
        company_map = {company.id: company for company in companies}

        for company_id in company_ids:
            dossier_result = await db.execute(
                select(CompanyDossier)
                .where(CompanyDossier.company_id == company_id)
                .order_by(CompanyDossier.version.desc())
                .limit(1)
            )
            dossier = dossier_result.scalar_one_or_none()
            if dossier:
                dossier_map[company_id] = dossier

        facts_result = await db.execute(
            select(CompanyFact).where(CompanyFact.company_id.in_(company_ids))
        )
        for fact in facts_result.scalars().all():
            facts_by_company.setdefault(fact.company_id, []).append(fact)

        evidence_result = await db.execute(
            select(SourceEvidence).where(SourceEvidence.company_id.in_(company_ids))
        )
        for evidence in evidence_result.scalars().all():
            evidence_by_company.setdefault(evidence.company_id, []).append(evidence)

    source_runs_result = await db.execute(
        select(ComparatorSourceRun)
        .where(ComparatorSourceRun.workspace_id == workspace_id)
        .order_by(ComparatorSourceRun.captured_at.desc())
        .limit(10)
    )
    source_runs = source_runs_result.scalars().all()
    source_run_ids = [run.id for run in source_runs]
    mentions: List[CompanyMention] = []
    if source_run_ids:
        mentions_result = await db.execute(
            select(CompanyMention).where(CompanyMention.source_run_id.in_(source_run_ids))
        )
        mentions = mentions_result.scalars().all()

    category_counts: Dict[str, int] = {}
    peer_groups: Dict[str, List[str]] = {}
    for mention in mentions:
        tags = mention.category_tags or []
        primary_tag = str(tags[0]) if tags else "unclassified"
        category_counts[primary_tag] = category_counts.get(primary_tag, 0) + 1
        peer_groups.setdefault(primary_tag, []).append(mention.company_name)

    i_cp_terms: Dict[str, int] = {}
    for claim in claims:
        if claim.dimension not in {"icp", "target_customer", "customers", "customer"}:
            continue
        text = (claim.claim_text or "").lower()
        for token in ["asset manager", "wealth manager", "private bank", "fund admin", "insurer", "bank"]:
            if token in text:
                i_cp_terms[token] = i_cp_terms.get(token, 0) + 1

    companies: List[Dict[str, Any]] = []
    reject_reason_counts: Dict[str, int] = {}
    source_type_distribution: Dict[str, int] = {}
    for claim in claims:
        source_key = claim.source_type or "unknown"
        source_type_distribution[source_key] = source_type_distribution.get(source_key, 0) + 1

    latest_discovery_job_result: Dict[str, Any] = {}
    discovery_job_result = await db.execute(
        select(Job)
        .where(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.discovery_universe,
        )
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    latest_discovery_job = discovery_job_result.scalar_one_or_none()
    if latest_discovery_job and isinstance(latest_discovery_job.result_json, dict):
        latest_discovery_job_result = latest_discovery_job.result_json

    for screening in screenings:
        company = company_map.get(screening.company_id) if screening.company_id else None
        candidate_entity = entity_map.get(screening.candidate_entity_id) if screening.candidate_entity_id else None
        company_claims = claims_by_screening.get(screening.id, [])
        report_item = item_by_company.get(company.id) if company else None
        dossier_json = {}
        if company and company.id in dossier_map:
            dossier_json = dossier_map[company.id].dossier_json or {}
        modules = modules_with_evidence(dossier_json)
        customers, integrations = extract_customers_and_integrations(dossier_json)
        facts = facts_by_company.get(company.id, []) if company else []

        size_estimate = None
        if report_item:
            size_estimate = report_item.lens_breakdown_json.get("size_estimate")
        if size_estimate is None and company:
            size_estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts,
                evidence_items=evidence_by_company.get(company.id, []),
                tags_custom=company.tags_custom or [],
                why_relevant=company.why_relevant or [],
            )
        size_low, size_high = _size_range_from_claims(size_estimate, facts, company_claims)
        size_confidence = "high" if (size_low is not None and size_high is not None and size_low == size_high) else "medium"

        component_scores = screening.component_scores_json or {}
        total_score = float(screening.total_score or 0.0)
        penalties = screening.penalties_json or []
        reject_reasons = [
            reason
            for reason in (screening.reject_reasons_json or [])
            if str(reason) != "score_below_threshold"
        ]
        for reason in reject_reasons:
            key = str(reason).strip() or "unknown_reason"
            reject_reason_counts[key] = reject_reason_counts.get(key, 0) + 1

        institutional_strength = float(component_scores.get("institutional_icp_fit", 0.0))
        product_depth = float(component_scores.get("platform_product_depth", 0.0))
        moat_strength = float(component_scores.get("defensibility_moat", 0.0))
        gtm_strength = float(component_scores.get("enterprise_gtm", 0.0))

        if report_item:
            compete_score = float(report_item.compete_score)
            complement_score = float(report_item.complement_score)
        else:
            compete_score = round(100.0 * ((institutional_strength * 0.5) + (product_depth * 0.5)), 2)
            complement_score = round(100.0 * ((moat_strength * 0.5) + (gtm_strength * 0.5)), 2)

        if compete_score >= 60.0 and complement_score >= 60.0:
            fit_classification = "both"
        elif compete_score >= complement_score:
            fit_classification = "competitor"
        else:
            fit_classification = "adjacent"

        source_pills: List[Dict[str, Any]] = []
        for claim in company_claims:
            if not claim.source_url or not is_trusted_source_url(claim.source_url):
                continue
            source_pills.append(
                {
                    "label": source_label_for_url(claim.source_url),
                    "url": claim.source_url,
                    "source_type": claim.source_type,
                    "captured_at": claim.created_at.isoformat() if claim.created_at else None,
                }
            )
        seen_source_urls: set[str] = set()
        deduped_pills: List[Dict[str, Any]] = []
        for pill in source_pills:
            key = str(pill.get("url") or "").strip().lower()
            if not key or key in seen_source_urls:
                continue
            seen_source_urls.add(key)
            deduped_pills.append(pill)

        icp_claims = [
            claim for claim in company_claims
            if (claim.dimension or "").lower() in {"icp", "target_customer", "customers", "customer"}
        ]
        product_claims = [
            claim for claim in company_claims
            if (claim.dimension or "").lower() in {"capability", "product", "services", "evidence", "directory_context"}
        ]

        target_segments = []
        all_text = " ".join((claim.claim_text or "") for claim in company_claims).lower()
        segment_tokens = {
            "asset manager": "asset manager",
            "wealth manager": "wealth manager",
            "private bank": "private bank",
            "fund admin": "fund admin",
            "insurer": "insurer",
            "bank": "bank",
        }
        for token, label in segment_tokens.items():
            if token in all_text:
                target_segments.append(label)
        target_segments = list(dict.fromkeys(target_segments))

        fit_summary = (
            f"Strong institutional fit with score {total_score:.1f}/100 and "
            f"{len(modules) if modules else len(product_claims)} capability signals."
        )
        principal_risk = (
            str(reject_reasons[0]) if reject_reasons else
            ("evidence density is moderate; validate customer depth before IC.")
        )
        next_question = (
            "Which two named institutional customers can be independently confirmed?"
            if not customers
            else "What is customer concentration risk among top logos?"
        )

        company_name = company.name if company else screening.candidate_name
        website = company.website if company else screening.candidate_website
        country = company.hq_country if company else (screening.screening_meta_json or {}).get("candidate_hq_country")
        screening_meta = screening.screening_meta_json or {}
        source_summary = screening.source_summary_json or {}
        first_party_enrichment = screening_meta.get("first_party_enrichment") if isinstance(screening_meta.get("first_party_enrichment"), dict) else {}
        identity_meta = screening_meta.get("identity") if isinstance(screening_meta.get("identity"), dict) else {}
        merge_rationale = []
        entity_aliases_payload: List[Dict[str, Any]] = []
        entity_origins_payload: List[Dict[str, Any]] = []
        registry_ids_payload: List[str] = []
        registry_identity_payload: Dict[str, Any] = {}
        industry_signature_payload: Dict[str, Any] = {}
        expansion_provenance_payload: List[Dict[str, Any]] = []
        if candidate_entity:
            merge_rationale = (candidate_entity.metadata_json or {}).get("merge_rationale") or []
            registry_identity_payload = (candidate_entity.metadata_json or {}).get("registry_identity") or {}
            industry_signature_payload = (candidate_entity.metadata_json or {}).get("industry_signature") or {}
            alias_rows = aliases_by_entity.get(candidate_entity.id, [])
            entity_aliases_payload = [
                {
                    "alias_name": alias.alias_name,
                    "alias_website": alias.alias_website,
                    "merge_confidence": alias.merge_confidence,
                    "merge_reason": alias.merge_reason,
                }
                for alias in alias_rows
            ]
            origin_rows = origins_by_entity.get(candidate_entity.id, [])
            entity_origins_payload = [
                {
                    "origin_type": origin.origin_type,
                    "origin_url": origin.origin_url,
                    "source_run_id": origin.source_run_id,
                    "metadata": origin.metadata_json or {},
                }
                for origin in origin_rows
            ]
            expansion_provenance_payload = [
                {
                    "seed_entity": (origin.metadata_json or {}).get("seed_entity_name"),
                    "query_type": (origin.metadata_json or {}).get("query_type"),
                    "country": (origin.metadata_json or {}).get("country"),
                    "query": (origin.metadata_json or {}).get("query"),
                }
                for origin in origin_rows
                if isinstance(origin.metadata_json, dict) and (origin.metadata_json or {}).get("query_type")
            ]
            if candidate_entity.registry_id:
                registry_ids_payload.append(candidate_entity.registry_id)

        legal_hint = None
        if company:
            for tag in company.tags_custom or []:
                if isinstance(tag, str) and tag.startswith("legal_status:"):
                    legal_hint = tag.split(":", 1)[1].strip()
                    break
        reason_codes_payload = _reason_codes_payload(screening)
        discovery_sources_raw = [str(screening.candidate_discovery_url or "")] + [
            str(origin.get("origin_url") or "") for origin in entity_origins_payload
        ]
        discovery_sources: List[str] = []
        seen_discovery_sources: set[str] = set()
        for source in discovery_sources_raw:
            normalized_source = source.strip()
            if not normalized_source:
                continue
            key = normalized_source.lower()
            if key in seen_discovery_sources:
                continue
            seen_discovery_sources.add(key)
            discovery_sources.append(normalized_source)
        entity_type = (
            str(candidate_entity.entity_type or "company")
            if candidate_entity
            else str(screening_meta.get("entity_type") or "company")
        )
        solutions_payload = (
            candidate_entity.solutions_json
            if candidate_entity and isinstance(candidate_entity.solutions_json, list)
            else (
                screening_meta.get("solutions")
                if isinstance(screening_meta.get("solutions"), list)
                else []
            )
        )
        top_claim_payload = screening.top_claim_json if isinstance(screening.top_claim_json, dict) else {}
        if not top_claim_payload.get("source_url") or not top_claim_payload.get("source_tier"):
            top_claim_payload = {}

        companies.append(
            {
                "identity": {
                    "name": company_name,
                    "website": website,
                    "official_website": (
                        screening.candidate_official_website
                        or identity_meta.get("official_website")
                        or website
                    ),
                    "discovery_sources": discovery_sources,
                    "input_website": identity_meta.get("input_website"),
                    "identity_confidence": identity_meta.get("identity_confidence"),
                    "identity_sources": identity_meta.get("identity_sources") or [],
                    "country": country,
                    "legal_entity_hints": legal_hint,
                    "entity_type": entity_type,
                    "solutions": solutions_payload,
                    "entity_id": candidate_entity.id if candidate_entity else None,
                },
                "screening": {
                    "status": screening.screening_status,
                    "total_score": total_score,
                    "component_scores": component_scores,
                    "penalties": penalties,
                    "reason_codes": reason_codes_payload,
                    "reject_reasons": reject_reasons,
                    "evidence_mix": source_summary.get("source_type_counts") or {},
                    "ranking_eligible": bool(screening.ranking_eligible),
                },
                "size": {
                    "employee_estimate": {
                        "point": size_estimate if (size_low is None or size_high is None or size_low == size_high) else None,
                        "range": [size_low, size_high] if (size_low is not None and size_high is not None and size_low != size_high) else None,
                        "confidence": size_confidence,
                    },
                    "supporting_claims": [
                        {
                            "text": claim.claim_text,
                            "source_url": claim.source_url,
                            "source_type": claim.source_type,
                        }
                        for claim in company_claims
                        if claim.claim_key == "employees"
                    ][:6],
                },
                "icp": {
                    "target_institution_types": target_segments,
                    "segment_focus": target_segments[0] if target_segments else "unknown",
                    "fit_strength": "high" if institutional_strength >= 0.7 else ("medium" if institutional_strength >= 0.4 else "low"),
                    "evidence": [
                        {
                            "text": claim.claim_text[:240],
                            "source_url": claim.source_url,
                        }
                        for claim in icp_claims[:6]
                    ],
                },
                "platform_product_services": {
                    "modules": [
                        {
                            "name": module.get("name"),
                            "brick_id": module.get("brick_id"),
                            "has_evidence": bool(module.get("has_evidence")),
                        }
                        for module in modules
                    ],
                    "services_indicators": [
                        claim.claim_text[:220]
                        for claim in product_claims
                        if any(token in claim.claim_text.lower() for token in ("implementation", "integration", "migration", "consulting"))
                    ][:8],
                    "positioning_signals": [
                        claim.claim_text[:220]
                        for claim in product_claims[:8]
                    ],
                },
                "customers": {
                    "named_logos": [
                        customer.get("name")
                        for customer in customers
                        if customer.get("name")
                    ][:12],
                    "customer_type": target_segments[0] if target_segments else "unknown",
                    "credibility_class": "high" if len(customers) >= 3 else ("medium" if len(customers) >= 1 else "low"),
                    "evidence": [
                        {
                            "text": claim.claim_text[:240],
                            "source_url": claim.source_url,
                        }
                        for claim in company_claims
                        if (claim.dimension or "").lower() in {"customer", "customers", "case_study"}
                    ][:8],
                },
                "fit_classification": {
                    "type": fit_classification,
                    "compete_score": compete_score,
                    "complement_score": complement_score,
                    "rationale": (
                        "High brick overlap and institutional ICP alignment."
                        if fit_classification == "competitor"
                        else "Adjacent capability stack with complementary integration surface."
                    ),
                },
                "buy_side_view": {
                    "summary": fit_summary,
                    "principal_risk": principal_risk,
                    "next_diligence_question": next_question,
                },
                "top_claim": top_claim_payload,
                "first_party_enrichment": {
                    "method": first_party_enrichment.get("method"),
                    "tier": first_party_enrichment.get("tier"),
                    "pages_crawled": int(first_party_enrichment.get("pages_crawled") or 0),
                    "signals_extracted": int(first_party_enrichment.get("signals_extracted") or 0),
                    "customer_evidence_count": int(first_party_enrichment.get("customer_evidence_count") or 0),
                    "page_types": first_party_enrichment.get("page_types") if isinstance(first_party_enrichment.get("page_types"), dict) else {},
                    "error": first_party_enrichment.get("error"),
                },
                "entity_id": candidate_entity.id if candidate_entity else None,
                "origins": entity_origins_payload,
                "aliases": entity_aliases_payload,
                "merge_rationale": merge_rationale,
                "registry_ids": registry_ids_payload,
                "registry_identity": registry_identity_payload,
                "industry_signature": industry_signature_payload,
                "expansion_provenance": expansion_provenance_payload,
                "source_pills": deduped_pills,
            }
        )

    companies.sort(
        key=lambda row: (
            0 if bool((row.get("screening") or {}).get("ranking_eligible")) else 1,
            0 if row["screening"]["status"] == "kept" else (1 if row["screening"]["status"] == "review" else 2),
            -float(row["screening"]["total_score"] or 0.0),
        )
    )

    origin_mix_distribution: Dict[str, int] = {}
    for edge_list in origins_by_entity.values():
        for edge in edge_list:
            origin_type = edge.origin_type or "unknown"
            origin_mix_distribution[origin_type] = origin_mix_distribution.get(origin_type, 0) + 1

    diagnostics = {
        "screening_totals": {
            "kept": len([row for row in companies if row["screening"]["status"] == "kept"]),
            "review": len([row for row in companies if row["screening"]["status"] == "review"]),
            "rejected": len([row for row in companies if row["screening"]["status"] == "rejected"]),
        },
        "filter_reason_counts": reject_reason_counts,
        "evidence_coverage_stats": {
            "companies_with_icp_evidence": len([row for row in companies if row["icp"]["evidence"]]),
            "companies_with_product_service_evidence": len([row for row in companies if row["platform_product_services"]["positioning_signals"]]),
            "companies_with_source_pills": len([row for row in companies if row["source_pills"]]),
        },
        "source_quality_distribution": source_type_distribution,
        "origin_mix_distribution": origin_mix_distribution,
        "dedupe_quality_metrics": {
            "entities_with_aliases": len([entity_id for entity_id, aliases in aliases_by_entity.items() if len(aliases) > 1]),
            "total_entities": len(entity_map),
        },
        "registry_expansion_yield": {
            "entities_with_registry_id": len([entity for entity in entity_map.values() if entity.registry_id]),
            "registry_identity_candidates_count": int(latest_discovery_job_result.get("registry_identity_candidates_count", 0)),
            "registry_identity_mapped_count": int(latest_discovery_job_result.get("registry_identity_mapped_count", 0)),
            "registry_queries_by_country": latest_discovery_job_result.get("registry_queries_by_country", {}),
            "registry_raw_hits_by_country": latest_discovery_job_result.get("registry_raw_hits_by_country", {}),
            "registry_neighbors_kept_pre_dedupe": int(latest_discovery_job_result.get("registry_neighbors_kept_pre_dedupe", 0)),
            "registry_neighbors_unique_post_dedupe": int(latest_discovery_job_result.get("registry_neighbors_unique_post_dedupe", 0)),
            "registry_reject_reason_breakdown": latest_discovery_job_result.get("registry_reject_reason_breakdown", {}),
        },
        "first_party_crawl_stats": {
            "attempted_count": int(latest_discovery_job_result.get("first_party_crawl_attempted_count", 0)),
            "success_count": int(latest_discovery_job_result.get("first_party_crawl_success_count", 0)),
            "failed_count": int(latest_discovery_job_result.get("first_party_crawl_failed_count", 0)),
            "deep_count": int(latest_discovery_job_result.get("first_party_crawl_deep_count", 0)),
            "light_count": int(latest_discovery_job_result.get("first_party_crawl_light_count", 0)),
            "fallback_count": int(latest_discovery_job_result.get("first_party_crawl_fallback_count", 0)),
            "pages_total": int(latest_discovery_job_result.get("first_party_crawl_pages_total", 0)),
        },
    }

    return {
        "run": {
            "workspace_id": workspace_id,
            "report_id": report_id,
            "snapshot_name": snapshot.name,
            "generated_at": snapshot.generated_at.isoformat(),
            "exported_at": datetime.utcnow().isoformat(),
            "screening_run_id": screening_run_id,
            "model_metadata": {
                "discovery_provider": "gemini_flash",
                "scoring_model": "evidence_weighted_buy_side_v1",
            },
            "source_coverage": [
                {
                    "source_name": run.source_name,
                    "source_url": run.source_url,
                    "mentions_found": run.mentions_found,
                    "status": run.status,
                    "captured_at": run.captured_at.isoformat() if run.captured_at else None,
                }
                for run in source_runs
            ],
        },
        "market_context": {
            "directory_taxonomy_clusters": [
                {"cluster": cluster, "count": count}
                for cluster, count in sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:20]
            ],
            "comparator_peer_groups": {
                cluster: sorted(list(dict.fromkeys(names)))[:30]
                for cluster, names in peer_groups.items()
            },
            "icp_language_signals": [
                {"term": term, "count": count}
                for term, count in sorted(i_cp_terms.items(), key=lambda item: item[1], reverse=True)
            ],
        },
        "companies": companies,
        "companies_kept": [row for row in companies if row["screening"]["status"] == "kept"],
        "companies_review": [row for row in companies if row["screening"]["status"] == "review"],
        "companies_rejected": [row for row in companies if row["screening"]["status"] == "rejected"],
        "diagnostics": diagnostics,
    }


# ============================================================================
# Decision Catalog / Policy / Diagnostics
# ============================================================================


@router.get("/{workspace_id}/decision-catalog")
async def get_decision_catalog(workspace_id: int, db: AsyncSession = Depends(get_db)):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return get_catalog_payload()


@router.get("/{workspace_id}/evidence-policy")
async def get_evidence_policy(workspace_id: int, db: AsyncSession = Depends(get_db)):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
    return {"workspace_id": workspace_id, "policy": policy}


@router.patch("/{workspace_id}/evidence-policy")
async def update_evidence_policy(
    workspace_id: int,
    payload: EvidencePolicyUpdate,
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    workspace.decision_policy_json = normalize_policy(payload.policy)
    await db.commit()
    return {"workspace_id": workspace_id, "policy": workspace.decision_policy_json}


@router.get("/{workspace_id}/companies/{company_id}/decision", response_model=CompanyDecisionResponse)
async def get_company_decision(
    workspace_id: int,
    company_id: int,
    db: AsyncSession = Depends(get_db),
):
    company_result = await db.execute(
        select(Company).where(Company.id == company_id, Company.workspace_id == workspace_id)
    )
    company = company_result.scalar_one_or_none()
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")

    screening = await _latest_screening_for_company(db, workspace_id, company_id)
    if screening:
        return CompanyDecisionResponse(
            company_id=company_id,
            workspace_id=workspace_id,
            classification=screening.decision_classification or "insufficient_evidence",
            evidence_sufficiency=screening.evidence_sufficiency or "insufficient",
            positive_reason_codes=screening.positive_reason_codes_json or [],
            caution_reason_codes=screening.caution_reason_codes_json or [],
            reject_reason_codes=screening.reject_reason_codes_json or [],
            missing_claim_groups=screening.missing_claim_groups_json or [],
            unresolved_contradictions_count=screening.unresolved_contradictions_count or 0,
            rationale_summary=screening.rationale_summary,
            rationale_markdown=screening.rationale_markdown,
            decision_engine_version=screening.decision_engine_version,
            gating_passed=bool(screening.gating_passed),
            generated_at=datetime.utcnow().isoformat(),
        )

    claims_result = await db.execute(
        select(CompanyClaim).where(
            CompanyClaim.workspace_id == workspace_id,
            CompanyClaim.company_id == company_id,
        )
    )
    claims = claims_result.scalars().all()
    decision = evaluate_decision(
        screening_status="review",
        reject_reasons=[],
        claims=claims,
        component_scores={},
        source_type_counts={},
    )
    return CompanyDecisionResponse(
        company_id=company_id,
        workspace_id=workspace_id,
        classification=decision.classification,
        evidence_sufficiency=decision.evidence_sufficiency,
        positive_reason_codes=decision.positive_reason_codes,
        caution_reason_codes=decision.caution_reason_codes,
        reject_reason_codes=decision.reject_reason_codes,
        missing_claim_groups=decision.missing_claim_groups,
        unresolved_contradictions_count=decision.unresolved_contradictions_count,
        rationale_summary=decision.rationale_summary,
        rationale_markdown=decision.rationale_markdown,
        decision_engine_version=decision.decision_engine_version,
        gating_passed=decision.gating_passed,
        generated_at=datetime.utcnow().isoformat(),
    )


@router.get("/{workspace_id}/diagnostics/decision-quality")
async def get_decision_quality_diagnostics(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    screenings_result = await db.execute(
        select(CompanyScreening)
        .where(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(3000)
    )
    screenings = screenings_result.scalars().all()
    if not screenings:
        feedback_count_result = await db.execute(
            select(func.count(WorkspaceFeedbackEvent.id)).where(
                WorkspaceFeedbackEvent.workspace_id == workspace_id
            )
        )
        feedback_count = int(feedback_count_result.scalar() or 0)
        return {
            "workspace_id": workspace_id,
            "totals": 0,
            "classification_distribution": {},
            "evidence_sufficiency_distribution": {},
            "contradiction_rate": 0.0,
            "evidence_tier_mix": {},
            "freshness_compliance_by_group": {},
            "keep_to_later_reject_rate": 0.0,
            "analyst_override_rate": 0.0,
            "ranking_eligible_count": 0,
            "directory_only_count": 0,
            "solution_entity_count": 0,
            "official_website_resolution_rate": 0.0,
            "feedback_events_count": feedback_count,
            "generated_at": datetime.utcnow().isoformat(),
        }

    company_ids = [row.company_id for row in screenings if row.company_id]
    claims_result = await db.execute(
        select(CompanyClaim).where(CompanyClaim.workspace_id == workspace_id)
    )
    claims = claims_result.scalars().all()

    evidence_result = await db.execute(
        select(SourceEvidence).where(SourceEvidence.workspace_id == workspace_id)
    )
    evidence_rows = evidence_result.scalars().all()

    classification_distribution: Dict[str, int] = {}
    sufficiency_distribution: Dict[str, int] = {}
    contradictions = 0
    for row in screenings:
        cls = str(row.decision_classification or "insufficient_evidence")
        classification_distribution[cls] = classification_distribution.get(cls, 0) + 1
        suff = str(row.evidence_sufficiency or "insufficient")
        sufficiency_distribution[suff] = sufficiency_distribution.get(suff, 0) + 1
        if (row.unresolved_contradictions_count or 0) > 0:
            contradictions += 1

    evidence_tier_mix: Dict[str, int] = {}
    for evidence in evidence_rows:
        key = str(evidence.source_tier or "tier3_third_party")
        evidence_tier_mix[key] = evidence_tier_mix.get(key, 0) + 1

    freshness_total_by_group: Dict[str, int] = {}
    freshness_ok_by_group: Dict[str, int] = {}
    for claim in claims:
        group = str(claim.claim_group or claim_group_for_dimension(claim.dimension, claim.claim_key))
        freshness_total_by_group[group] = freshness_total_by_group.get(group, 0) + 1
        if is_fresh(claim.valid_through):
            freshness_ok_by_group[group] = freshness_ok_by_group.get(group, 0) + 1
    freshness_compliance = {
        group: round(freshness_ok_by_group.get(group, 0) / max(1, total), 4)
        for group, total in freshness_total_by_group.items()
    }

    by_company: Dict[int, List[CompanyScreening]] = {}
    for row in screenings:
        if row.company_id:
            by_company.setdefault(row.company_id, []).append(row)
    downgraded = 0
    for rows in by_company.values():
        statuses = [str(item.decision_classification or "") for item in rows]
        if "good_target" in statuses and "not_good_target" in statuses:
            downgraded += 1
    keep_to_later_reject_rate = round(downgraded / max(1, len(by_company)), 4)
    feedback_count_result = await db.execute(
        select(func.count(WorkspaceFeedbackEvent.id)).where(
            WorkspaceFeedbackEvent.workspace_id == workspace_id
        )
    )
    feedback_count = int(feedback_count_result.scalar() or 0)
    analyst_override_rate = round(feedback_count / max(1, len(by_company)), 4)
    ranking_eligible_count = len([row for row in screenings if bool(row.ranking_eligible)])
    directory_only_count = len(
        [
            row for row in screenings
            if not str(row.candidate_official_website or "").strip()
            or _is_directory_host_url(row.candidate_official_website)
        ]
    )
    solution_entity_count = len(
        [
            row for row in screenings
            if str(
                ((row.screening_meta_json or {}).get("entity_type") if isinstance(row.screening_meta_json, dict) else "")
                or ""
            ).strip().lower() == "solution"
        ]
    )
    official_website_resolution_rate = round(
        (
            len(
                [
                    row for row in screenings
                    if str(row.candidate_official_website or "").strip()
                    and not _is_directory_host_url(row.candidate_official_website)
                ]
            )
            / max(1, len(screenings))
        ),
        4,
    )

    return {
        "workspace_id": workspace_id,
        "totals": len(screenings),
        "classification_distribution": classification_distribution,
        "evidence_sufficiency_distribution": sufficiency_distribution,
        "contradiction_rate": round(contradictions / max(1, len(screenings)), 4),
        "evidence_tier_mix": evidence_tier_mix,
        "freshness_compliance_by_group": freshness_compliance,
        "keep_to_later_reject_rate": keep_to_later_reject_rate,
        "analyst_override_rate": analyst_override_rate,
        "ranking_eligible_count": ranking_eligible_count,
        "directory_only_count": directory_only_count,
        "solution_entity_count": solution_entity_count,
        "official_website_resolution_rate": official_website_resolution_rate,
        "feedback_events_count": feedback_count,
        "generated_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Monitoring / Feedback / Graph / Evaluation
# ============================================================================


@router.post("/{workspace_id}/monitoring:run", response_model=JobResponse)
async def run_monitoring(
    workspace_id: int,
    data: MonitoringRunRequest = MonitoringRunRequest(),
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    job = Job(
        workspace_id=workspace_id,
        job_type=JobType.monitoring_delta,
        state=JobState.queued,
        provider=JobProvider.crawler,
        result_json={
            "max_companies": data.max_companies,
            "stale_only": data.stale_only,
            "classifications": data.classifications,
        },
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    from app.workers.workspace_tasks import run_monitoring_delta

    run_monitoring_delta.delay(job.id)
    return _to_job_response(job)


@router.get("/{workspace_id}/claims-graph", response_model=ClaimsGraphSummaryResponse)
async def get_claims_graph_summary(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    nodes_count = (
        await db.execute(
            select(func.count(ClaimGraphNode.id)).where(ClaimGraphNode.workspace_id == workspace_id)
        )
    ).scalar() or 0
    edges_count = (
        await db.execute(
            select(func.count(ClaimGraphEdge.id)).where(ClaimGraphEdge.workspace_id == workspace_id)
        )
    ).scalar() or 0
    edge_evidence_count = (
        await db.execute(
            select(func.count(ClaimGraphEdgeEvidence.id))
            .join(ClaimGraphEdge, ClaimGraphEdge.id == ClaimGraphEdgeEvidence.edge_id)
            .where(ClaimGraphEdge.workspace_id == workspace_id)
        )
    ).scalar() or 0

    nodes_result = await db.execute(
        select(ClaimGraphNode.node_type).where(ClaimGraphNode.workspace_id == workspace_id)
    )
    node_type_distribution: Dict[str, int] = {}
    for (node_type,) in nodes_result.all():
        key = str(node_type or "unknown")
        node_type_distribution[key] = node_type_distribution.get(key, 0) + 1

    edges_result = await db.execute(
        select(ClaimGraphEdge.relation_type).where(ClaimGraphEdge.workspace_id == workspace_id)
    )
    relation_type_distribution: Dict[str, int] = {}
    for (relation_type,) in edges_result.all():
        key = str(relation_type or "unknown")
        relation_type_distribution[key] = relation_type_distribution.get(key, 0) + 1

    return ClaimsGraphSummaryResponse(
        workspace_id=workspace_id,
        nodes_count=int(nodes_count),
        edges_count=int(edges_count),
        edge_evidence_count=int(edge_evidence_count),
        relation_type_distribution=relation_type_distribution,
        node_type_distribution=node_type_distribution,
        generated_at=datetime.utcnow().isoformat(),
    )


@router.post("/{workspace_id}/claims-graph:refresh")
async def refresh_claims_graph(
    workspace_id: int,
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    from app.workers.workspace_tasks import run_claims_graph_refresh

    task_result = run_claims_graph_refresh.delay(workspace_id)
    return {
        "workspace_id": workspace_id,
        "queued": True,
        "task_id": task_result.id,
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.post("/{workspace_id}/feedback", response_model=WorkspaceFeedbackResponse)
async def create_workspace_feedback(
    workspace_id: int,
    data: WorkspaceFeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    screening: Optional[CompanyScreening] = None
    if data.company_id:
        company_result = await db.execute(
            select(Company).where(Company.id == data.company_id, Company.workspace_id == workspace_id)
        )
        company = company_result.scalar_one_or_none()
        if not company:
            raise HTTPException(status_code=404, detail="Company not found")
    if data.company_screening_id:
        screening_result = await db.execute(
            select(CompanyScreening).where(
                CompanyScreening.id == data.company_screening_id,
                CompanyScreening.workspace_id == workspace_id,
            )
        )
        screening = screening_result.scalar_one_or_none()
    elif data.company_id:
        screening = await _latest_screening_for_company(db, workspace_id, data.company_id)

    feedback_event = WorkspaceFeedbackEvent(
        workspace_id=workspace_id,
        company_id=data.company_id,
        company_screening_id=screening.id if screening else data.company_screening_id,
        feedback_type=data.feedback_type,
        previous_classification=data.previous_classification
        or (screening.decision_classification if screening else None),
        new_classification=data.new_classification,
        reason_codes_json=data.reason_codes,
        comment=data.comment,
        metadata_json=data.metadata,
        created_by=data.created_by,
    )
    db.add(feedback_event)

    if screening and data.new_classification:
        screening.decision_classification = data.new_classification
        if data.reason_codes:
            screening.positive_reason_codes_json = [
                code for code in data.reason_codes if str(code).startswith("POS-")
            ]
            screening.caution_reason_codes_json = [
                code for code in data.reason_codes if str(code).startswith("CAU-")
            ]
            screening.reject_reason_codes_json = [
                code for code in data.reason_codes if str(code).startswith("REJ-")
            ]

    await db.commit()
    await db.refresh(feedback_event)
    return _to_workspace_feedback_response(feedback_event)


@router.get("/{workspace_id}/feedback", response_model=List[WorkspaceFeedbackResponse])
async def list_workspace_feedback(
    workspace_id: int,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    result = await db.execute(
        select(WorkspaceFeedbackEvent)
        .where(WorkspaceFeedbackEvent.workspace_id == workspace_id)
        .order_by(WorkspaceFeedbackEvent.created_at.desc())
        .limit(limit)
    )
    return [_to_workspace_feedback_response(row) for row in result.scalars().all()]


@router.post("/{workspace_id}/evaluations/replay", response_model=EvaluationReplayResponse)
async def replay_workspace_evaluation(
    workspace_id: int,
    payload: EvaluationReplayRequest,
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    run = EvaluationRun(
        workspace_id=workspace_id,
        run_type="gold_set_replay",
        status="completed",
        model_version=payload.model_version,
        metrics_json={},
    )
    db.add(run)
    await db.flush()

    total = 0
    matched = 0
    for sample in payload.samples:
        expected = str(sample.get("expected_classification") or "")
        predicted = str(sample.get("predicted_classification") or "")
        sample_matched = int(bool(expected and predicted and expected == predicted))
        total += 1
        matched += sample_matched
        db.add(
            EvaluationSampleResult(
                run_id=run.id,
                company_id=sample.get("company_id"),
                expected_classification=expected or None,
                predicted_classification=predicted or None,
                matched=sample_matched,
                confidence=float(sample.get("confidence") or 0.0),
                details_json=sample.get("details_json") if isinstance(sample.get("details_json"), dict) else {},
            )
        )

    precision_proxy = round((matched / total) if total else 0.0, 4)
    run.metrics_json = {
        "samples_total": total,
        "samples_matched": matched,
        "precision_proxy": precision_proxy,
    }
    await db.commit()
    await db.refresh(run)

    return EvaluationReplayResponse(
        workspace_id=workspace_id,
        run_id=run.id,
        metrics=run.metrics_json or {},
        created_at=run.created_at.isoformat() if run.created_at else datetime.utcnow().isoformat(),
    )


@router.get("/{workspace_id}/evaluations")
async def list_workspace_evaluations(
    workspace_id: int,
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    result = await db.execute(
        select(EvaluationRun)
        .where(EvaluationRun.workspace_id == workspace_id)
        .order_by(EvaluationRun.created_at.desc())
        .limit(limit)
    )
    runs = result.scalars().all()
    return [
        {
            "id": run.id,
            "run_type": run.run_type,
            "status": run.status,
            "model_version": run.model_version,
            "metrics": run.metrics_json or {},
            "created_at": run.created_at.isoformat() if run.created_at else None,
        }
        for run in runs
    ]


# ============================================================================
# Gates
# ============================================================================

@router.get("/{workspace_id}/gates", response_model=GatesResponse)
async def get_gates(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get gating status for the workspace - what's unlocked."""
    workspace_result = await db.execute(select(Workspace).where(Workspace.id == workspace_id))
    workspace = workspace_result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    policy = normalize_policy(workspace.decision_policy_json or DEFAULT_EVIDENCE_POLICY)
    gate_cfg = policy.get("gate_requirements", {})

    missing_items: Dict[str, List[str]] = {
        "context_pack": [],
        "scope_review": [],
        "universe": [],
        "segmentation": [],
        "enrichment": []
    }
    
    # Check company context
    profile = await _get_company_profile(db, workspace_id)
    context_pack_ready = False
    if profile:
        context_claim_groups_available = set()
        company_context_result = await db.execute(
            select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
        )
        company_context_pack = company_context_result.scalar_one_or_none()
        company_context_payload = (
            _company_context_payload_from_pack(company_context_pack, profile=profile)
            if company_context_pack
            else build_company_context_artifacts(profile)
        )
        has_company = bool(profile.buyer_company_url)
        if not has_company:
            missing_items["context_pack"].append("Add a source company URL")
        else:
            context_claim_groups_available.add("identity_scope")
        sourcing_brief = company_context_payload.get("sourcing_brief") or {}
        if not (
            isinstance(sourcing_brief, dict)
            and (
                str(sourcing_brief.get("source_summary") or "").strip()
                or sourcing_brief.get("customer_nodes")
                or sourcing_brief.get("capability_nodes")
            )
        ):
            missing_items["context_pack"].append("Generate or refresh the sourcing brief")
        else:
            context_claim_groups_available.add("product_depth")
        if profile.comparator_seed_urls:
            context_claim_groups_available.add("vertical_workflow")
        if not (company_context_payload.get("source_documents") or []) and (profile.comparator_seed_urls or profile.supporting_evidence_urls):
            missing_items["context_pack"].append("Review supporting evidence links for the sourcing brief")

        required_groups = gate_cfg.get("context_pack", {}).get("required_claim_groups", ["identity_scope", "product_depth"])
        min_required = min(
            int(gate_cfg.get("context_pack", {}).get("min_required_groups_met", 2)),
            len(required_groups),
        )
        covered = len([group for group in required_groups if group in context_claim_groups_available])
        if covered < min_required:
            missing_items["context_pack"].append(
                f"Evidence pattern coverage too low ({covered}/{len(required_groups)})"
            )
        context_pack_ready = bool(
            has_company_or_brief
            and (
                str(sourcing_brief.get("source_summary") or "").strip()
                or sourcing_brief.get("customer_nodes")
                or sourcing_brief.get("capability_nodes")
            )
            and covered >= min_required
        )
    else:
        missing_items["context_pack"].append("Create company profile")
    
    # Check scope review
    scope_review_ready = False
    if profile:
        company_context_result = await db.execute(
            select(CompanyContextPack).where(CompanyContextPack.workspace_id == workspace_id)
        )
        company_context_pack = company_context_result.scalar_one_or_none()
        if company_context_pack:
            scope_payload = derive_scope_review_payload(company_context_pack, profile)
            discovery_scope_hints = derive_discovery_scope_hints(company_context_pack, profile)
            if not (discovery_scope_hints.get("source_capabilities") or []):
                missing_items["scope_review"].append("Keep at least 1 source capability or workflow in scope")
            if not (discovery_scope_hints.get("adjacent_capabilities") or []):
                missing_items["scope_review"].append("Approve at least 1 adjacent capability before discovery")
            if not normalize_expansion_brief(company_context_pack.expansion_brief_json or {}).get("confirmed_at"):
                missing_items["scope_review"].append("Confirm the reviewed scope before universe discovery")
            if not (
                scope_payload.get("source_capabilities")
                or scope_payload.get("adjacent_capabilities")
                or scope_payload.get("adjacent_customer_segments")
            ):
                missing_items["scope_review"].append("Generate the expansion brief before scope review")
            scope_review_ready = bool(
                normalize_expansion_brief(company_context_pack.expansion_brief_json or {}).get("confirmed_at")
                and (discovery_scope_hints.get("source_capabilities") or [])
                and (discovery_scope_hints.get("adjacent_capabilities") or [])
            )
        else:
            missing_items["scope_review"].append("Generate company context first")
    else:
        missing_items["scope_review"].append("Create company profile")
    
    # Check universe with evidence-pattern decisions
    screenings_result = await db.execute(
        select(CompanyScreening)
        .where(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
    )
    screenings_all = screenings_result.scalars().all()
    latest_by_company: Dict[int, CompanyScreening] = {}
    for row in screenings_all:
        if not row.company_id:
            continue
        if row.company_id not in latest_by_company:
            latest_by_company[row.company_id] = row

    universe_cfg = gate_cfg.get("universe", {})
    allowed_classes = set(universe_cfg.get("allowed_classes", ["good_target", "borderline_watchlist"]))
    min_decision_qualified = int(
        universe_cfg.get("min_decision_qualified_companies", universe_cfg.get("min_decision_qualified_vendors", 5))
    )
    max_insufficient_ratio = float(universe_cfg.get("max_insufficient_ratio", 0.5))

    decision_qualified = [
        row for row in latest_by_company.values()
        if str(row.decision_classification or "") in allowed_classes
    ]
    insufficient_count = len(
        [row for row in latest_by_company.values() if str(row.evidence_sufficiency or "") == "insufficient"]
    )
    insufficient_ratio = insufficient_count / max(1, len(latest_by_company))

    kept_companies_result = await db.execute(
        select(func.count(Company.id)).where(
            Company.workspace_id == workspace_id,
            Company.status.in_([CompanyStatus.kept, CompanyStatus.enriched]),
        )
    )
    kept_companies_count = kept_companies_result.scalar() or 0
    universe_ready = (
        len(decision_qualified) >= min_decision_qualified
        and insufficient_ratio <= max_insufficient_ratio
    )
    if len(decision_qualified) < min_decision_qualified:
        missing_items["universe"].append(
            f"Need decision-qualified companies ({len(decision_qualified)}/{min_decision_qualified})"
        )
    if insufficient_ratio > max_insufficient_ratio:
        missing_items["universe"].append(
            f"Evidence insufficiency ratio too high ({round(insufficient_ratio, 2)} > {max_insufficient_ratio})"
        )
    # Legacy fallback messaging (one release cycle)
    if kept_companies_count < 5:
        missing_items["universe"].append(f"Keep at least 5 companies ({kept_companies_count} kept)")
    
    # Check segmentation (has reviewed and focused)
    segmentation_ready = universe_ready and len(decision_qualified) >= 10
    if not segmentation_ready and universe_ready:
        missing_items["segmentation"].append("Review and keep at least 10 companies")
    
    # Check enrichment
    enrichment_cfg = gate_cfg.get("enrichment", {})
    min_enriched = int(enrichment_cfg.get("min_enriched_companies", enrichment_cfg.get("min_enriched_vendors", 5)))
    required_groups = set(
        enrichment_cfg.get("required_groups_per_company", enrichment_cfg.get("required_groups_per_vendor", ["product_depth", "traction"]))
    )

    enriched_result = await db.execute(
        select(func.count(Company.id)).where(
            Company.workspace_id == workspace_id,
            Company.status == CompanyStatus.enriched
        )
    )
    enriched_count = enriched_result.scalar() or 0

    enriched_companies_result = await db.execute(
        select(Company).where(
            Company.workspace_id == workspace_id,
            Company.status == CompanyStatus.enriched,
        )
    )
    enriched_companies = enriched_companies_result.scalars().all()
    enriched_with_groups = 0
    for company in enriched_companies:
        screening = latest_by_company.get(company.id)
        missing_groups = set(screening.missing_claim_groups_json or []) if screening else set(required_groups)
        if required_groups.isdisjoint(missing_groups):
            enriched_with_groups += 1

    enrichment_ready = enriched_count >= min_enriched and enriched_with_groups >= min_enriched
    if not enrichment_ready:
        missing_items["enrichment"].append(
            f"Enrich at least {min_enriched} companies with required evidence groups ({enriched_with_groups}/{min_enriched})"
        )
    
    return GatesResponse(
        context_pack=context_pack_ready,
        scope_review=scope_review_ready,
        universe=universe_ready,
        segmentation=segmentation_ready,
        enrichment=enrichment_ready,
        missing_items=missing_items
    )


# ============================================================================
# Jobs
# ============================================================================

@router.get("/{workspace_id}/jobs", response_model=List[JobResponse])
async def list_workspace_jobs(
    workspace_id: int,
    job_type: Optional[str] = Query(None),
    state: Optional[str] = Query(None),
    limit: int = Query(50),
    db: AsyncSession = Depends(get_db)
):
    """List jobs for a workspace."""
    query = select(Job).where(Job.workspace_id == workspace_id)
    
    if job_type:
        query = query.where(Job.job_type == JobType(job_type))
    if state:
        query = query.where(Job.state == JobState(state))
    
    query = query.order_by(Job.created_at.desc()).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return [
        JobResponse(
            id=job.id,
            workspace_id=job.workspace_id,
            company_id=job.company_id,
            job_type=job.job_type.value,
            state=job.state.value,
            provider=job.provider.value,
            progress=job.progress,
            progress_message=job.progress_message,
            result_json=job.result_json,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at
        )
        for job in jobs
    ]


@router.get("/{workspace_id}/jobs/{job_id}", response_model=JobResponse)
async def get_job(
    workspace_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get a specific job."""
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        company_id=job.company_id,
        job_type=job.job_type.value,
        state=job.state.value,
        provider=job.provider.value,
        progress=job.progress,
        progress_message=job.progress_message,
        result_json=job.result_json,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at
    )


@router.post("/{workspace_id}/jobs/{job_id}:cancel", response_model=JobResponse)
async def cancel_job(
    workspace_id: int,
    job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a queued or running workspace job."""
    result = await db.execute(
        select(Job).where(Job.id == job_id, Job.workspace_id == workspace_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.state not in RUNTIME_ACTIVE_JOB_STATES:
        raise HTTPException(status_code=400, detail="Job is not cancelable")

    if job.interaction_id:
        try:
            from app.workers.celery_app import celery_app

            celery_app.control.revoke(job.interaction_id, terminate=True, signal="SIGTERM")
        except Exception:
            # Best-effort revoke. We still mark the job stopped in the app model.
            pass

    job.state = JobState.failed
    job.error_message = "Stopped by user"
    job.progress_message = "Stopped"
    job.finished_at = datetime.utcnow()
    await db.commit()
    await db.refresh(job)

    return _to_job_response(job)
