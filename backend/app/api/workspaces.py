"""Workspace API routes - Full CRUD and workflow endpoints."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

from app.models.base import get_db
from app.models.workspace import Workspace, CompanyProfile, BrickTaxonomy, BrickMapping, DEFAULT_BRICKS
from app.models.vendor import Vendor, VendorDossier, VendorStatus
from app.models.job import Job, JobType, JobState, JobProvider
from app.models.workspace_evidence import WorkspaceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, VendorFact
from app.models.claims_graph import ClaimGraphNode, ClaimGraphEdge, ClaimGraphEdgeEvidence
from app.models.workspace_feedback import WorkspaceFeedbackEvent
from app.models.evaluation import EvaluationRun, EvaluationSampleResult
from app.models.intelligence import (
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    ComparatorSourceRun,
    RegistryQueryLog,
    VendorMention,
    VendorScreening,
    VendorClaim,
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
    build_adjacency_map,
    classify_size_bucket,
    compute_lens_scores,
    estimate_size_from_signals,
    extract_customers_and_integrations,
    normalize_domain,
    is_trusted_source_url,
    modules_with_evidence,
    normalize_country,
    source_label_for_url,
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
    vendor_count: int = 0
    has_context_pack: bool = False
    has_confirmed_taxonomy: bool = False

    class Config:
        from_attributes = True


class GeoScope(BaseModel):
    region: str = "EU+UK"
    include_countries: List[str] = Field(default_factory=list)
    exclude_countries: List[str] = Field(default_factory=list)


class CompanyProfileUpdate(BaseModel):
    buyer_company_url: Optional[str] = None
    reference_vendor_urls: Optional[List[str]] = None
    reference_evidence_urls: Optional[List[str]] = None
    geo_scope: Optional[GeoScope] = None


class CompanyProfileResponse(BaseModel):
    id: int
    workspace_id: int
    buyer_company_url: Optional[str]
    buyer_context_summary: Optional[str]
    reference_vendor_urls: List[str]
    reference_evidence_urls: List[str]
    reference_summaries: Dict[str, str]
    geo_scope: Dict[str, Any]
    context_pack_markdown: Optional[str]
    context_pack_generated_at: Optional[datetime]
    product_pages_found: int
    context_pack_json: Optional[Dict[str, Any]] = None  # Full structured data

    class Config:
        from_attributes = True


class BrickItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None


class BrickTaxonomyUpdate(BaseModel):
    bricks: Optional[List[BrickItem]] = None
    priority_brick_ids: Optional[List[str]] = None
    vertical_focus: Optional[List[str]] = None


class BrickTaxonomyResponse(BaseModel):
    id: int
    workspace_id: int
    bricks: List[Dict[str, Any]]
    priority_brick_ids: List[str]
    vertical_focus: List[str]
    version: int
    confirmed: bool

    class Config:
        from_attributes = True


class VendorCreate(BaseModel):
    name: str
    website: Optional[str] = None
    hq_country: Optional[str] = None
    tags_vertical: List[str] = Field(default_factory=list)


class VendorUpdate(BaseModel):
    name: Optional[str] = None
    website: Optional[str] = None
    hq_country: Optional[str] = None
    operating_countries: Optional[List[str]] = None
    tags_vertical: Optional[List[str]] = None
    tags_custom: Optional[List[str]] = None
    status: Optional[str] = None


class VendorResponse(BaseModel):
    id: int
    workspace_id: int
    name: str
    website: Optional[str]
    official_website_url: Optional[str] = None
    discovery_url: Optional[str] = None
    entity_type: Optional[str] = None
    hq_country: Optional[str]
    operating_countries: List[str]
    tags_vertical: List[str]
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

    class Config:
        from_attributes = True


class VendorDossierResponse(BaseModel):
    id: int
    vendor_id: int
    dossier_json: Dict[str, Any]
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


class EnrichBatchRequest(BaseModel):
    vendor_ids: List[int]
    job_types: List[str] = Field(default=["enrich_full"])


class JobResponse(BaseModel):
    id: int
    workspace_id: int
    vendor_id: Optional[int]
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
    brick_model: bool
    universe: bool
    segmentation: bool
    enrichment: bool
    missing_items: Dict[str, List[str]]


class LensVendor(BaseModel):
    id: int
    name: str
    website: Optional[str]
    overlapping_bricks: List[str] = Field(default_factory=list)
    added_bricks: List[str] = Field(default_factory=list)
    evidence_count: int = 0
    customer_overlaps: List[str] = Field(default_factory=list)
    proof_bullets: List[Dict[str, Any]] = Field(default_factory=list)


class LensResponse(BaseModel):
    vendors: List[LensVendor]
    total_count: int


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
    vendor_id: int
    name: str
    website: Optional[str] = None
    hq_country: Optional[str] = None
    legal_status: Optional[str] = None
    size_bucket: str
    size_estimate: Optional[int] = None
    size_range_low: Optional[int] = None
    size_range_high: Optional[int] = None
    compete_score: float
    complement_score: float
    brick_mapping: List[ReportClaim] = Field(default_factory=list)
    customer_partner_evidence: List[ReportClaim] = Field(default_factory=list)
    filing_metrics: Dict[str, SourcedValue] = Field(default_factory=dict)
    source_pills: List[SourcePill] = Field(default_factory=list)
    coverage_note: Optional[str] = None
    next_validation_questions: List[str] = Field(default_factory=list)
    decision_classification: Optional[str] = None
    reason_highlights: List[str] = Field(default_factory=list)
    evidence_quality_summary: Dict[str, Any] = Field(default_factory=dict)
    known_unknowns: List[str] = Field(default_factory=list)


class UniverseTopCandidateResponse(BaseModel):
    vendor_id: Optional[int] = None
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
    degraded_reasons: List[str] = Field(default_factory=list)


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


class ReportLensItem(BaseModel):
    vendor_id: int
    name: str
    website: Optional[str] = None
    size_bucket: str
    score: float
    lens_breakdown: Dict[str, Any] = Field(default_factory=dict)
    highlights: List[ReportClaim] = Field(default_factory=list)


class ReportLensResponse(BaseModel):
    mode: str
    items: List[ReportLensItem]
    total_count: int
    counts_by_bucket: Dict[str, int] = Field(default_factory=dict)


class VendorDecisionResponse(BaseModel):
    vendor_id: int
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
    max_vendors: int = 80
    stale_only: bool = False
    classifications: List[str] = Field(
        default_factory=lambda: ["borderline_watchlist", "insufficient_evidence"]
    )


class WorkspaceFeedbackCreate(BaseModel):
    vendor_id: Optional[int] = None
    screening_id: Optional[int] = None
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
    vendor_id: Optional[int]
    screening_id: Optional[int]
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
        vendor_id=job.vendor_id,
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


def _quality_payload_from_job_result(result_json: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    payload = result_json if isinstance(result_json, dict) else {}
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
    return {
        "run_quality_tier": run_quality_tier,
        "quality_gate_passed": quality_gate_passed,
        "degraded_reasons": degraded_reasons,
        "model_attempt_trace": payload.get("model_attempt_trace") if isinstance(payload.get("model_attempt_trace"), list) else [],
        "stage_time_ms": payload.get("stage_time_ms") if isinstance(payload.get("stage_time_ms"), dict) else {},
        "timeout_events": payload.get("timeout_events") if isinstance(payload.get("timeout_events"), list) else [],
        "screening_run_id": str(payload.get("screening_run_id") or "").strip() or None,
    }


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


def _source_from_evidence(evidence: Optional[WorkspaceEvidence]) -> Optional[SourcePill]:
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
    evidence_by_url: Dict[str, WorkspaceEvidence],
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
    facts: List[VendorFact],
    claims: List[VendorClaim],
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


def _is_directory_host_url(url: Optional[str]) -> bool:
    domain = normalize_domain(url)
    if not domain:
        return False
    return any(domain == token or domain.endswith(f".{token}") for token in DIRECTORY_HOST_TOKENS)


def _reason_codes_payload(screening: Optional[VendorScreening]) -> Dict[str, List[str]]:
    if not screening:
        return {"positive": [], "caution": [], "reject": []}
    return {
        "positive": [str(code) for code in (screening.positive_reason_codes_json or []) if str(code)],
        "caution": [str(code) for code in (screening.caution_reason_codes_json or []) if str(code)],
        "reject": [str(code) for code in (screening.reject_reason_codes_json or []) if str(code)],
    }


async def _latest_screening_for_vendor(
    db: AsyncSession,
    workspace_id: int,
    vendor_id: int,
) -> Optional[VendorScreening]:
    screening_result = await db.execute(
        select(VendorScreening)
        .where(
            VendorScreening.workspace_id == workspace_id,
            VendorScreening.vendor_id == vendor_id,
        )
        .order_by(VendorScreening.created_at.desc())
        .limit(1)
    )
    return screening_result.scalar_one_or_none()


async def _vendor_response_from_row(db: AsyncSession, vendor: Vendor) -> VendorResponse:
    evidence_count_result = await db.execute(
        select(func.count(WorkspaceEvidence.id)).where(WorkspaceEvidence.vendor_id == vendor.id)
    )
    evidence_count = evidence_count_result.scalar() or 0
    screening = await _latest_screening_for_vendor(db, vendor.workspace_id, vendor.id)
    screening_meta = screening.screening_meta_json if screening and isinstance(screening.screening_meta_json, dict) else {}
    citation_summary_v1 = _citation_summary_from_meta(screening_meta)
    diagnostics = _screening_diagnostics_from_meta(screening_meta)
    return VendorResponse(
        id=vendor.id,
        workspace_id=vendor.workspace_id,
        name=vendor.name,
        website=vendor.website,
        official_website_url=(
            screening.candidate_official_website
            if screening and screening.candidate_official_website
            else vendor.website
        ),
        discovery_url=screening.candidate_discovery_url if screening else None,
        entity_type=str(screening_meta.get("entity_type") or "company") if screening else "company",
        hq_country=vendor.hq_country,
        operating_countries=vendor.operating_countries or [],
        tags_vertical=vendor.tags_vertical or [],
        tags_custom=vendor.tags_custom or [],
        status=vendor.status.value,
        why_relevant=vendor.why_relevant or [],
        is_manual=vendor.is_manual,
        created_at=vendor.created_at,
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
    )


def _to_workspace_feedback_response(event: WorkspaceFeedbackEvent) -> WorkspaceFeedbackResponse:
    return WorkspaceFeedbackResponse(
        id=event.id,
        workspace_id=event.workspace_id,
        vendor_id=event.vendor_id,
        screening_id=event.screening_id,
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
    """Create a new workspace with default taxonomy."""
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
    
    # Create default brick taxonomy
    default_bricks = [
        {"id": str(uuid.uuid4()), "name": brick, "description": None}
        for brick in DEFAULT_BRICKS
    ]
    taxonomy = BrickTaxonomy(
        workspace_id=workspace.id,
        bricks=default_bricks,
        priority_brick_ids=[]
    )
    db.add(taxonomy)
    
    await db.commit()
    await db.refresh(workspace)
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        region_scope=workspace.region_scope,
        created_at=workspace.created_at,
        vendor_count=0,
        has_context_pack=False,
        has_confirmed_taxonomy=False
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
        # Get vendor count
        vendor_count_result = await db.execute(
            select(func.count(Vendor.id)).where(Vendor.workspace_id == ws.id)
        )
        vendor_count = vendor_count_result.scalar() or 0
        
        # Check context pack
        profile_result = await db.execute(
            select(CompanyProfile).where(CompanyProfile.workspace_id == ws.id)
        )
        profile = profile_result.scalar_one_or_none()
        has_context_pack = profile is not None and profile.context_pack_markdown is not None
        
        # Check taxonomy
        taxonomy_result = await db.execute(
            select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == ws.id)
        )
        taxonomy = taxonomy_result.scalar_one_or_none()
        has_confirmed_taxonomy = taxonomy is not None and taxonomy.confirmed
        
        responses.append(WorkspaceResponse(
            id=ws.id,
            name=ws.name,
            region_scope=ws.region_scope,
            created_at=ws.created_at,
            vendor_count=vendor_count,
            has_context_pack=has_context_pack,
            has_confirmed_taxonomy=has_confirmed_taxonomy
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
    
    vendor_count_result = await db.execute(
        select(func.count(Vendor.id)).where(Vendor.workspace_id == workspace_id)
    )
    vendor_count = vendor_count_result.scalar() or 0
    
    profile_result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = profile_result.scalar_one_or_none()
    has_context_pack = profile is not None and profile.context_pack_markdown is not None
    
    taxonomy_result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = taxonomy_result.scalar_one_or_none()
    has_confirmed_taxonomy = taxonomy is not None and taxonomy.confirmed
    
    return WorkspaceResponse(
        id=workspace.id,
        name=workspace.name,
        region_scope=workspace.region_scope,
        created_at=workspace.created_at,
        vendor_count=vendor_count,
        has_context_pack=has_context_pack,
        has_confirmed_taxonomy=has_confirmed_taxonomy
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
        workspace.region_scope = data.region_scope
    
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
        buyer_context_summary=profile.buyer_context_summary,
        reference_vendor_urls=profile.reference_vendor_urls or [],
        reference_evidence_urls=profile.reference_evidence_urls or [],
        reference_summaries=profile.reference_summaries or {},
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
        "buyer_context_summary": profile.buyer_context_summary,
        "reference_vendor_urls": profile.reference_vendor_urls or [],
        "reference_evidence_urls": profile.reference_evidence_urls or [],
        "reference_summaries": profile.reference_summaries or {},
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
    if data.reference_vendor_urls is not None:
        profile.reference_vendor_urls = _clean_url_list(data.reference_vendor_urls, max_items=10)
    if data.reference_evidence_urls is not None:
        profile.reference_evidence_urls = _clean_url_list(data.reference_evidence_urls, max_items=50)
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
    
    # Trigger async task (Celery)
    from app.workers.workspace_tasks import generate_context_pack_v2
    generate_context_pack_v2.delay(job.id)
    
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        vendor_id=job.vendor_id,
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
# Brick Taxonomy
# ============================================================================

@router.get("/{workspace_id}/bricks", response_model=BrickTaxonomyResponse)
async def get_bricks(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get the brick taxonomy for a workspace."""
    result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = result.scalar_one_or_none()
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Brick taxonomy not found")
    
    return BrickTaxonomyResponse(
        id=taxonomy.id,
        workspace_id=taxonomy.workspace_id,
        bricks=taxonomy.bricks or [],
        priority_brick_ids=taxonomy.priority_brick_ids or [],
        vertical_focus=taxonomy.vertical_focus or [],
        version=taxonomy.version,
        confirmed=taxonomy.confirmed
    )


@router.patch("/{workspace_id}/bricks", response_model=BrickTaxonomyResponse)
async def update_bricks(
    workspace_id: int,
    data: BrickTaxonomyUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update brick taxonomy (rename, merge, split, set priorities)."""
    result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = result.scalar_one_or_none()
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Brick taxonomy not found")
    
    if data.bricks is not None:
        taxonomy.bricks = [b.model_dump() for b in data.bricks]
        taxonomy.version += 1
    if data.priority_brick_ids is not None:
        taxonomy.priority_brick_ids = data.priority_brick_ids
    if data.vertical_focus is not None:
        taxonomy.vertical_focus = data.vertical_focus
    
    taxonomy.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(taxonomy)
    
    return await get_bricks(workspace_id, db)


@router.post("/{workspace_id}/bricks:confirm", response_model=BrickTaxonomyResponse)
async def confirm_bricks(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Confirm brick taxonomy to unlock discovery."""
    result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = result.scalar_one_or_none()
    if not taxonomy:
        raise HTTPException(status_code=404, detail="Brick taxonomy not found")
    
    taxonomy.confirmed = True
    taxonomy.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(taxonomy)
    
    return await get_bricks(workspace_id, db)


# ============================================================================
# Discovery & Vendors
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
    run_discovery_universe.delay(job.id)
    
    return JobResponse(
        id=job.id,
        workspace_id=job.workspace_id,
        vendor_id=job.vendor_id,
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

    screenings_query = (
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
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
    claims: List[VendorClaim] = []
    if screening_ids:
        claims_result = await db.execute(
            select(VendorClaim).where(VendorClaim.screening_id.in_(screening_ids))
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
        evidence_claims_by_screening[claim.screening_id] = evidence_claims_by_screening.get(claim.screening_id, 0) + 1
        seen_dimensions_per_screening.setdefault(claim.screening_id, set()).add((claim.dimension or "").lower())

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
        "degraded_reasons": quality_payload["degraded_reasons"],
        "model_attempt_trace": quality_payload["model_attempt_trace"],
        "stage_time_ms": quality_payload["stage_time_ms"],
        "timeout_events": quality_payload["timeout_events"],
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/{workspace_id}/vendors", response_model=List[VendorResponse])
async def list_vendors(
    workspace_id: int,
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """List vendors in workspace with optional status filter."""
    query = select(Vendor).where(Vendor.workspace_id == workspace_id)
    
    if status:
        query = query.where(Vendor.status == VendorStatus(status))
    
    query = query.order_by(Vendor.created_at.desc())
    result = await db.execute(query)
    vendors = result.scalars().all()

    responses: List[VendorResponse] = []
    for row in vendors:
        responses.append(await _vendor_response_from_row(db, row))
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
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
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

    vendor_ids = [row.vendor_id for row in candidates if row.vendor_id]
    entity_ids = [row.candidate_entity_id for row in candidates if row.candidate_entity_id]

    vendor_map: Dict[int, Vendor] = {}
    if vendor_ids:
        vendor_result = await db.execute(select(Vendor).where(Vendor.id.in_(vendor_ids)))
        vendor_map = {row.id: row for row in vendor_result.scalars().all()}

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
        vendor = vendor_map.get(screening.vendor_id) if screening.vendor_id else None
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
                vendor_id=screening.vendor_id,
                candidate_entity_id=screening.candidate_entity_id,
                company_name=(vendor.name if vendor else screening.candidate_name),
                official_website_url=(
                    screening.candidate_official_website
                    or (vendor.website if vendor else None)
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
                degraded_reasons=quality_payload["degraded_reasons"],
            )
        )

    return output


@router.post("/{workspace_id}/vendors", response_model=VendorResponse)
async def create_vendor(
    workspace_id: int,
    data: VendorCreate,
    db: AsyncSession = Depends(get_db)
):
    """Manually add a vendor to the workspace."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    vendor = Vendor(
        workspace_id=workspace_id,
        name=data.name,
        website=data.website,
        hq_country=data.hq_country,
        tags_vertical=data.tags_vertical,
        status=VendorStatus.candidate,
        is_manual=True
    )
    db.add(vendor)
    await db.commit()
    await db.refresh(vendor)
    
    return await _vendor_response_from_row(db, vendor)


@router.patch("/{workspace_id}/vendors/{vendor_id}", response_model=VendorResponse)
async def update_vendor(
    workspace_id: int,
    vendor_id: int,
    data: VendorUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update vendor details or status (keep/remove)."""
    result = await db.execute(
        select(Vendor).where(
            Vendor.id == vendor_id,
            Vendor.workspace_id == workspace_id
        )
    )
    vendor = result.scalar_one_or_none()
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")
    
    if data.name is not None:
        vendor.name = data.name
    if data.website is not None:
        vendor.website = data.website
    if data.hq_country is not None:
        vendor.hq_country = data.hq_country
    if data.operating_countries is not None:
        vendor.operating_countries = data.operating_countries
    if data.tags_vertical is not None:
        vendor.tags_vertical = data.tags_vertical
    if data.tags_custom is not None:
        vendor.tags_custom = data.tags_custom
    if data.status is not None:
        vendor.status = VendorStatus(data.status)
    
    vendor.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(vendor)
    
    return await _vendor_response_from_row(db, vendor)


# ============================================================================
# Enrichment
# ============================================================================

@router.post("/{workspace_id}/vendors:enrich", response_model=List[JobResponse])
async def enrich_vendors_batch(
    workspace_id: int,
    data: EnrichBatchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Batch enrich multiple vendors."""
    # Verify workspace exists
    result = await db.execute(
        select(Workspace).where(Workspace.id == workspace_id)
    )
    workspace = result.scalar_one_or_none()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    jobs = []
    for vendor_id in data.vendor_ids:
        # Verify vendor exists
        vendor_result = await db.execute(
            select(Vendor).where(Vendor.id == vendor_id, Vendor.workspace_id == workspace_id)
        )
        vendor = vendor_result.scalar_one_or_none()
        if not vendor:
            continue
        
        for job_type_str in data.job_types:
            job_type = JobType(job_type_str)
            job = Job(
                workspace_id=workspace_id,
                vendor_id=vendor_id,
                job_type=job_type,
                state=JobState.queued,
                provider=JobProvider.gemini_flash
            )
            db.add(job)
            await db.flush()
            jobs.append(job)
    
    await db.commit()
    
    # Trigger async tasks
    from app.workers.workspace_tasks import run_enrich_vendor
    for job in jobs:
        run_enrich_vendor.delay(job.id)
    
    return [
        JobResponse(
            id=job.id,
            workspace_id=job.workspace_id,
            vendor_id=job.vendor_id,
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


@router.get("/{workspace_id}/vendors/{vendor_id}/dossier", response_model=Optional[VendorDossierResponse])
async def get_vendor_dossier(
    workspace_id: int,
    vendor_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get the latest dossier for a vendor."""
    result = await db.execute(
        select(VendorDossier)
        .where(VendorDossier.vendor_id == vendor_id)
        .order_by(VendorDossier.version.desc())
        .limit(1)
    )
    dossier = result.scalar_one_or_none()
    
    if not dossier:
        return None
    
    return VendorDossierResponse(
        id=dossier.id,
        vendor_id=dossier.vendor_id,
        dossier_json=dossier.dossier_json,
        version=dossier.version,
        created_at=dossier.created_at
    )


# ============================================================================
# Lenses
# ============================================================================

@router.get("/{workspace_id}/lenses/similarity", response_model=LensResponse)
async def get_similarity_lens(
    workspace_id: int,
    brick_ids: Optional[str] = Query(None, description="Comma-separated brick IDs to filter"),
    db: AsyncSession = Depends(get_db)
):
    """Get evidence-backed compete lens using fixed weighted rules."""
    profile_result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = profile_result.scalar_one_or_none()

    reference_tokens: set[str] = set()
    if profile:
        if profile.buyer_company_url:
            domain = _safe_domain(profile.buyer_company_url)
            if domain:
                reference_tokens.add(domain)
                reference_tokens.add(domain.split(".")[0])
        for url in (profile.reference_vendor_urls or []):
            domain = _safe_domain(url)
            if domain:
                reference_tokens.add(domain)
                reference_tokens.add(domain.split(".")[0])

    result = await db.execute(
        select(Vendor).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched])
        )
    )
    vendors = result.scalars().all()
    
    taxonomy_result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = taxonomy_result.scalar_one_or_none()
    if not taxonomy:
        return LensResponse(vendors=[], total_count=0)

    priority_bricks = set(taxonomy.priority_brick_ids or [])
    if not priority_bricks:
        priority_bricks = {b.get("id") for b in (taxonomy.bricks or []) if b.get("id")}

    selected_bricks = set(brick_ids.split(",")) if brick_ids else priority_bricks
    adjacency_map = build_adjacency_map(taxonomy.bricks or [])
    geo_scope = profile.geo_scope if profile and profile.geo_scope else {}
    include_countries = {normalize_country(c) for c in geo_scope.get("include_countries", []) if normalize_country(c)}
    exclude_countries = {normalize_country(c) for c in geo_scope.get("exclude_countries", []) if normalize_country(c)}
    vertical_focus = set(taxonomy.vertical_focus or [])
    
    lens_vendors = []
    for vendor in vendors:
        dossier_result = await db.execute(
            select(VendorDossier)
            .where(VendorDossier.vendor_id == vendor.id)
            .order_by(VendorDossier.version.desc())
            .limit(1)
        )
        dossier = dossier_result.scalar_one_or_none()
        evidence_result = await db.execute(
            select(WorkspaceEvidence).where(WorkspaceEvidence.vendor_id == vendor.id)
        )
        evidence_items = evidence_result.scalars().all()
        evidence_count = len(evidence_items)
        evidence_by_url = {e.source_url: e for e in evidence_items if e.source_url}

        dossier_json = dossier.dossier_json if dossier else {}
        fallback_capabilities = [
            tag.split(":", 1)[1].strip()
            for tag in (vendor.tags_custom or [])
            if isinstance(tag, str) and tag.startswith("capability:")
        ]
        fallback_evidence_urls = [
            evidence.source_url
            for evidence in evidence_items
            if evidence.source_url and is_trusted_source_url(evidence.source_url)
        ]
        modules = modules_with_evidence(
            dossier_json,
            fallback_capabilities=fallback_capabilities,
            fallback_evidence_urls=fallback_evidence_urls,
        )
        customers, integrations = extract_customers_and_integrations(dossier_json)

        normalized_country = normalize_country(vendor.hq_country)
        if include_countries:
            geo_match = normalized_country in include_countries
        else:
            geo_match = True
        if normalized_country in exclude_countries:
            geo_match = False
        vertical_match = bool(set(vendor.tags_vertical or []).intersection(vertical_focus)) if vertical_focus else True

        lens = compute_lens_scores(
            vendor_modules=modules,
            customers=customers,
            integrations=integrations,
            priority_bricks=selected_bricks,
            adjacency_map=adjacency_map,
            reference_tokens=reference_tokens,
            geo_vertical_match=geo_match and vertical_match,
            has_geo_vertical_evidence=bool(evidence_items),
        )

        overlap_brick_ids = lens.get("compete", {}).get("brick_overlap", {}).get("overlap_bricks", [])
        overlapping_bricks = [m["name"] for m in modules if m.get("brick_id") in overlap_brick_ids]

        proof_bullets = []
        for module in modules:
            if not module.get("has_evidence"):
                continue
            first_url = module["evidence_urls"][0]
            proof_bullets.append(
                {
                    "text": f"Compete signal: offers {module.get('name', 'capability')}",
                    "citation_url": first_url,
                }
            )
            if len(proof_bullets) >= 3:
                break
        if not proof_bullets and modules:
            proof_bullets.append(
                {
                    "text": "Capability overlap inferred but not fully source-backed yet",
                    "citation_url": None,
                }
            )
        
        lens_vendors.append(LensVendor(
            id=vendor.id,
            name=vendor.name,
            website=vendor.website,
            overlapping_bricks=overlapping_bricks,
            added_bricks=[],
            evidence_count=evidence_count,
            customer_overlaps=[],
            proof_bullets=proof_bullets
        ))
    
    lens_vendors.sort(key=lambda v: len(v.overlapping_bricks), reverse=True)
    
    return LensResponse(vendors=lens_vendors, total_count=len(lens_vendors))


@router.get("/{workspace_id}/lenses/complementarity", response_model=LensResponse)
async def get_complementarity_lens(
    workspace_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Get evidence-backed complement lens using fixed weighted rules."""
    profile_result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = profile_result.scalar_one_or_none()
    
    taxonomy_result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = taxonomy_result.scalar_one_or_none()
    if not taxonomy:
        return LensResponse(vendors=[], total_count=0)

    priority_bricks = set(taxonomy.priority_brick_ids or [])
    if not priority_bricks:
        priority_bricks = {b.get("id") for b in (taxonomy.bricks or []) if b.get("id")}
    adjacency_map = build_adjacency_map(taxonomy.bricks or [])

    reference_tokens: set[str] = set()
    if profile:
        for url in [profile.buyer_company_url] + (profile.reference_vendor_urls or []):
            domain = _safe_domain(url)
            if domain:
                reference_tokens.add(domain)
                reference_tokens.add(domain.split(".")[0])

    geo_scope = profile.geo_scope if profile and profile.geo_scope else {}
    include_countries = {normalize_country(c) for c in geo_scope.get("include_countries", []) if normalize_country(c)}
    exclude_countries = {normalize_country(c) for c in geo_scope.get("exclude_countries", []) if normalize_country(c)}
    vertical_focus = set(taxonomy.vertical_focus or [])

    result = await db.execute(
        select(Vendor).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched])
        )
    )
    vendors = result.scalars().all()
    
    lens_vendors = []
    for vendor in vendors:
        dossier_result = await db.execute(
            select(VendorDossier)
            .where(VendorDossier.vendor_id == vendor.id)
            .order_by(VendorDossier.version.desc())
            .limit(1)
        )
        dossier = dossier_result.scalar_one_or_none()
        evidence_result = await db.execute(
            select(WorkspaceEvidence).where(WorkspaceEvidence.vendor_id == vendor.id)
        )
        evidence_items = evidence_result.scalars().all()
        evidence_count = len(evidence_items)

        dossier_json = dossier.dossier_json if dossier else {}
        fallback_capabilities = [
            tag.split(":", 1)[1].strip()
            for tag in (vendor.tags_custom or [])
            if isinstance(tag, str) and tag.startswith("capability:")
        ]
        fallback_evidence_urls = [
            evidence.source_url
            for evidence in evidence_items
            if evidence.source_url and is_trusted_source_url(evidence.source_url)
        ]
        modules = modules_with_evidence(
            dossier_json,
            fallback_capabilities=fallback_capabilities,
            fallback_evidence_urls=fallback_evidence_urls,
        )
        customers, integrations = extract_customers_and_integrations(dossier_json)

        normalized_country = normalize_country(vendor.hq_country)
        if include_countries:
            geo_match = normalized_country in include_countries
        else:
            geo_match = True
        if normalized_country in exclude_countries:
            geo_match = False
        vertical_match = bool(set(vendor.tags_vertical or []).intersection(vertical_focus)) if vertical_focus else True

        lens = compute_lens_scores(
            vendor_modules=modules,
            customers=customers,
            integrations=integrations,
            priority_bricks=priority_bricks,
            adjacency_map=adjacency_map,
            reference_tokens=reference_tokens,
            geo_vertical_match=geo_match and vertical_match,
            has_geo_vertical_evidence=bool(evidence_items),
        )

        added_brick_ids = lens.get("complement", {}).get("adjacent_brick_potential", {}).get("adjacent_bricks", [])
        added_bricks = [m["name"] for m in modules if m.get("brick_id") in added_brick_ids]

        proof_bullets = []
        for integration in integrations:
            if integration.get("has_evidence"):
                proof_bullets.append(
                    {
                        "text": f"Integration adjacency: {integration.get('name', 'system')}",
                        "citation_url": integration.get("source_url"),
                    }
                )
            if len(proof_bullets) >= 3:
                break
        if not proof_bullets and integrations:
            proof_bullets.append(
                {
                    "text": "Integration potential inferred but not fully source-backed yet",
                    "citation_url": None,
                }
            )
        
        lens_vendors.append(LensVendor(
            id=vendor.id,
            name=vendor.name,
            website=vendor.website,
            overlapping_bricks=[],
            added_bricks=added_bricks,
            evidence_count=evidence_count,
            customer_overlaps=[],
            proof_bullets=proof_bullets
        ))
    
    lens_vendors.sort(key=lambda v: len(v.added_bricks), reverse=True)
    
    return LensResponse(vendors=lens_vendors, total_count=len(lens_vendors))


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

    generate_static_report.delay(
        job.id,
        {
            "name": data.name,
            "include_unknown_size": data.include_unknown_size,
            "include_outside_sme": data.include_outside_sme,
        },
    )
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
        vendor_result = await db.execute(
            select(Vendor).where(Vendor.id == item.vendor_id, Vendor.workspace_id == workspace_id)
        )
        vendor = vendor_result.scalar_one_or_none()
        if not vendor:
            continue

        dossier_result = await db.execute(
            select(VendorDossier)
            .where(VendorDossier.vendor_id == vendor.id)
            .order_by(VendorDossier.version.desc())
            .limit(1)
        )
        dossier = dossier_result.scalar_one_or_none()
        dossier_json = dossier.dossier_json if dossier else {}

        evidence_result = await db.execute(
            select(WorkspaceEvidence).where(WorkspaceEvidence.vendor_id == vendor.id)
        )
        evidence_items = evidence_result.scalars().all()
        evidence_by_url = {e.source_url: e for e in evidence_items if e.source_url}
        evidence_by_id = {e.id: e for e in evidence_items}
        fallback_capabilities = [
            tag.split(":", 1)[1].strip()
            for tag in (vendor.tags_custom or [])
            if isinstance(tag, str) and tag.startswith("capability:")
        ]
        fallback_evidence_urls = [
            evidence.source_url
            for evidence in evidence_items
            if evidence.source_url and is_trusted_source_url(evidence.source_url)
        ]

        modules = modules_with_evidence(
            dossier_json,
            fallback_capabilities=fallback_capabilities,
            fallback_evidence_urls=fallback_evidence_urls,
        )
        customers, integrations = extract_customers_and_integrations(dossier_json)

        brick_claims: List[ReportClaim] = []
        for module in modules:
            source_url = module["evidence_urls"][0] if module.get("has_evidence") else None
            claim = _build_claim(
                text=f"{module['name']} mapped to a strategy brick",
                source_url=source_url,
                evidence_by_url=evidence_by_url,
                confidence="high" if module.get("has_evidence") else "low",
            )
            brick_claims.append(claim)

        customer_claims: List[ReportClaim] = []
        for customer in customers:
            customer_claims.append(
                _build_claim(
                    text=f"Customer evidence: {customer['name']}",
                    source_url=customer.get("source_url"),
                    evidence_by_url=evidence_by_url,
                    confidence="high" if customer.get("has_evidence") else "low",
                )
            )
        for integration in integrations:
            customer_claims.append(
                _build_claim(
                    text=f"Integration evidence: {integration['name']}",
                    source_url=integration.get("source_url"),
                    evidence_by_url=evidence_by_url,
                    confidence="high" if integration.get("has_evidence") else "low",
                )
            )

        # Include discovery-time qualification reasons so report cards show auditable inclusion rationale.
        for reason in (vendor.why_relevant or [])[:6]:
            if not isinstance(reason, dict):
                continue
            reason_copy = str(reason.get("text") or "").strip()
            source_url = reason.get("citation_url")
            if not reason_copy:
                continue
            customer_claims.append(
                _build_claim(
                    text=f"Discovery rationale: {reason_copy}",
                    source_url=source_url,
                    evidence_by_url=evidence_by_url,
                    confidence="medium",
                )
            )

        facts_result = await db.execute(
            select(VendorFact).where(VendorFact.vendor_id == vendor.id)
        )
        facts = facts_result.scalars().all()
        claims_result = await db.execute(
            select(VendorClaim).where(VendorClaim.vendor_id == vendor.id)
        )
        claims = claims_result.scalars().all()
        screening = await _latest_screening_for_vendor(db, workspace_id, vendor.id)

        estimate = item.lens_breakdown_json.get("size_estimate")
        if estimate is None:
            estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts,
                evidence_items=evidence_items,
                tags_custom=vendor.tags_custom or [],
                why_relevant=vendor.why_relevant or [],
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
                    select(WorkspaceEvidence).where(WorkspaceEvidence.id == fact.source_evidence_id)
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

        source_pills: List[SourcePill] = []
        for claim in brick_claims + customer_claims:
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
        for tag in vendor.tags_custom or []:
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
            vendor_id=vendor.id,
            name=vendor.name,
            website=vendor.website,
            hq_country=vendor.hq_country,
            legal_status=legal_status,
            size_bucket=bucket,
            size_estimate=estimate,
            size_range_low=size_range_low,
            size_range_high=size_range_high,
            compete_score=item.compete_score,
            complement_score=item.complement_score,
            brick_mapping=brick_claims,
            customer_partner_evidence=customer_claims,
            filing_metrics=filing_metrics,
            source_pills=_dedupe_source_pills(source_pills),
            coverage_note=_coverage_note_for_country(vendor.hq_country),
            next_validation_questions=[
                "Can we verify customer concentration from at least two independent sources?",
                "What implementation risk exists in the top two capabilities?",
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


@router.get("/{workspace_id}/reports/{report_id}/lenses", response_model=ReportLensResponse)
async def get_report_lens(
    workspace_id: int,
    report_id: int,
    mode: str = Query(..., pattern="^(compete|complement)$"),
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
        select(ReportSnapshotItem).where(ReportSnapshotItem.report_id == report_id)
    )
    items = items_result.scalars().all()

    lens_items: List[ReportLensItem] = []
    counts_by_bucket: Dict[str, int] = {"sme_in_range": 0, "unknown": 0, "outside_sme_range": 0}
    for item in items:
        vendor_result = await db.execute(
            select(Vendor).where(Vendor.id == item.vendor_id, Vendor.workspace_id == workspace_id)
        )
        vendor = vendor_result.scalar_one_or_none()
        if not vendor:
            continue

        source_result = await db.execute(
            select(WorkspaceEvidence)
            .where(WorkspaceEvidence.vendor_id == vendor.id)
            .order_by(WorkspaceEvidence.captured_at.desc())
            .limit(1)
        )
        sample_source = _source_from_evidence(source_result.scalar_one_or_none())

        bucket = item.lens_breakdown_json.get("size_bucket", "unknown")
        counts_by_bucket[bucket] = counts_by_bucket.get(bucket, 0) + 1
        score = item.compete_score if mode == "compete" else item.complement_score
        component = item.lens_breakdown_json.get(mode, {})
        highlights: List[ReportClaim] = []
        for key, details in component.items():
            value = details.get("value", 0)
            evidence_count = details.get("evidence_count", 0)
            rendering = "fact" if evidence_count and sample_source else "hypothesis"
            highlights.append(
                ReportClaim(
                    text=f"{key.replace('_', ' ')}: {value}",
                    confidence="high" if evidence_count and sample_source else "low",
                    rendering=rendering,
                    source=sample_source if rendering == "fact" else None,
                )
            )

        lens_items.append(
            ReportLensItem(
                vendor_id=vendor.id,
                name=vendor.name,
                website=vendor.website,
                size_bucket=bucket,
                score=score,
                lens_breakdown=component,
                highlights=highlights,
            )
        )

    lens_items.sort(key=lambda row: row.score, reverse=True)
    return ReportLensResponse(
        mode=mode,
        items=lens_items,
        total_count=len(lens_items),
        counts_by_bucket=counts_by_bucket,
    )


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
    item_by_vendor: Dict[int, ReportSnapshotItem] = {item.vendor_id: item for item in report_items}

    screenings_result = await db.execute(
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
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
    claims: List[VendorClaim] = []
    if screening_ids:
        claims_result = await db.execute(
            select(VendorClaim).where(VendorClaim.screening_id.in_(screening_ids))
        )
        claims = claims_result.scalars().all()
    claims_by_screening: Dict[int, List[VendorClaim]] = {}
    for claim in claims:
        claims_by_screening.setdefault(claim.screening_id, []).append(claim)

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

    vendor_ids = [row.vendor_id for row in screenings if row.vendor_id]
    vendor_map: Dict[int, Vendor] = {}
    dossier_map: Dict[int, VendorDossier] = {}
    facts_by_vendor: Dict[int, List[VendorFact]] = {}
    evidence_by_vendor: Dict[int, List[WorkspaceEvidence]] = {}
    if vendor_ids:
        vendor_result = await db.execute(
            select(Vendor).where(Vendor.id.in_(vendor_ids), Vendor.workspace_id == workspace_id)
        )
        vendors = vendor_result.scalars().all()
        vendor_map = {vendor.id: vendor for vendor in vendors}

        for vendor_id in vendor_ids:
            dossier_result = await db.execute(
                select(VendorDossier)
                .where(VendorDossier.vendor_id == vendor_id)
                .order_by(VendorDossier.version.desc())
                .limit(1)
            )
            dossier = dossier_result.scalar_one_or_none()
            if dossier:
                dossier_map[vendor_id] = dossier

        facts_result = await db.execute(
            select(VendorFact).where(VendorFact.vendor_id.in_(vendor_ids))
        )
        for fact in facts_result.scalars().all():
            facts_by_vendor.setdefault(fact.vendor_id, []).append(fact)

        evidence_result = await db.execute(
            select(WorkspaceEvidence).where(WorkspaceEvidence.vendor_id.in_(vendor_ids))
        )
        for evidence in evidence_result.scalars().all():
            evidence_by_vendor.setdefault(evidence.vendor_id, []).append(evidence)

    source_runs_result = await db.execute(
        select(ComparatorSourceRun)
        .where(ComparatorSourceRun.workspace_id == workspace_id)
        .order_by(ComparatorSourceRun.captured_at.desc())
        .limit(10)
    )
    source_runs = source_runs_result.scalars().all()
    source_run_ids = [run.id for run in source_runs]
    mentions: List[VendorMention] = []
    if source_run_ids:
        mentions_result = await db.execute(
            select(VendorMention).where(VendorMention.source_run_id.in_(source_run_ids))
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
        vendor = vendor_map.get(screening.vendor_id) if screening.vendor_id else None
        candidate_entity = entity_map.get(screening.candidate_entity_id) if screening.candidate_entity_id else None
        vendor_claims = claims_by_screening.get(screening.id, [])
        report_item = item_by_vendor.get(vendor.id) if vendor else None
        dossier_json = {}
        if vendor and vendor.id in dossier_map:
            dossier_json = dossier_map[vendor.id].dossier_json or {}
        modules = modules_with_evidence(dossier_json)
        customers, integrations = extract_customers_and_integrations(dossier_json)
        facts = facts_by_vendor.get(vendor.id, []) if vendor else []

        size_estimate = None
        if report_item:
            size_estimate = report_item.lens_breakdown_json.get("size_estimate")
        if size_estimate is None and vendor:
            size_estimate = estimate_size_from_signals(
                dossier_json=dossier_json,
                facts=facts,
                evidence_items=evidence_by_vendor.get(vendor.id, []),
                tags_custom=vendor.tags_custom or [],
                why_relevant=vendor.why_relevant or [],
            )
        size_low, size_high = _size_range_from_claims(size_estimate, facts, vendor_claims)
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
        for claim in vendor_claims:
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
            claim for claim in vendor_claims
            if (claim.dimension or "").lower() in {"icp", "target_customer", "customers", "customer"}
        ]
        product_claims = [
            claim for claim in vendor_claims
            if (claim.dimension or "").lower() in {"capability", "product", "services", "evidence", "directory_context"}
        ]

        target_segments = []
        all_text = " ".join((claim.claim_text or "") for claim in vendor_claims).lower()
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

        thesis = (
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

        company_name = vendor.name if vendor else screening.candidate_name
        website = vendor.website if vendor else screening.candidate_website
        country = vendor.hq_country if vendor else (screening.screening_meta_json or {}).get("candidate_hq_country")
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
        if vendor:
            for tag in vendor.tags_custom or []:
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
                        for claim in vendor_claims
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
                        for claim in vendor_claims
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
                    "thesis": thesis,
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


@router.get("/{workspace_id}/vendors/{vendor_id}/decision", response_model=VendorDecisionResponse)
async def get_vendor_decision(
    workspace_id: int,
    vendor_id: int,
    db: AsyncSession = Depends(get_db),
):
    vendor_result = await db.execute(
        select(Vendor).where(Vendor.id == vendor_id, Vendor.workspace_id == workspace_id)
    )
    vendor = vendor_result.scalar_one_or_none()
    if not vendor:
        raise HTTPException(status_code=404, detail="Vendor not found")

    screening = await _latest_screening_for_vendor(db, workspace_id, vendor_id)
    if screening:
        return VendorDecisionResponse(
            vendor_id=vendor_id,
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
        select(VendorClaim).where(
            VendorClaim.workspace_id == workspace_id,
            VendorClaim.vendor_id == vendor_id,
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
    return VendorDecisionResponse(
        vendor_id=vendor_id,
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
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
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

    vendor_ids = [row.vendor_id for row in screenings if row.vendor_id]
    claims_result = await db.execute(
        select(VendorClaim).where(VendorClaim.workspace_id == workspace_id)
    )
    claims = claims_result.scalars().all()

    evidence_result = await db.execute(
        select(WorkspaceEvidence).where(WorkspaceEvidence.workspace_id == workspace_id)
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

    by_vendor: Dict[int, List[VendorScreening]] = {}
    for row in screenings:
        if row.vendor_id:
            by_vendor.setdefault(row.vendor_id, []).append(row)
    downgraded = 0
    for rows in by_vendor.values():
        statuses = [str(item.decision_classification or "") for item in rows]
        if "good_target" in statuses and "not_good_target" in statuses:
            downgraded += 1
    keep_to_later_reject_rate = round(downgraded / max(1, len(by_vendor)), 4)
    feedback_count_result = await db.execute(
        select(func.count(WorkspaceFeedbackEvent.id)).where(
            WorkspaceFeedbackEvent.workspace_id == workspace_id
        )
    )
    feedback_count = int(feedback_count_result.scalar() or 0)
    analyst_override_rate = round(feedback_count / max(1, len(by_vendor)), 4)
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
            "max_vendors": data.max_vendors,
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

    screening: Optional[VendorScreening] = None
    if data.vendor_id:
        vendor_result = await db.execute(
            select(Vendor).where(Vendor.id == data.vendor_id, Vendor.workspace_id == workspace_id)
        )
        vendor = vendor_result.scalar_one_or_none()
        if not vendor:
            raise HTTPException(status_code=404, detail="Vendor not found")
    if data.screening_id:
        screening_result = await db.execute(
            select(VendorScreening).where(
                VendorScreening.id == data.screening_id,
                VendorScreening.workspace_id == workspace_id,
            )
        )
        screening = screening_result.scalar_one_or_none()
    elif data.vendor_id:
        screening = await _latest_screening_for_vendor(db, workspace_id, data.vendor_id)

    feedback_event = WorkspaceFeedbackEvent(
        workspace_id=workspace_id,
        vendor_id=data.vendor_id,
        screening_id=screening.id if screening else data.screening_id,
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
                vendor_id=sample.get("vendor_id"),
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
        "brick_model": [],
        "universe": [],
        "segmentation": [],
        "enrichment": []
    }
    
    # Check context pack
    profile_result = await db.execute(
        select(CompanyProfile).where(CompanyProfile.workspace_id == workspace_id)
    )
    profile = profile_result.scalar_one_or_none()
    context_pack_ready = False
    if profile:
        context_claim_groups_available = set()
        if not profile.buyer_company_url:
            missing_items["context_pack"].append("Add your company URL")
        else:
            context_claim_groups_available.add("identity_scope")
        if not profile.context_pack_markdown:
            missing_items["context_pack"].append("Generate context pack")
        elif profile.reference_vendor_urls:
            context_claim_groups_available.add("vertical_workflow")
        if profile.product_pages_found < 3:
            missing_items["context_pack"].append(f"Need more product pages (found {profile.product_pages_found})")
        else:
            context_claim_groups_available.add("product_depth")

        required_groups = gate_cfg.get("context_pack", {}).get("required_claim_groups", ["identity_scope", "product_depth"])
        min_required = int(gate_cfg.get("context_pack", {}).get("min_required_groups_met", 2))
        covered = len([group for group in required_groups if group in context_claim_groups_available])
        if covered < min_required:
            missing_items["context_pack"].append(
                f"Evidence pattern coverage too low ({covered}/{len(required_groups)})"
            )
        context_pack_ready = bool(
            profile.context_pack_markdown
            and profile.product_pages_found >= 3
            and covered >= min_required
        )
    else:
        missing_items["context_pack"].append("Create company profile")
    
    # Check brick model
    taxonomy_result = await db.execute(
        select(BrickTaxonomy).where(BrickTaxonomy.workspace_id == workspace_id)
    )
    taxonomy = taxonomy_result.scalar_one_or_none()
    brick_model_ready = False
    if taxonomy:
        if not taxonomy.confirmed:
            missing_items["brick_model"].append("Confirm taxonomy")
        min_priority = int(gate_cfg.get("brick_model", {}).get("min_priority_bricks", 3))
        if len(taxonomy.priority_brick_ids or []) < min_priority:
            missing_items["brick_model"].append("Select at least 3 priority bricks")
        required_mapped = int(gate_cfg.get("brick_model", {}).get("require_evidence_mapped_bricks", 2))
        mapped_priority = 0
        if taxonomy.priority_brick_ids:
            vendors_result = await db.execute(
                select(Vendor).where(
                    Vendor.workspace_id == workspace_id,
                    Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched]),
                )
            )
            vendors = vendors_result.scalars().all()
            seen_bricks: set[str] = set()
            for vendor in vendors:
                dossier_result = await db.execute(
                    select(VendorDossier)
                    .where(VendorDossier.vendor_id == vendor.id)
                    .order_by(VendorDossier.version.desc())
                    .limit(1)
                )
                dossier = dossier_result.scalar_one_or_none()
                modules = (dossier.dossier_json or {}).get("modules", []) if dossier else []
                for module in modules:
                    brick_id = str(module.get("brick_id") or "").strip()
                    if brick_id and brick_id in set(taxonomy.priority_brick_ids or []):
                        urls = [url for url in module.get("evidence_urls", []) if is_trusted_source_url(url)]
                        if urls:
                            seen_bricks.add(brick_id)
            mapped_priority = len(seen_bricks)
        if mapped_priority < required_mapped:
            missing_items["brick_model"].append(
                f"Need evidence-backed priority brick mappings ({mapped_priority}/{required_mapped})"
            )
        brick_model_ready = (
            taxonomy.confirmed
            and len(taxonomy.priority_brick_ids or []) >= min_priority
            and mapped_priority >= required_mapped
        )
    else:
        missing_items["brick_model"].append("Create brick taxonomy")
    
    # Check universe with evidence-pattern decisions
    screenings_result = await db.execute(
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
    )
    screenings_all = screenings_result.scalars().all()
    latest_by_vendor: Dict[int, VendorScreening] = {}
    for row in screenings_all:
        if not row.vendor_id:
            continue
        if row.vendor_id not in latest_by_vendor:
            latest_by_vendor[row.vendor_id] = row

    universe_cfg = gate_cfg.get("universe", {})
    allowed_classes = set(universe_cfg.get("allowed_classes", ["good_target", "borderline_watchlist"]))
    min_decision_qualified = int(universe_cfg.get("min_decision_qualified_vendors", 5))
    max_insufficient_ratio = float(universe_cfg.get("max_insufficient_ratio", 0.5))

    decision_qualified = [
        row for row in latest_by_vendor.values()
        if str(row.decision_classification or "") in allowed_classes
    ]
    insufficient_count = len(
        [row for row in latest_by_vendor.values() if str(row.evidence_sufficiency or "") == "insufficient"]
    )
    insufficient_ratio = insufficient_count / max(1, len(latest_by_vendor))

    kept_vendors_result = await db.execute(
        select(func.count(Vendor.id)).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched]),
        )
    )
    kept_vendors_count = kept_vendors_result.scalar() or 0
    universe_ready = (
        len(decision_qualified) >= min_decision_qualified
        and insufficient_ratio <= max_insufficient_ratio
    )
    if len(decision_qualified) < min_decision_qualified:
        missing_items["universe"].append(
            f"Need decision-qualified vendors ({len(decision_qualified)}/{min_decision_qualified})"
        )
    if insufficient_ratio > max_insufficient_ratio:
        missing_items["universe"].append(
            f"Evidence insufficiency ratio too high ({round(insufficient_ratio, 2)} > {max_insufficient_ratio})"
        )
    # Legacy fallback messaging (one release cycle)
    if kept_vendors_count < 5:
        missing_items["universe"].append(f"Keep at least 5 vendors ({kept_vendors_count} kept)")
    
    # Check segmentation (has reviewed and focused)
    segmentation_ready = universe_ready and len(decision_qualified) >= 10
    if not segmentation_ready and universe_ready:
        missing_items["segmentation"].append("Review and keep at least 10 vendors")
    
    # Check enrichment
    enrichment_cfg = gate_cfg.get("enrichment", {})
    min_enriched = int(enrichment_cfg.get("min_enriched_vendors", 5))
    required_groups = set(enrichment_cfg.get("required_groups_per_vendor", ["product_depth", "traction"]))

    enriched_result = await db.execute(
        select(func.count(Vendor.id)).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status == VendorStatus.enriched
        )
    )
    enriched_count = enriched_result.scalar() or 0

    enriched_vendors_result = await db.execute(
        select(Vendor).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status == VendorStatus.enriched,
        )
    )
    enriched_vendors = enriched_vendors_result.scalars().all()
    enriched_with_groups = 0
    for vendor in enriched_vendors:
        screening = latest_by_vendor.get(vendor.id)
        missing_groups = set(screening.missing_claim_groups_json or []) if screening else set(required_groups)
        if required_groups.isdisjoint(missing_groups):
            enriched_with_groups += 1

    enrichment_ready = enriched_count >= min_enriched and enriched_with_groups >= min_enriched
    if not enrichment_ready:
        missing_items["enrichment"].append(
            f"Enrich at least {min_enriched} vendors with required evidence groups ({enriched_with_groups}/{min_enriched})"
        )
    
    return GatesResponse(
        context_pack=context_pack_ready,
        brick_model=brick_model_ready,
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
            vendor_id=job.vendor_id,
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
        vendor_id=job.vendor_id,
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
