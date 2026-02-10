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
from app.services.reporting import (
    RELIABLE_FILINGS_COUNTRIES,
    build_adjacency_map,
    classify_size_bucket,
    compute_lens_scores,
    estimate_size_from_signals,
    extract_customers_and_integrations,
    is_trusted_source_url,
    modules_with_evidence,
    normalize_country,
    source_label_for_url,
)

router = APIRouter()


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
    geo_scope: Optional[GeoScope] = None


class CompanyProfileResponse(BaseModel):
    id: int
    workspace_id: int
    buyer_company_url: Optional[str]
    buyer_context_summary: Optional[str]
    reference_vendor_urls: List[str]
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
    hq_country: Optional[str]
    operating_countries: List[str]
    tags_vertical: List[str]
    tags_custom: List[str]
    status: str
    why_relevant: List[Dict[str, Any]]
    is_manual: bool
    created_at: datetime
    evidence_count: int = 0

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
        region_scope=data.region_scope
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
        profile.reference_vendor_urls = data.reference_vendor_urls
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

    screenings_result = await db.execute(
        select(VendorScreening)
        .where(VendorScreening.workspace_id == workspace_id)
        .order_by(VendorScreening.created_at.desc())
        .limit(1000)
    )
    screenings = screenings_result.scalars().all()

    latest_run_id = None
    if screenings:
        latest_run_id = (screenings[0].screening_meta_json or {}).get("screening_run_id")

    if latest_run_id:
        screenings = [
            screening
            for screening in screenings
            if (screening.screening_meta_json or {}).get("screening_run_id") == latest_run_id
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
        },
        "origin_mix_distribution": latest_discovery_job_result.get("origin_mix_distribution", origin_mix_distribution),
        "dedupe_quality_metrics": latest_discovery_job_result.get("dedupe_quality_metrics", {}),
        "registry_expansion_yield": latest_discovery_job_result.get("registry_expansion_yield", {}),
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
    
    responses = []
    for v in vendors:
        # Get evidence count
        evidence_count_result = await db.execute(
            select(func.count(WorkspaceEvidence.id)).where(WorkspaceEvidence.vendor_id == v.id)
        )
        evidence_count = evidence_count_result.scalar() or 0
        
        responses.append(VendorResponse(
            id=v.id,
            workspace_id=v.workspace_id,
            name=v.name,
            website=v.website,
            hq_country=v.hq_country,
            operating_countries=v.operating_countries or [],
            tags_vertical=v.tags_vertical or [],
            tags_custom=v.tags_custom or [],
            status=v.status.value,
            why_relevant=v.why_relevant or [],
            is_manual=v.is_manual,
            created_at=v.created_at,
            evidence_count=evidence_count
        ))
    
    return responses


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
    
    return VendorResponse(
        id=vendor.id,
        workspace_id=vendor.workspace_id,
        name=vendor.name,
        website=vendor.website,
        hq_country=vendor.hq_country,
        operating_countries=vendor.operating_countries or [],
        tags_vertical=vendor.tags_vertical or [],
        tags_custom=vendor.tags_custom or [],
        status=vendor.status.value,
        why_relevant=vendor.why_relevant or [],
        is_manual=vendor.is_manual,
        created_at=vendor.created_at,
        evidence_count=0
    )


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
    
    evidence_count_result = await db.execute(
        select(func.count(WorkspaceEvidence.id)).where(WorkspaceEvidence.vendor_id == vendor_id)
    )
    evidence_count = evidence_count_result.scalar() or 0
    
    return VendorResponse(
        id=vendor.id,
        workspace_id=vendor.workspace_id,
        name=vendor.name,
        website=vendor.website,
        hq_country=vendor.hq_country,
        operating_countries=vendor.operating_countries or [],
        tags_vertical=vendor.tags_vertical or [],
        tags_custom=vendor.tags_custom or [],
        status=vendor.status.value,
        why_relevant=vendor.why_relevant or [],
        is_manual=vendor.is_manual,
        created_at=vendor.created_at,
        evidence_count=evidence_count
    )


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
            reason_text = str(reason.get("text") or "").strip()
            source_url = reason.get("citation_url")
            if not reason_text:
                continue
            customer_claims.append(
                _build_claim(
                    text=f"Discovery rationale: {reason_text}",
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
        reject_reasons = screening.reject_reasons_json or []
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

        companies.append(
            {
                "identity": {
                    "name": company_name,
                    "website": website,
                    "official_website": identity_meta.get("official_website") or website,
                    "input_website": identity_meta.get("input_website"),
                    "identity_confidence": identity_meta.get("identity_confidence"),
                    "identity_sources": identity_meta.get("identity_sources") or [],
                    "country": country,
                    "legal_entity_hints": legal_hint,
                    "entity_id": candidate_entity.id if candidate_entity else None,
                },
                "screening": {
                    "status": screening.screening_status,
                    "total_score": total_score,
                    "component_scores": component_scores,
                    "penalties": penalties,
                    "reason_codes": reject_reasons,
                    "reject_reasons": reject_reasons,
                    "evidence_mix": source_summary.get("source_type_counts") or {},
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
# Gates
# ============================================================================

@router.get("/{workspace_id}/gates", response_model=GatesResponse)
async def get_gates(workspace_id: int, db: AsyncSession = Depends(get_db)):
    """Get gating status for the workspace - what's unlocked."""
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
        if not profile.buyer_company_url:
            missing_items["context_pack"].append("Add your company URL")
        if not profile.context_pack_markdown:
            missing_items["context_pack"].append("Generate context pack")
        if profile.product_pages_found < 3:
            missing_items["context_pack"].append(f"Need more product pages (found {profile.product_pages_found})")
        context_pack_ready = bool(profile.context_pack_markdown and profile.product_pages_found >= 3)
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
        if len(taxonomy.priority_brick_ids or []) < 3:
            missing_items["brick_model"].append("Select at least 3 priority bricks")
        brick_model_ready = taxonomy.confirmed and len(taxonomy.priority_brick_ids or []) >= 3
    else:
        missing_items["brick_model"].append("Create brick taxonomy")
    
    # Check universe
    kept_vendors_result = await db.execute(
        select(func.count(Vendor.id)).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status.in_([VendorStatus.kept, VendorStatus.enriched])
        )
    )
    kept_vendors_count = kept_vendors_result.scalar() or 0
    
    universe_ready = False
    if kept_vendors_count < 5:
        missing_items["universe"].append(f"Keep at least 5 vendors ({kept_vendors_count} kept)")
    else:
        universe_ready = True
    
    # Check segmentation (has reviewed and focused)
    segmentation_ready = universe_ready and kept_vendors_count >= 10
    if not segmentation_ready and universe_ready:
        missing_items["segmentation"].append("Review and keep at least 10 vendors")
    
    # Check enrichment
    enriched_result = await db.execute(
        select(func.count(Vendor.id)).where(
            Vendor.workspace_id == workspace_id,
            Vendor.status == VendorStatus.enriched
        )
    )
    enriched_count = enriched_result.scalar() or 0
    
    enrichment_ready = enriched_count >= 5
    if not enrichment_ready:
        missing_items["enrichment"].append(f"Enrich at least 5 vendors ({enriched_count} enriched)")
    
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
