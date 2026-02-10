from app.models.base import Base
from app.models.strategy import Strategy
from app.models.target import Target
from app.models.research_job import ResearchJob
from app.models.evidence import EvidenceItem
from app.models.feedback import FeedbackEvent

# New workspace-based models
from app.models.workspace import Workspace, CompanyProfile, BrickTaxonomy, BrickMapping, DEFAULT_BRICKS
from app.models.vendor import Vendor, VendorDossier, VendorStatus
from app.models.job import Job, JobType, JobState, JobProvider
from app.models.workspace_evidence import WorkspaceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, VendorFact
from app.models.intelligence import (
    ComparatorSourceRun,
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    RegistryQueryLog,
    VendorMention,
    VendorScreening,
    VendorClaim,
)

__all__ = [
    # Legacy models (kept for backwards compat)
    "Base", "Strategy", "Target", "ResearchJob", "EvidenceItem", "FeedbackEvent",
    # New workspace models
    "Workspace", "CompanyProfile", "BrickTaxonomy", "BrickMapping", "DEFAULT_BRICKS",
    "Vendor", "VendorDossier", "VendorStatus",
    "Job", "JobType", "JobState", "JobProvider",
    "WorkspaceEvidence",
    "ReportSnapshot", "ReportSnapshotItem", "VendorFact",
    "ComparatorSourceRun", "CandidateEntity", "CandidateEntityAlias", "CandidateOriginEdge", "RegistryQueryLog",
    "VendorMention", "VendorScreening", "VendorClaim",
]
