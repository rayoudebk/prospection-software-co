from app.models.base import Base
from app.models.strategy import Strategy
from app.models.target import Target
from app.models.research_job import ResearchJob
from app.models.evidence import EvidenceItem
from app.models.feedback import FeedbackEvent

# New workspace-based models
from app.models.workspace import Workspace, CompanyProfile
from app.models.company_context import CompanyContextPack
from app.models.company import Company, CompanyDossier, CompanyStatus
from app.models.job import Job, JobType, JobState, JobProvider
from app.models.source_evidence import SourceEvidence
from app.models.report import ReportSnapshot, ReportSnapshotItem, CompanyFact
from app.models.workspace_feedback import WorkspaceFeedbackEvent
from app.models.intelligence import (
    ComparatorSourceRun,
    CandidateEntity,
    CandidateEntityAlias,
    CandidateOriginEdge,
    RegistryQueryLog,
    CompanyMention,
    CompanyScreening,
    CompanyClaim,
)
from app.models.claims_graph import ClaimGraphNode, ClaimGraphEdge, ClaimGraphEdgeEvidence
from app.models.evaluation import EvaluationRun, EvaluationSampleResult
from app.models.external_search import ExternalSearchRun, ExternalSearchResult

__all__ = [
    # Legacy models (kept for backwards compat)
    "Base", "Strategy", "Target", "ResearchJob", "EvidenceItem", "FeedbackEvent",
    # New workspace models
    "Workspace", "CompanyProfile",
    "CompanyContextPack",
    "Company", "CompanyDossier", "CompanyStatus",
    "Job", "JobType", "JobState", "JobProvider",
    "SourceEvidence",
    "ReportSnapshot", "ReportSnapshotItem", "CompanyFact",
    "WorkspaceFeedbackEvent",
    "ComparatorSourceRun", "CandidateEntity", "CandidateEntityAlias", "CandidateOriginEdge", "RegistryQueryLog",
    "CompanyMention", "CompanyScreening", "CompanyClaim",
    "ClaimGraphNode", "ClaimGraphEdge", "ClaimGraphEdgeEvidence",
    "EvaluationRun", "EvaluationSampleResult",
    "ExternalSearchRun", "ExternalSearchResult",
]
