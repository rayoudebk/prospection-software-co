from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class JobType(enum.Enum):
    context_pack = "context_pack"
    landscape = "landscape"
    deep_profile = "deep_profile"


class JobState(enum.Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class ResearchJob(Base):
    __tablename__ = "research_jobs"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    target_id = Column(Integer, ForeignKey("targets.id"), nullable=True)
    job_type = Column(Enum(JobType), nullable=False)
    state = Column(Enum(JobState), default=JobState.queued)
    
    # For Gemini Interactions API (future)
    interaction_id = Column(String(255), nullable=True)
    
    # Results
    result_json = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    # Relationships
    strategy = relationship("Strategy", back_populates="jobs")
    target = relationship("Target", back_populates="jobs")

