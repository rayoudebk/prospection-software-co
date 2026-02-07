from sqlalchemy import Column, Integer, String, DateTime, JSON, Text, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from app.models.base import Base


class TargetStatus(enum.Enum):
    candidate = "candidate"
    shortlisted = "shortlisted"
    rejected = "rejected"
    watching = "watching"


class Target(Base):
    __tablename__ = "targets"

    id = Column(Integer, primary_key=True, index=True)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    name = Column(String(255), nullable=False)
    website = Column(String(500), nullable=True)
    country = Column(String(100), nullable=True)
    registry_ids = Column(JSON, nullable=True)  # {siren: "...", company_number: "..."}
    status = Column(Enum(TargetStatus), default=TargetStatus.candidate)
    
    # Seed marker - True if this company was provided as a seed URL
    is_seed = Column(Boolean, default=False)
    
    # Scoring
    bpo_score = Column(Integer, nullable=True)  # 0-100
    bpo_rationale = Column(Text, nullable=True)
    fit_score = Column(Integer, nullable=True)  # 0-100
    fit_rationale = Column(Text, nullable=True)
    
    # AI-generated comparison insights
    similarities = Column(Text, nullable=True)  # What this company has in common with seed companies
    watchouts = Column(Text, nullable=True)  # Differences/risks compared to seed companies
    
    # Deep profile
    profile_markdown = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    strategy = relationship("Strategy", back_populates="targets")
    evidence = relationship("EvidenceItem", back_populates="target", cascade="all, delete-orphan")
    jobs = relationship("ResearchJob", back_populates="target", cascade="all, delete-orphan")
    feedback_events = relationship("FeedbackEvent", back_populates="target", cascade="all, delete-orphan")

