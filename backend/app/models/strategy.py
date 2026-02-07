from sqlalchemy import Column, Integer, String, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from datetime import datetime

from app.models.base import Base


class Strategy(Base):
    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    region_scope = Column(String(100), default="EU+UK")
    intent = Column(String(100), default="capability")  # capability, vertical, geo
    seed_urls = Column(JSON, default=list)
    exclusions = Column(JSON, default=dict)  # bpo_toggle, keywords, industries
    context_pack = Column(Text, nullable=True)  # Generated markdown from seed URLs
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    targets = relationship("Target", back_populates="strategy", cascade="all, delete-orphan")
    jobs = relationship("ResearchJob", back_populates="strategy", cascade="all, delete-orphan")
    feedback_events = relationship("FeedbackEvent", back_populates="strategy", cascade="all, delete-orphan")

