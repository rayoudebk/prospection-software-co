from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class LLMStage(str, Enum):
    discovery_retrieval = "discovery_retrieval"
    evidence_adjudication = "evidence_adjudication"
    structured_normalization = "structured_normalization"
    context_summary = "context_summary"
    crawler_triage = "crawler_triage"


@dataclass
class LLMRequest:
    stage: LLMStage
    prompt: str
    timeout_seconds: int = 60
    use_web_search: bool = False
    expect_json: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelAttemptTrace:
    stage: str
    provider: str
    model: str
    latency_ms: int
    status: str
    retry_count: int
    error_class: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    ended_at: Optional[str] = None


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    attempts: List[ModelAttemptTrace] = field(default_factory=list)


class LLMOrchestrationError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempts: Optional[List[ModelAttemptTrace]] = None,
    ) -> None:
        super().__init__(message)
        self.attempts = attempts or []


class LLMProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


def classify_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if "rate limit" in text or "429" in text:
        return True
    if "timeout" in text or "timed out" in text:
        return True
    if "connection reset" in text or "connection aborted" in text:
        return True
    if "502" in text or "503" in text or "504" in text:
        return True
    return False


def now_iso() -> str:
    return datetime.utcnow().isoformat()
