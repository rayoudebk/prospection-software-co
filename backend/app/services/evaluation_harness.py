"""Gold-set replay harness utilities."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List

from sqlalchemy.orm import Session

from app.models.evaluation import EvaluationRun, EvaluationSampleResult


def run_gold_set_replay(
    db: Session,
    workspace_id: int,
    samples: Iterable[Dict[str, Any]],
    model_version: str,
) -> Dict[str, Any]:
    sample_rows: List[Dict[str, Any]] = [row for row in samples if isinstance(row, dict)]
    run = EvaluationRun(
        workspace_id=workspace_id,
        run_type="gold_set_replay",
        status="completed",
        model_version=model_version,
        metrics_json={},
    )
    db.add(run)
    db.flush()

    total = 0
    matched = 0
    for sample in sample_rows:
        expected = str(sample.get("expected_classification") or "")
        predicted = str(sample.get("predicted_classification") or "")
        ok = int(bool(expected and predicted and expected == predicted))
        total += 1
        matched += ok
        db.add(
            EvaluationSampleResult(
                run_id=run.id,
                vendor_id=sample.get("vendor_id"),
                expected_classification=expected or None,
                predicted_classification=predicted or None,
                matched=ok,
                confidence=float(sample.get("confidence") or 0.0),
                details_json=sample.get("details_json") if isinstance(sample.get("details_json"), dict) else {},
            )
        )

    precision_proxy = (matched / total) if total else 0.0
    run.metrics_json = {
        "samples_total": total,
        "samples_matched": matched,
        "precision_proxy": round(precision_proxy, 4),
    }
    db.flush()
    return {
        "run_id": run.id,
        "metrics": run.metrics_json,
    }

