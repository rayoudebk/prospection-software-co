#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.models.job import Job, JobState, JobType
from app.models.intelligence import CompanyClaim, CompanyScreening
from app.services.quality_audit import (
    DEFAULT_QUALITY_AUDIT_THRESHOLDS,
    QUALITY_AUDIT_PATTERN_ORDER,
    build_quality_audit_v1,
    normalize_quality_audit_v1,
    quality_audit_thresholds_from_settings,
)

ENV_THRESHOLD_KEY_MAP = {
    "fp_low_ticket_without_pricing_evidence": "audit_max_fp_low_ticket_without_pricing_evidence",
    "fn_missing_vertical_with_institutional_workflow_text": "audit_max_fn_missing_vertical_with_institutional_text",
    "fp_registry_or_directory_overweight": "audit_max_fp_registry_or_directory_overweight",
    "fn_customer_proof_present_but_thin_grouping": "audit_max_fn_customer_proof_but_thin_grouping",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _resolve_thresholds(
    settings: Any,
    thresholds_path: Optional[str],
) -> Dict[str, int]:
    thresholds = quality_audit_thresholds_from_settings(settings)
    if not thresholds_path:
        return thresholds
    payload = _load_json(Path(thresholds_path))
    if not isinstance(payload, dict):
        return thresholds
    merged = dict(thresholds)
    for pattern_key in QUALITY_AUDIT_PATTERN_ORDER:
        if pattern_key in payload:
            try:
                merged[pattern_key] = int(payload.get(pattern_key))
            except Exception:
                continue
            continue
        env_like = ENV_THRESHOLD_KEY_MAP.get(pattern_key)
        if env_like and env_like in payload:
            try:
                merged[pattern_key] = int(payload.get(env_like))
            except Exception:
                continue
            continue
        default_from_file = payload.get("quality_audit_thresholds", {})
        if isinstance(default_from_file, dict) and pattern_key in default_from_file:
            try:
                merged[pattern_key] = int(default_from_file.get(pattern_key))
            except Exception:
                continue
    return merged


def _screening_run_id_from_screening(row: CompanyScreening) -> str:
    meta = row.screening_meta_json if isinstance(row.screening_meta_json, dict) else {}
    return str(meta.get("screening_run_id") or "").strip()


def _resolve_run_id(session, workspace_id: int, explicit_run_id: Optional[str]) -> Optional[str]:
    if explicit_run_id:
        return str(explicit_run_id).strip()
    latest_job = (
        session.query(Job)
        .filter(
            Job.workspace_id == workspace_id,
            Job.job_type == JobType.discovery_universe,
            Job.state == JobState.completed,
        )
        .order_by(Job.finished_at.desc(), Job.created_at.desc())
        .first()
    )
    if latest_job and isinstance(latest_job.result_json, dict):
        run_id = str(latest_job.result_json.get("screening_run_id") or "").strip()
        if run_id:
            return run_id

    latest_screening = (
        session.query(CompanyScreening)
        .filter(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .first()
    )
    if latest_screening:
        run_id = _screening_run_id_from_screening(latest_screening)
        if run_id:
            return run_id
    return None


def _collect_workspace_run(
    session,
    workspace_id: int,
    run_id: str,
) -> tuple[List[CompanyScreening], Dict[int, List[CompanyClaim]]]:
    screenings = (
        session.query(CompanyScreening)
        .filter(CompanyScreening.workspace_id == workspace_id)
        .order_by(CompanyScreening.created_at.desc())
        .limit(5000)
        .all()
    )
    run_screenings = [row for row in screenings if _screening_run_id_from_screening(row) == run_id]
    screening_ids = [row.id for row in run_screenings if row.id]
    if not screening_ids:
        return run_screenings, {}
    claims = session.query(CompanyClaim).filter(CompanyClaim.company_screening_id.in_(screening_ids)).all()
    claims_by_screening: Dict[int, List[CompanyClaim]] = {}
    for claim in claims:
        if claim.company_screening_id is None:
            continue
        claims_by_screening.setdefault(int(claim.company_screening_id), []).append(claim)
    return run_screenings, claims_by_screening


def _report_for_workspace_mode(
    workspace_id: int,
    run_id: str,
    thresholds: Mapping[str, int],
) -> Dict[str, Any]:
    settings = get_settings()
    engine = create_engine(settings.database_url_sync)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        run_screenings, claims_by_screening = _collect_workspace_run(
            session=session,
            workspace_id=workspace_id,
            run_id=run_id,
        )
        audit = build_quality_audit_v1(
            screenings=run_screenings,
            claims_by_screening=claims_by_screening,
            run_id=run_id,
            thresholds=thresholds,
        )
        audit = normalize_quality_audit_v1(audit) or audit
        return {
            "mode": "workspace",
            "workspace_id": workspace_id,
            "run_id": run_id,
            "pass": bool(audit.get("pass")),
            "screenings_in_run": len(run_screenings),
            "quality_audit_v1": audit,
            "generated_at": datetime.utcnow().isoformat(),
        }
    finally:
        session.close()


def _report_for_corpus_mode(
    corpus_path: str,
    thresholds: Mapping[str, int],
) -> Dict[str, Any]:
    payload = _load_json(Path(corpus_path))
    if not isinstance(payload, list):
        raise ValueError("Corpus file must be a JSON array")

    rows = [row for row in payload if isinstance(row, dict)]
    audits: List[Dict[str, Any]] = []
    missing_rows: List[Dict[str, Any]] = []
    failed_runs: List[str] = []
    pattern_totals = {key: 0 for key in QUALITY_AUDIT_PATTERN_ORDER}

    for index, row in enumerate(rows):
        run_id = str(
            row.get("screening_run_id")
            or row.get("run_id")
            or f"row_{index}"
        ).strip()
        normalized = normalize_quality_audit_v1(row.get("quality_audit_v1"))
        if not normalized:
            missing_rows.append(
                {
                    "index": index,
                    "run_id": run_id,
                }
            )
            continue

        adjusted_payload = {
            **normalized,
            "thresholds": dict(thresholds),
        }
        adjusted = normalize_quality_audit_v1(adjusted_payload) or adjusted_payload
        audits.append(adjusted)
        if not bool(adjusted.get("pass")):
            failed_runs.append(str(adjusted.get("run_id") or run_id))
        for pattern in adjusted.get("patterns", []):
            if not isinstance(pattern, dict):
                continue
            key = str(pattern.get("pattern_key") or "").strip()
            if key not in pattern_totals:
                continue
            pattern_totals[key] += int(pattern.get("count") or 0)

    return {
        "mode": "corpus",
        "corpus_path": corpus_path,
        "rows": len(rows),
        "audited_rows": len(audits),
        "missing_quality_audit_rows": missing_rows,
        "failed_run_ids": failed_runs,
        "pass": len(missing_rows) == 0 and len(failed_runs) == 0,
        "thresholds": dict(thresholds),
        "pattern_totals": pattern_totals,
        "quality_audits": audits[:100],
        "generated_at": datetime.utcnow().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run discovery quality audit for a workspace run or replay corpus.")
    parser.add_argument("--workspace-id", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--corpus")
    parser.add_argument("--thresholds")
    parser.add_argument("--out", default="backend/benchmarks/quality_audit_report.json")
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    if not args.workspace_id and not args.corpus:
        raise SystemExit("Provide either --workspace-id or --corpus")

    settings = get_settings()
    thresholds = _resolve_thresholds(settings, args.thresholds)
    for key, default_value in DEFAULT_QUALITY_AUDIT_THRESHOLDS.items():
        thresholds.setdefault(key, default_value)

    if args.corpus:
        report = _report_for_corpus_mode(args.corpus, thresholds)
    else:
        engine = create_engine(settings.database_url_sync)
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        try:
            run_id = _resolve_run_id(session, int(args.workspace_id), args.run_id)
        finally:
            session.close()
        if not run_id:
            raise SystemExit("Unable to resolve screening run id for workspace")
        report = _report_for_workspace_mode(int(args.workspace_id), run_id, thresholds)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if args.fail_on_violation and not bool(report.get("pass")):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
