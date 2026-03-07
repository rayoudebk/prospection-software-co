#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import quantiles
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {path}")
    return [row for row in payload if isinstance(row, dict)]


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(quantiles(values, n=100)[94])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _variance_hotspots(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = {
        "first_party_hint_urls_used_count": lambda row: row.get("first_party_hint_urls_used_count"),
        "first_party_hint_pages_crawled_total": lambda row: row.get("first_party_hint_pages_crawled_total"),
        "first_party_crawl_pages_total": lambda row: row.get("first_party_crawl_pages_total"),
        "stage_llm_discovery_fanout.llm_ms": lambda row: (
            (row.get("stage_time_ms") or {}).get("stage_llm_discovery_fanout")
            if isinstance(row.get("stage_time_ms"), dict)
            else None
        ),
        "ranking_eligible_count": lambda row: row.get("ranking_eligible_count"),
    }
    hotspots: List[Dict[str, Any]] = []
    for metric, extractor in metrics.items():
        values = [
            parsed
            for parsed in (
                _safe_float(extractor(row))
                for row in rows
                if isinstance(row, dict)
            )
            if parsed is not None
        ]
        if not values:
            continue
        avg = sum(values) / len(values)
        variance = sum((value - avg) ** 2 for value in values) / max(1, len(values))
        hotspots.append(
            {
                "metric": metric,
                "min": min(values),
                "max": max(values),
                "avg": round(avg, 4),
                "stddev": round(variance ** 0.5, 4),
                "run_count": len(values),
            }
        )
    hotspots.sort(key=lambda row: float(row.get("stddev") or 0.0), reverse=True)
    return hotspots


def build_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [
        float((row.get("stage_time_ms") or {}).get("pipeline_total_ms") or 0.0)
        for row in rows
        if isinstance(row.get("stage_time_ms"), dict)
    ]
    high_quality = len([row for row in rows if str(row.get("run_quality_tier")) == "high_quality"])
    degraded = len([row for row in rows if str(row.get("run_quality_tier")) == "degraded"])
    ranking_counts = [int(row.get("ranking_eligible_count") or 0) for row in rows]
    citation_counts = [int(row.get("citation_sentence_count") or 0) for row in rows]
    quality_audits = [
        row.get("quality_audit_v1")
        for row in rows
        if isinstance(row.get("quality_audit_v1"), dict)
    ]
    quality_audit_pass_runs = len(
        [
            audit
            for audit in quality_audits
            if bool(audit.get("pass"))
        ]
    )
    quality_audit_pattern_totals: Dict[str, int] = {}
    for audit in quality_audits:
        for pattern in (audit.get("patterns") or []):
            if not isinstance(pattern, dict):
                continue
            key = str(pattern.get("pattern_key") or "").strip()
            if not key:
                continue
            quality_audit_pattern_totals[key] = quality_audit_pattern_totals.get(key, 0) + int(pattern.get("count") or 0)
    return {
        "runs": len(rows),
        "runtime_ms": {
            "avg": (sum(runtimes) / len(runtimes)) if runtimes else 0.0,
            "p95": _p95(runtimes),
            "max": max(runtimes) if runtimes else 0.0,
        },
        "quality": {
            "high_quality_runs": high_quality,
            "degraded_runs": degraded,
            "high_quality_rate": (high_quality / len(rows)) if rows else 0.0,
        },
        "ranking": {
            "avg_ranking_eligible_count": (sum(ranking_counts) / len(ranking_counts)) if ranking_counts else 0.0,
            "avg_citation_sentence_count": (sum(citation_counts) / len(citation_counts)) if citation_counts else 0.0,
        },
        "quality_audit": {
            "audited_runs": len(quality_audits),
            "pass_runs": quality_audit_pass_runs,
            "pass_rate": (quality_audit_pass_runs / len(quality_audits)) if quality_audits else 0.0,
            "pattern_totals": quality_audit_pattern_totals,
        },
        "variance_hotspots_v1": _variance_hotspots(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark metrics from replay corpus.")
    parser.add_argument("--corpus", default="backend/benchmarks/corpus/sample_replay.json")
    parser.add_argument("--out", default="backend/benchmarks/latest_metrics.json")
    args = parser.parse_args()

    rows = _load_rows(Path(args.corpus))
    metrics = build_metrics(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
