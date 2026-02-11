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
