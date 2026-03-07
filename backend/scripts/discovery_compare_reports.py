#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _pct_delta(old: float, new: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return ((new - old) / old) * 100.0


def _hotspot_stddev_map(payload: Dict[str, Any]) -> Dict[str, float]:
    rows = payload.get("variance_hotspots_v1")
    if not isinstance(rows, list):
        return {}
    output: Dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "").strip()
        if not metric:
            continue
        try:
            output[metric] = float(row.get("stddev") or 0.0)
        except Exception:
            continue
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare discovery benchmark metrics against baseline.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--thresholds", default="backend/benchmarks/thresholds.json")
    parser.add_argument("--out", default="backend/benchmarks/compare_report.json")
    args = parser.parse_args()

    baseline = _load_json(Path(args.baseline))
    candidate = _load_json(Path(args.candidate))
    thresholds = _load_json(Path(args.thresholds))

    base_high_quality_rate = float(((baseline.get("quality") or {}).get("high_quality_rate") or 0.0))
    cand_high_quality_rate = float(((candidate.get("quality") or {}).get("high_quality_rate") or 0.0))
    base_p95 = float(((baseline.get("runtime_ms") or {}).get("p95") or 0.0))
    cand_p95 = float(((candidate.get("runtime_ms") or {}).get("p95") or 0.0))
    base_quality_audit_pass_rate = float(((baseline.get("quality_audit") or {}).get("pass_rate") or 0.0))
    cand_quality_audit_pass_rate = float(((candidate.get("quality_audit") or {}).get("pass_rate") or 0.0))

    quality_drop_pct = _pct_delta(base_high_quality_rate, cand_high_quality_rate) * -1.0
    p95_regression_pct = _pct_delta(base_p95, cand_p95)
    max_quality_drop = float(thresholds.get("max_quality_drop_pct") or 0.0)
    max_runtime_regression = float(thresholds.get("max_p95_runtime_regression_pct") or 0.0)
    default_variance_regression = float(thresholds.get("max_variance_stddev_regression_pct") or 0.0)
    per_metric_variance_regression = (
        thresholds.get("max_variance_stddev_regression_pct_by_metric")
        if isinstance(thresholds.get("max_variance_stddev_regression_pct_by_metric"), dict)
        else {}
    )

    quality_ok = quality_drop_pct <= max_quality_drop
    runtime_ok = p95_regression_pct <= max_runtime_regression

    baseline_hotspots = _hotspot_stddev_map(baseline)
    candidate_hotspots = _hotspot_stddev_map(candidate)
    hotspot_stddev_delta: Dict[str, float] = {}
    hotspot_stddev_regression_pct: Dict[str, float] = {}
    variance_regressions: Dict[str, Dict[str, float]] = {}
    variance_ok = True
    for metric in sorted(set(baseline_hotspots.keys()) | set(candidate_hotspots.keys())):
        base_stddev = float(baseline_hotspots.get(metric, 0.0))
        cand_stddev = float(candidate_hotspots.get(metric, 0.0))
        hotspot_stddev_delta[metric] = round(cand_stddev - base_stddev, 4)
        regression_pct = _pct_delta(base_stddev, cand_stddev)
        hotspot_stddev_regression_pct[metric] = round(regression_pct, 4)
        threshold = float(per_metric_variance_regression.get(metric, default_variance_regression))
        if regression_pct > threshold:
            variance_ok = False
            variance_regressions[metric] = {
                "baseline_stddev": round(base_stddev, 4),
                "candidate_stddev": round(cand_stddev, 4),
                "regression_pct": round(regression_pct, 4),
                "threshold_pct": round(threshold, 4),
            }

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "diff": {
            "quality_drop_pct": quality_drop_pct,
            "p95_runtime_regression_pct": p95_regression_pct,
            "quality_audit_pass_rate_delta_pct": _pct_delta(base_quality_audit_pass_rate, cand_quality_audit_pass_rate),
            "variance_hotspot_stddev_delta": hotspot_stddev_delta,
            "variance_hotspot_stddev_regression_pct": hotspot_stddev_regression_pct,
        },
        "thresholds": thresholds,
        "gate_passed": bool(quality_ok and runtime_ok and variance_ok),
        "quality_gate_passed": bool(quality_ok),
        "runtime_gate_passed": bool(runtime_ok),
        "variance_gate_passed": bool(variance_ok),
        "variance_regressions": variance_regressions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
