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

    quality_drop_pct = _pct_delta(base_high_quality_rate, cand_high_quality_rate) * -1.0
    p95_regression_pct = _pct_delta(base_p95, cand_p95)
    max_quality_drop = float(thresholds.get("max_quality_drop_pct") or 0.0)
    max_runtime_regression = float(thresholds.get("max_p95_runtime_regression_pct") or 0.0)

    quality_ok = quality_drop_pct <= max_quality_drop
    runtime_ok = p95_regression_pct <= max_runtime_regression

    report = {
        "baseline": baseline,
        "candidate": candidate,
        "diff": {
            "quality_drop_pct": quality_drop_pct,
            "p95_runtime_regression_pct": p95_regression_pct,
        },
        "thresholds": thresholds,
        "gate_passed": bool(quality_ok and runtime_ok),
        "quality_gate_passed": bool(quality_ok),
        "runtime_gate_passed": bool(runtime_ok),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
