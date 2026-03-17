#!/usr/bin/env python3
"""Offline structural evaluation of expansion brief versions against saved raw reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.company_context import normalize_expansion_brief  # noqa: E402


DEFAULT_CASES = {
    "hublo": {
        "workspace_id": 9,
        "research_report": "tmp/expansion-ab-v3/workspace-9/gemini-stage1-report.md",
        "v2_artifact": "tmp/expansion-ab-v3/workspace-9/gemini-stage1-normalized.json",
        "v3_artifact": "tmp/expansion-ab-v4/workspace-9/gemini-stage1-v3-normalized.json",
    },
    "4tpm": {
        "workspace_id": 7,
        "research_report": "tmp/expansion-ab-v5/workspace-7/gemini-stage1-report.md",
        "v2_artifact": "tmp/expansion-ab-v5/workspace-7/gemini-stage1-normalized-v2.json",
        "v3_artifact": "tmp/expansion-ab-v5/workspace-7/gemini-stage1-normalized-v3.json",
    },
}

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
ITEM_RE = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Representation:
    name: str
    adjacency_entries: list[dict[str, Any]]
    company_seeds: list[dict[str, Any]]
    technology_shift_claims: list[dict[str, Any]]
    named_account_anchors: list[dict[str, Any]]
    geography_expansions: list[dict[str, Any]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_artifact_payload(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if "expansion_brief_v3" in payload:
        return payload["expansion_brief_v3"]
    if "expansion_brief" in payload:
        return payload["expansion_brief"]
    return payload


def _ascii_phrase(text: str) -> str:
    return (
        unicodedata.normalize("NFKD", text or "")
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _clean_label(text: str) -> str:
    cleaned = re.sub(r"^\d+\.\s*", "", str(text or "").strip())
    cleaned = cleaned.replace("&", " and ")
    cleaned = re.sub(r"[^a-z0-9]+", " ", _ascii_phrase(cleaned).lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def _token_set(text: str) -> set[str]:
    stopwords = {"and", "the", "of", "for", "to", "in", "via", "with", "management", "system", "platform"}
    return {
        token
        for token in _clean_label(text).split()
        if token and token not in stopwords and len(token) > 2
    }


def _similarity(a: str, b: str) -> float:
    a_clean = _clean_label(a)
    b_clean = _clean_label(b)
    if not a_clean or not b_clean:
        return 0.0
    if a_clean == b_clean:
        return 1.0
    if a_clean in b_clean or b_clean in a_clean:
        return 0.92
    a_tokens = _token_set(a)
    b_tokens = _token_set(b)
    if not a_tokens or not b_tokens:
        return 0.0
    intersection = len(a_tokens & b_tokens)
    union = len(a_tokens | b_tokens)
    jaccard = intersection / union if union else 0.0
    containment = intersection / min(len(a_tokens), len(b_tokens))
    return max(jaccard, containment * 0.9)


def _parse_section_items(report_text: str, section_title: str) -> list[str]:
    starts = list(SECTION_RE.finditer(report_text))
    section_body = ""
    for idx, match in enumerate(starts):
        if match.group(1).strip() != section_title:
            continue
        start = match.end()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(report_text)
        section_body = report_text[start:end]
        break
    if not section_body:
        return []
    items = []
    for match in ITEM_RE.finditer(section_body):
        items.append(re.sub(r"^\d+\.\s*", "", match.group(1).strip()))
    return items


def _parse_report_baseline(report_path: Path) -> dict[str, list[str]]:
    text = report_path.read_text(encoding="utf-8")
    return {
        "adjacency_boxes": _parse_section_items(text, "Adjacency Boxes"),
        "company_seeds": _parse_section_items(text, "Company Seeds"),
        "technology_shift_claims": _parse_section_items(text, "Technology and Market Shifts"),
    }


def _canonicalize_saved_artifact(path: Path) -> dict[str, Any]:
    return normalize_expansion_brief(_load_artifact_payload(path))


def _derive_v1_projection(canonical: dict[str, Any]) -> Representation:
    legacy_items = list(canonical.get("adjacent_capabilities") or []) + list(
        canonical.get("adjacent_customer_segments") or []
    )
    if not legacy_items:
        for box in canonical.get("adjacency_boxes") or []:
            if not isinstance(box, dict):
                continue
            criticality = box.get("criticality") or {}
            evidence = box.get("evidence") or []
            legacy_items.append(
                {
                    "label": box.get("label"),
                    "expansion_type": (
                        "adjacent_customer_segment"
                        if box.get("adjacency_kind") == "adjacent_customer_segment"
                        else "adjacent_capability"
                    ),
                    "confidence": box.get("confidence"),
                    "why_it_matters": box.get("why_it_matters"),
                    "evidence_urls": [item.get("url") for item in evidence if isinstance(item, dict) and item.get("url")],
                    "source_entity_names": [
                        item.get("source_entity_name")
                        for item in evidence
                        if isinstance(item, dict) and item.get("source_entity_name")
                    ],
                    "market_importance": criticality.get("market_importance"),
                    "operational_centrality": criticality.get("operational_centrality"),
                    "workflow_criticality": criticality.get("workflow_criticality"),
                    "daily_operator_usage": criticality.get("daily_operator_usage"),
                    "switching_cost_intensity": criticality.get("switching_cost_intensity"),
                    "priority_tier": box.get("priority_tier"),
                }
            )
    return Representation(
        name="v1_proxy",
        adjacency_entries=legacy_items,
        company_seeds=[],
        technology_shift_claims=[],
        named_account_anchors=list(canonical.get("named_account_anchors") or []),
        geography_expansions=list(canonical.get("geography_expansions") or []),
    )


def _representation_from_canonical(name: str, canonical: dict[str, Any]) -> Representation:
    return Representation(
        name=name,
        adjacency_entries=list(canonical.get("adjacency_boxes") or []),
        company_seeds=list(canonical.get("company_seeds") or []),
        technology_shift_claims=list(canonical.get("technology_shift_claims") or []),
        named_account_anchors=list(canonical.get("named_account_anchors") or []),
        geography_expansions=list(canonical.get("geography_expansions") or []),
    )


def _best_matches(expected: list[str], candidate_labels: list[str], threshold: float = 0.45) -> tuple[list[tuple[str, str, float]], list[str]]:
    matches: list[tuple[str, str, float]] = []
    used: set[int] = set()
    missing: list[str] = []
    for expected_label in expected:
        best_idx = None
        best_score = 0.0
        for idx, candidate_label in enumerate(candidate_labels):
            if idx in used:
                continue
            score = _similarity(expected_label, candidate_label)
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None or best_score < threshold:
            missing.append(expected_label)
            continue
        used.add(best_idx)
        matches.append((expected_label, candidate_labels[best_idx], best_score))
    return matches, missing


def _box_workflow_completeness(item: dict[str, Any]) -> float:
    anatomy = item.get("workflow_anatomy") or {}
    fields = [
        bool(anatomy.get("primary_operators")),
        bool(anatomy.get("primary_triggers")),
        bool(anatomy.get("core_actions")),
        bool(anatomy.get("systems_touched")),
        str(anatomy.get("frequency") or "mixed") != "mixed",
        bool(str(anatomy.get("failure_cost") or "").strip()),
        bool(str(anatomy.get("management_value") or "").strip()),
    ]
    return sum(fields) / len(fields)


def _criticality_richness(item: dict[str, Any]) -> float:
    criticality = item.get("criticality") or item
    fields = [
        bool(str(criticality.get("market_importance") or "").strip()),
        bool(str(criticality.get("operational_centrality") or "").strip()),
        bool(str(criticality.get("workflow_criticality") or "").strip()),
        bool(str(criticality.get("daily_operator_usage") or "").strip()),
        bool(str(criticality.get("switching_cost_intensity") or "").strip()),
        bool(str(item.get("priority_tier") or criticality.get("priority_tier") or "").strip()),
        bool(str(criticality.get("strategic_value_hypothesis") or "").strip()),
    ]
    return sum(fields) / len(fields)


def _evidence_richness(item: dict[str, Any], *, legacy: bool) -> float:
    if legacy:
        urls = item.get("evidence_urls") or []
        names = item.get("source_entity_names") or []
        score = 0.0
        if urls:
            score += 0.7
        if names:
            score += 0.3
        return min(score, 1.0)
    evidence = item.get("evidence") or []
    if not evidence:
        return 0.0
    structured = 0.0
    for ev in evidence:
        if not isinstance(ev, dict):
            continue
        if ev.get("supports"):
            structured += 0.4
        if ev.get("claim_text"):
            structured += 0.3
        if ev.get("source_entity_name"):
            structured += 0.2
        if ev.get("language"):
            structured += 0.1
    return min(1.0, 0.4 + structured / max(1, len(evidence)))


def _retrieval_richness(item: dict[str, Any], *, legacy: bool) -> float:
    if legacy:
        return 0.0
    seeds = item.get("retrieval_query_seeds") or []
    score = min(1.0, len([seed for seed in seeds if str(seed).strip()]) / 3)
    return score


def _multilingual_richness(item: dict[str, Any], *, legacy: bool) -> float:
    if legacy:
        label = str(item.get("label") or "")
        return 0.35 if any(ord(ch) > 127 for ch in label) else 0.1
    alias_count = len(item.get("original_language_aliases") or [])
    query_count = len(item.get("language_specific_query_seeds") or [])
    label = str(item.get("label") or "")
    bilingual_label = 1.0 if any(ord(ch) > 127 for ch in label) or "(" in label else 0.0
    return min(1.0, (min(alias_count, 2) / 2) * 0.5 + (min(query_count, 2) / 2) * 0.3 + bilingual_label * 0.2)


def _score_1_to_5(value: float) -> int:
    value = max(0.0, min(1.0, value))
    return max(1, min(5, int(round(1 + value * 4))))


def _find_entry_by_label(entries: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    best = None
    best_score = 0.0
    for entry in entries:
        score = _similarity(label, str(entry.get("label") or entry.get("name") or ""))
        if score > best_score:
            best_score = score
            best = entry
    return best if best_score >= 0.45 else None


def _evaluate_representation(baseline: dict[str, list[str]], representation: Representation) -> dict[str, Any]:
    is_legacy = representation.name == "v1_proxy"
    candidate_labels = [str(item.get("label") or "") for item in representation.adjacency_entries]
    adj_matches, adj_missing = _best_matches(baseline["adjacency_boxes"], candidate_labels)
    adj_coverage = len(adj_matches) / max(1, len(baseline["adjacency_boxes"]))
    precision = min(1.0, len(adj_matches) / max(1, len(candidate_labels))) if candidate_labels else 0.0
    adjacency_relevance = _score_1_to_5(0.75 * adj_coverage + 0.25 * precision)

    matched_entries = []
    for expected, _, _ in adj_matches:
        entry = _find_entry_by_label(representation.adjacency_entries, expected)
        if entry:
            matched_entries.append(entry)

    workflow_score_raw = (
        sum(_box_workflow_completeness(item) for item in matched_entries) / len(matched_entries)
        if matched_entries and not is_legacy
        else 0.0
    )
    workflow_specificity = _score_1_to_5(0.55 * adj_coverage + 0.45 * workflow_score_raw)

    criticality_score_raw = (
        sum(_criticality_richness(item) for item in matched_entries) / len(matched_entries)
        if matched_entries
        else 0.0
    )
    criticality_usefulness = _score_1_to_5(0.4 * adj_coverage + 0.6 * criticality_score_raw)

    evidence_score_raw = (
        sum(_evidence_richness(item, legacy=is_legacy) for item in matched_entries) / len(matched_entries)
        if matched_entries
        else 0.0
    )
    evidence_quality = _score_1_to_5(0.35 * adj_coverage + 0.65 * evidence_score_raw)

    seed_names = [str(item.get("name") or "") for item in representation.company_seeds]
    seed_matches, seed_missing = _best_matches(baseline["company_seeds"], seed_names, threshold=0.5)
    seed_coverage = len(seed_matches) / max(1, len(baseline["company_seeds"])) if baseline["company_seeds"] else 0.0
    mapped_seed_ratio = 0.0
    if representation.company_seeds:
        mapped_seed_ratio = len([
            item for item in representation.company_seeds if item.get("fit_to_adjacency_box_ids")
        ]) / len(representation.company_seeds)
    company_seed_usefulness = _score_1_to_5(0.7 * seed_coverage + 0.3 * mapped_seed_ratio)

    query_richness = (
        sum(_retrieval_richness(item, legacy=is_legacy) for item in matched_entries) / len(matched_entries)
        if matched_entries
        else 0.0
    )
    anchor_bonus = 1.0 if representation.named_account_anchors else 0.0
    geo_bonus = 1.0 if representation.geography_expansions else 0.0
    retrieval_usefulness = _score_1_to_5(0.5 * query_richness + 0.25 * anchor_bonus + 0.25 * geo_bonus)

    shift_labels = [str(item.get("label") or "") for item in representation.technology_shift_claims]
    shift_matches, shift_missing = _best_matches(baseline["technology_shift_claims"], shift_labels, threshold=0.45)
    if baseline["technology_shift_claims"]:
        shift_coverage = len(shift_matches) / len(baseline["technology_shift_claims"])
        linked_shift_ratio = 0.0
        if representation.technology_shift_claims:
            linked_shift_ratio = len([
                item for item in representation.technology_shift_claims if item.get("affected_adjacency_box_ids")
            ]) / len(representation.technology_shift_claims)
        trend_usefulness = _score_1_to_5(0.7 * shift_coverage + 0.3 * linked_shift_ratio)
    else:
        emerging_signal_count = 0
        if not is_legacy:
            for item in matched_entries:
                emerging_signal_count += len(item.get("emerging_signals") or [])
        trend_usefulness = _score_1_to_5(min(1.0, emerging_signal_count / max(1, len(matched_entries) or 1)))
        shift_missing = []

    multilingual_score_raw = (
        sum(_multilingual_richness(item, legacy=is_legacy) for item in matched_entries) / len(matched_entries)
        if matched_entries
        else 0.0
    )
    multilingual_normalization = _score_1_to_5(multilingual_score_raw)

    scores = {
        "adjacency_relevance": adjacency_relevance,
        "workflow_specificity": workflow_specificity,
        "criticality_usefulness": criticality_usefulness,
        "evidence_quality": evidence_quality,
        "company_seed_usefulness": company_seed_usefulness,
        "retrieval_usefulness": retrieval_usefulness,
        "trend_usefulness": trend_usefulness,
        "multilingual_normalization": multilingual_normalization,
    }
    core_dimensions = [
        "adjacency_relevance",
        "workflow_specificity",
        "criticality_usefulness",
        "evidence_quality",
        "company_seed_usefulness",
    ]
    core_total = sum(scores[key] for key in core_dimensions)
    extended_total = sum(scores.values())
    return {
        "representation": representation.name,
        "scores": scores,
        "core_total": core_total,
        "extended_total": extended_total,
        "matched_adjacencies": [expected for expected, _, _ in adj_matches],
        "missing_adjacencies": adj_missing,
        "matched_company_seeds": [expected for expected, _, _ in seed_matches],
        "missing_company_seeds": seed_missing,
        "matched_technology_shifts": [expected for expected, _, _ in shift_matches],
        "missing_technology_shifts": shift_missing,
        "candidate_adjacencies": candidate_labels,
        "candidate_company_seeds": seed_names,
        "candidate_technology_shifts": shift_labels,
    }


def _ranked_results(case_name: str, baseline: dict[str, list[str]], v2: dict[str, Any], v3: dict[str, Any]) -> dict[str, Any]:
    representations = [
        _derive_v1_projection(v2),
        _representation_from_canonical("v2", v2),
        _representation_from_canonical("v3", v3),
    ]
    evaluations = [_evaluate_representation(baseline, rep) for rep in representations]
    evaluations.sort(key=lambda item: (item["core_total"], item["extended_total"]), reverse=True)
    return {
        "case": case_name,
        "baseline": baseline,
        "evaluations": evaluations,
    }


def _markdown_report(results: list[dict[str, Any]]) -> str:
    lines = ["# Offline Expansion Version Eval", ""]
    lines.append("Baseline = saved raw Gemini report. Scores compare how much each representation preserves from that same report.")
    lines.append("")
    for case in results:
        lines.append(f"## {case['case']}")
        lines.append("")
        lines.append(f"- Baseline adjacency boxes: {', '.join(case['baseline']['adjacency_boxes']) or 'None'}")
        lines.append(f"- Baseline company seeds: {', '.join(case['baseline']['company_seeds']) or 'None'}")
        lines.append(f"- Baseline technology shifts: {', '.join(case['baseline']['technology_shift_claims']) or 'None'}")
        lines.append("")
        lines.append("| Representation | Core Total | Extended Total | Adjacency | Workflow | Criticality | Evidence | Seeds | Retrieval | Trend | Multilingual |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for evaluation in case["evaluations"]:
            scores = evaluation["scores"]
            lines.append(
                "| "
                f"{evaluation['representation']} | {evaluation['core_total']} | {evaluation['extended_total']} | "
                f"{scores['adjacency_relevance']} | {scores['workflow_specificity']} | "
                f"{scores['criticality_usefulness']} | {scores['evidence_quality']} | "
                f"{scores['company_seed_usefulness']} | {scores['retrieval_usefulness']} | "
                f"{scores['trend_usefulness']} | {scores['multilingual_normalization']} |"
            )
        lines.append("")
        for evaluation in case["evaluations"]:
            lines.append(f"### {evaluation['representation']}")
            lines.append(f"- Matched adjacencies: {', '.join(evaluation['matched_adjacencies']) or 'None'}")
            lines.append(f"- Missing adjacencies: {', '.join(evaluation['missing_adjacencies']) or 'None'}")
            lines.append(f"- Matched company seeds: {', '.join(evaluation['matched_company_seeds']) or 'None'}")
            lines.append(f"- Missing company seeds: {', '.join(evaluation['missing_company_seeds']) or 'None'}")
            if case["baseline"]["technology_shift_claims"]:
                lines.append(f"- Matched technology shifts: {', '.join(evaluation['matched_technology_shifts']) or 'None'}")
                lines.append(f"- Missing technology shifts: {', '.join(evaluation['missing_technology_shifts']) or 'None'}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="tmp/expansion-eval",
        help="Directory for markdown and JSON output.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for case_name, paths in DEFAULT_CASES.items():
        baseline = _parse_report_baseline(Path(paths["research_report"]))
        v2 = _canonicalize_saved_artifact(Path(paths["v2_artifact"]))
        v3 = _canonicalize_saved_artifact(Path(paths["v3_artifact"]))
        result = _ranked_results(case_name, baseline, v2, v3)
        result["artifacts"] = {
            "research_report": str(Path(paths["research_report"]).resolve()),
            "v2_artifact": str(Path(paths["v2_artifact"]).resolve()),
            "v3_artifact": str(Path(paths["v3_artifact"]).resolve()),
        }
        results.append(result)

    report_json = output_dir / "offline-expansion-version-eval.json"
    report_md = output_dir / "offline-expansion-version-eval.md"
    report_json.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_md.write_text(_markdown_report(results), encoding="utf-8")

    print(json.dumps({"json": str(report_json), "markdown": str(report_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
