from app.api.workspaces import (
    _compute_variance_hotspots_from_results,
    _citation_summary_from_meta,
    _clean_url_list,
    _quality_payload_from_job_result,
    _screening_diagnostics_from_meta,
)


def test_clean_url_list_normalizes_dedupes_and_supports_path_requirement():
    values = [
        "upvest.co/blog/zopa-bank-partners-with-upvest",
        "https://upvest.co/blog/zopa-bank-partners-with-upvest",
        "HTTPS://upvest.co/blog/boerse-stuttgart-and-upvest",
        "https://upvest.co",
        "mailto:foo@upvest.co",
        "",
    ]
    cleaned = _clean_url_list(values, max_items=10)
    assert cleaned == [
        "https://upvest.co/blog/zopa-bank-partners-with-upvest",
        "https://upvest.co/blog/boerse-stuttgart-and-upvest",
        "https://upvest.co",
    ]

    path_only = _clean_url_list(values, max_items=10, require_path=True)
    assert path_only == [
        "https://upvest.co/blog/zopa-bank-partners-with-upvest",
        "https://upvest.co/blog/boerse-stuttgart-and-upvest",
    ]


def test_citation_summary_from_meta_validates_payload_shape_and_version():
    valid_meta = {
        "citation_summary_v1": {
            "version": "v1",
            "sentences": [
                {
                    "id": "s1",
                    "text": "Upvest partners with Zopa.",
                    "citation_pill_ids": ["p1"],
                }
            ],
            "source_pills": [
                {
                    "pill_id": "p1",
                    "label": "upvest.co",
                    "url": "https://upvest.co/blog/zopa-bank-partners-with-upvest",
                    "source_tier": "tier1_vendor",
                    "source_kind": "first_party",
                    "captured_at": "2026-01-01T00:00:00",
                    "claim_group": "traction",
                }
            ],
        }
    }
    parsed = _citation_summary_from_meta(valid_meta)
    assert parsed is not None
    assert parsed.version == "v1"
    assert len(parsed.sentences) == 1
    assert len(parsed.source_pills) == 1

    wrong_version = {
        "citation_summary_v1": {
            **valid_meta["citation_summary_v1"],
            "version": "v2",
        }
    }
    assert _citation_summary_from_meta(wrong_version) is None
    assert _citation_summary_from_meta({}) is None


def test_screening_diagnostics_from_meta_exposes_new_registry_and_hint_fields():
    diagnostics = _screening_diagnostics_from_meta(
        {
            "registry_neighbors_with_first_party_website_count": 8,
            "registry_neighbors_dropped_missing_official_website_count": 5,
            "registry_origin_screening_counts": {
                "records_screened": 20,
                "records_accepted": 8,
                "records_rejected": 12,
            },
            "first_party_hint_urls_used_count": 3,
            "first_party_hint_pages_crawled_total": 6,
        }
    )

    assert diagnostics["registry_neighbors_with_first_party_website_count"] == 8
    assert diagnostics["registry_neighbors_dropped_missing_official_website_count"] == 5
    assert diagnostics["registry_origin_screening_counts"]["records_accepted"] == 8
    assert diagnostics["first_party_hint_urls_used_count"] == 3
    assert diagnostics["first_party_hint_pages_crawled_total"] == 6


def test_quality_payload_from_job_result_defaults_and_explicit_fields():
    payload = _quality_payload_from_job_result(
        {
            "fallback_mode": True,
            "screening_run_id": "20260211T101500",
        }
    )
    assert payload["run_quality_tier"] == "degraded"
    assert payload["quality_gate_passed"] is False
    assert payload["quality_audit_v1"] is None
    assert payload["quality_audit_passed"] is False
    assert payload["screening_run_id"] == "20260211T101500"
    assert payload["stage_execution_mode"] == "hybrid_preflight_monolith"

    explicit = _quality_payload_from_job_result(
        {
            "run_quality_tier": "high_quality",
            "quality_gate_passed": True,
            "degraded_reasons": [],
            "model_attempt_trace": [{"provider": "gemini"}],
            "stage_time_ms": {"stage_seed_ingest": 1200},
            "timeout_events": [],
            "queue_wait_ms_by_stage": {"stage_seed_ingest": 45},
            "stage_retry_counts": {"stage_llm_discovery_fanout": 1},
            "cache_hit_rates": {"search_cache_hit_rate": 0.5, "url_cache_hit_rate": 0.25},
            "candidate_dropoff_funnel_v1": {"seed_total_count": 120, "kept_count": 25},
            "stage_execution_mode": "fully_staged",
            "screening_run_id": "abc",
            "quality_audit_v1": {
                "run_id": "abc",
                "pass": True,
                "patterns": [
                    {
                        "pattern_key": "fp_low_ticket_without_pricing_evidence",
                        "count": 0,
                        "sample_screening_ids": [],
                        "sample_candidate_names": [],
                    }
                ],
                "thresholds": {
                    "fp_low_ticket_without_pricing_evidence": 0,
                    "fn_missing_vertical_with_institutional_workflow_text": 8,
                    "fp_registry_or_directory_overweight": 5,
                    "fn_customer_proof_present_but_thin_grouping": 8,
                },
                "top_impacted_candidates": [],
            },
        }
    )
    assert explicit["run_quality_tier"] == "high_quality"
    assert explicit["quality_gate_passed"] is True
    assert explicit["quality_audit_passed"] is True
    assert explicit["quality_validation_ready"] is True
    assert explicit["quality_audit_v1"]["run_id"] == "abc"
    assert explicit["quality_audit_v1"]["pass"] is True
    assert explicit["model_attempt_trace"] == [{"provider": "gemini"}]
    assert explicit["stage_time_ms"]["stage_seed_ingest"] == 1200
    assert explicit["queue_wait_ms_by_stage"]["stage_seed_ingest"] == 45
    assert explicit["stage_retry_counts"]["stage_llm_discovery_fanout"] == 1
    assert explicit["cache_hit_rates"]["search_cache_hit_rate"] == 0.5
    assert explicit["candidate_dropoff_funnel_v1"]["seed_total_count"] == 120
    assert explicit["stage_execution_mode"] == "fully_staged"


def test_compute_variance_hotspots_from_results_tracks_high_variance_metrics():
    rows = [
        {
            "first_party_hint_urls_used_count": 4,
            "first_party_hint_pages_crawled_total": 1,
            "first_party_crawl_pages_total": 25,
            "stage_time_ms": {"stage_llm_discovery_fanout": 1100},
            "ranking_eligible_count": 9,
        },
        {
            "first_party_hint_urls_used_count": 96,
            "first_party_hint_pages_crawled_total": 11,
            "first_party_crawl_pages_total": 118,
            "stage_time_ms": {"stage_llm_discovery_fanout": 3900},
            "ranking_eligible_count": 24,
        },
        {
            "first_party_hint_urls_used_count": 12,
            "first_party_hint_pages_crawled_total": 2,
            "first_party_crawl_pages_total": 44,
            "stage_time_ms": {"stage_llm_discovery_fanout": 980},
            "ranking_eligible_count": 12,
        },
    ]
    hotspots = _compute_variance_hotspots_from_results(rows)
    assert hotspots
    metrics = {row["metric"]: row for row in hotspots}
    assert "first_party_hint_urls_used_count" in metrics
    assert metrics["first_party_hint_urls_used_count"]["max"] == 96.0
    assert metrics["first_party_hint_urls_used_count"]["min"] == 4.0
    assert metrics["first_party_hint_urls_used_count"]["run_count"] == 3
    assert "stage_llm_discovery_fanout.llm_ms" in metrics


def test_quality_payload_run_consistency_with_quality_audit():
    payload = _quality_payload_from_job_result(
        {
            "screening_run_id": "20260212T151814",
            "quality_audit_v1": {
                "run_id": "20260212T151814",
                "pass": False,
                "patterns": [
                    {
                        "pattern_key": "fp_low_ticket_without_pricing_evidence",
                        "count": 2,
                        "sample_screening_ids": [1211, 1220],
                        "sample_candidate_names": ["Upvest", "Allvue Systems"],
                    }
                ],
                "thresholds": {
                    "fp_low_ticket_without_pricing_evidence": 0,
                    "fn_missing_vertical_with_institutional_workflow_text": 8,
                    "fp_registry_or_directory_overweight": 5,
                    "fn_customer_proof_present_but_thin_grouping": 8,
                },
                "top_impacted_candidates": [
                    {
                        "screening_id": 1211,
                        "candidate_name": "Upvest",
                        "reasons": ["fp_low_ticket_without_pricing_evidence"],
                    }
                ],
            },
        }
    )
    assert payload["screening_run_id"] == "20260212T151814"
    assert payload["quality_audit_v1"] is not None
    assert payload["quality_audit_v1"]["run_id"] == payload["screening_run_id"]
    assert payload["quality_audit_passed"] is False
