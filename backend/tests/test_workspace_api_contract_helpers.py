from app.api.workspaces import (
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
    assert payload["screening_run_id"] == "20260211T101500"

    explicit = _quality_payload_from_job_result(
        {
            "run_quality_tier": "high_quality",
            "quality_gate_passed": True,
            "degraded_reasons": [],
            "model_attempt_trace": [{"provider": "gemini"}],
            "stage_time_ms": {"stage_seed_ingest": 1200},
            "timeout_events": [],
            "screening_run_id": "abc",
        }
    )
    assert explicit["run_quality_tier"] == "high_quality"
    assert explicit["quality_gate_passed"] is True
    assert explicit["model_attempt_trace"] == [{"provider": "gemini"}]
    assert explicit["stage_time_ms"]["stage_seed_ingest"] == 1200
