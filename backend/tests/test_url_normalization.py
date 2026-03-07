from app.services.retrieval.url_normalization import dedupe_results, normalize_url


def test_normalize_url_strips_tracking_and_fragment():
    raw = "https://www.example.com/path/?utm_source=chatgpt&gclid=123&id=5#section"
    assert normalize_url(raw) == "https://example.com/path?id=5"


def test_dedupe_results_respects_domain_cap():
    results = [
        {"url": "https://example.com/a?utm_source=1"},
        {"url": "https://example.com/b?utm_source=2"},
        {"url": "https://example.com/c?utm_source=3"},
        {"url": "https://other.com/x"},
    ]
    deduped, stats = dedupe_results(results, per_domain_cap=2, total_cap=10)
    urls = [row["normalized_url"] for row in deduped]
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
    assert "https://example.com/c" not in urls
    assert stats["per_domain_cap_dropped"] == 1
