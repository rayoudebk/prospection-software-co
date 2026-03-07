from app.services.comparator_sources import parse_wealth_mosaic_listing
from app.workers.workspace_tasks import _seed_candidates_from_mentions, _source_type_for_url


def test_parse_wealth_mosaic_listing_extracts_company_and_solution_metadata():
    html = """
    <html>
      <body>
        <div class="sol-block">
          <h3>Dorsum Wealth Platform</h3>
          <a href="/vendors/dorsum/dorsum-wealth-platform/">View solution</a>
          <a href="https://www.dorsum.eu/">Website Address</a>
          <p>Portfolio and wealth management workflow platform.</p>
        </div>
      </body>
    </html>
    """
    mentions = parse_wealth_mosaic_listing(
        html,
        "https://www.thewealthmosaic.com/needs/portfolio-wealth-management-systems/",
    )
    assert mentions
    row = mentions[0]
    assert row["company_name"] == "Dorsum"
    assert row["company_slug"] == "dorsum"
    assert row["solution_slug"] == "dorsum-wealth-platform"
    assert row["entity_type"] == "solution"
    assert row["profile_url"].endswith("/vendors/dorsum/dorsum-wealth-platform/")
    assert row["official_website_url"] == "https://dorsum.eu/"


def test_seed_candidates_from_mentions_uses_official_website_not_directory_profile():
    seeded = _seed_candidates_from_mentions(
        [
            {
                "company_name": "Dorsum",
                "profile_url": "https://www.thewealthmosaic.com/vendors/dorsum/dorsum-wealth-platform/",
                "official_website_url": "https://dorsum.eu",
                "listing_url": "https://www.thewealthmosaic.com/needs/portfolio-wealth-management-systems/",
                "entity_type": "solution",
                "solution_slug": "dorsum-wealth-platform",
                "company_slug": "dorsum",
                "listing_text_snippets": ["Solution: Dorsum Wealth Platform"],
            }
        ]
    )
    assert seeded
    row = seeded[0]
    assert row["website"] == "https://dorsum.eu"
    assert row["official_website_url"] == "https://dorsum.eu"
    assert row["discovery_url"].startswith("https://www.thewealthmosaic.com/vendors/dorsum/")
    assert row["entity_type"] == "solution"


def test_source_type_for_url_never_marks_directory_as_first_party():
    directory_url = "https://www.thewealthmosaic.com/vendors/dorsum/dorsum-wealth-platform/"
    first_party = ["dorsum.eu"]
    assert _source_type_for_url(directory_url, "dorsum.eu", first_party_domains=first_party) == "directory_comparator"
    assert _source_type_for_url("https://dorsum.eu/products", "dorsum.eu", first_party_domains=first_party) == "first_party_website"

