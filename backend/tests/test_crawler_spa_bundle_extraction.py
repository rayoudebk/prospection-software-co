import asyncio

import httpx

from app.services.crawler import extraction as extraction_module
from app.services.crawler.extraction import ContentExtractor
from app.services.crawler.models import PagePreview


def test_content_extractor_recovers_spa_bundle_logos_and_route_labels():
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <title>4TPM - Plateforme Wealth Management</title>
        <meta name="description" content="Front-to-back office trading operations. Fully automated STP." />
        <script type="module" src="/assets/index-test.js"></script>
      </head>
      <body>
        <div id="root"></div>
      </body>
    </html>
    """
    bundle = """
    const menu = [
      {to:"/solutions/online-brokerage",children:"Bourse en ligne"},
      {to:"/solutions/private-banks",children:"Banques privées"},
      {to:"/platform/front-office",children:"Front office titres"}
    ];
    const customers = [
      {name:"Allianz Bank",logo:"/customer_logo/allianz_bank.png"},
      {name:"AXA Banque",logo:"/customer_logo/axa_banque_logo.png"}
    ];
    const partners = [
      {name:"ABN AMRO",src:"/partenaire/abn_amro.png"},
      {name:"ODDO BHF",src:"/partenaire/oddo-logo.png"}
    ];
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://4tpm.fr/offers/patio-oms/?lang=en":
            return httpx.Response(200, text=html)
        if str(request.url) == "https://4tpm.fr/assets/index-test.js":
            return httpx.Response(200, text=bundle)
        return httpx.Response(404, text="")

    async def run_test():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            extractor = ContentExtractor(client)
            preview = PagePreview(
                url="https://4tpm.fr/offers/patio-oms/?lang=en",
                title="4TPM - Plateforme Wealth Management",
                meta_description="",
                h1="",
                headings=[],
                path_depth=2,
            )
            return await extractor._extract_page(preview)

    page = asyncio.run(run_test())

    assert page is not None
    assert {item.name for item in page.customer_evidence} >= {"Allianz Bank", "AXA Banque"}
    assert any(signal.type == "integration" and signal.value == "ABN AMRO" for signal in page.signals)
    assert any(signal.type == "customer_archetype" and signal.value == "Banques privées" for signal in page.signals)
    assert any(signal.type == "workflow" and signal.value == "Front office titres" for signal in page.signals)
    assert "Front-to-back office trading operations. Fully automated STP." in page.raw_content
    assert "Bourse en ligne" in page.raw_content


def test_content_extractor_renders_product_pages_for_interactive_enrichment(monkeypatch):
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <title>Front Office</title>
        <meta name="description" content="Fonctionnalites detaillees available below." />
      </head>
      <body>
        <main>
          <h1>Front Office</h1>
          <p>Portfolio management system for wealth managers.</p>
        </main>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://4tpm.fr/platform/front-office":
            return httpx.Response(200, text=html)
        return httpx.Response(404, text="")

    monkeypatch.setattr(
        extraction_module,
        "render_page_via_chrome_devtools_mcp",
        lambda url, timeout_seconds=20, prefer_playwright=False: {
            "url": url,
            "final_url": url,
            "provider": "test",
            "content": (
                "Front Office Portfolio management system for wealth managers. "
                "Fonctionnalites detaillees Order management Compliance controls "
                "Portfolio analytics Trade capture Reporting."
            ),
            "html": """
            <html><body><main>
              <h1>Front Office</h1>
              <h2>Fonctionnalites detaillees</h2>
              <button>Pre-trade, trading et post-trade PATIO OMS</button>
              <button>Compliance et controles reglementaires PATIO OMS</button>
              <ul>
                <li>Order management</li>
                <li>Compliance controls</li>
                <li>Portfolio analytics</li>
              </ul>
            </main></body></html>
            """,
            "error": None,
        },
    )

    async def run_test():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            extractor = ContentExtractor(client)
            preview = PagePreview(
                url="https://4tpm.fr/platform/front-office",
                title="Front Office",
                meta_description="Fonctionnalites detaillees available below.",
                h1="Front Office",
                headings=["Fonctionnalites detaillees"],
                path_depth=2,
            )
            return await extractor._extract_page(preview)

    page = asyncio.run(run_test())

    assert page is not None
    assert "Order management" in page.raw_content
    assert "Compliance controls" in page.raw_content
    assert any(block.type == "heading" and "Fonctionnalites detaillees" in block.content for block in page.blocks)
    assert any(block.type == "list" and "Order management" in block.content for block in page.blocks)
    assert any(block.type == "list" and "Pre-trade, trading et post-trade PATIO OMS" in block.content for block in page.blocks)


def test_content_extractor_prefers_playwright_for_interactive_enrichment(monkeypatch):
    html = """
    <!doctype html>
    <html lang="en">
      <head>
        <title>Front Office</title>
      </head>
      <body>
        <main>
          <h1>Front Office</h1>
          <p>Thin page.</p>
        </main>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url) == "https://4tpm.fr/platform/front-office":
            return httpx.Response(200, text=html)
        return httpx.Response(404, text="")

    captured = {}

    def _render(url, timeout_seconds=20, prefer_playwright=False):
        captured["prefer_playwright"] = prefer_playwright
        return {
            "url": url,
            "final_url": url,
            "provider": "playwright",
            "content": "Expanded DOM content Order management Compliance controls",
            "html": "<html><body><main><button>Order management</button><button>Compliance controls</button></main></body></html>",
            "error": None,
        }

    monkeypatch.setattr(extraction_module, "render_page_via_chrome_devtools_mcp", _render)

    async def run_test():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            extractor = ContentExtractor(client)
            preview = PagePreview(
                url="https://4tpm.fr/platform/front-office",
                title="Front Office",
                meta_description="",
                h1="Front Office",
                headings=[],
                path_depth=2,
            )
            return await extractor._extract_page(preview)

    page = asyncio.run(run_test())

    assert page is not None
    assert captured["prefer_playwright"] is True


def test_render_enrichment_always_runs_for_platform_and_solution_pages():
    extractor = ContentExtractor(client=None)  # type: ignore[arg-type]

    assert extractor._should_attempt_render_enrichment(
        preview=PagePreview(
            url="https://4tpm.fr/platform/front-office",
            title="Front Office",
            meta_description="",
            h1="",
            headings=[],
            path_depth=2,
        ),
        page_type="product",
        raw_content="Front-to-back office trading operations. Procapital Allianz Bank AXA Banque SwissLife Deutsche Bank Monte Paschi Banque Uptevia Milleis Banque Privée",
    ) is True

    assert extractor._should_attempt_render_enrichment(
        preview=PagePreview(
            url="https://4tpm.fr/solutions/private-banks",
            title="Private banks",
            meta_description="",
            h1="",
            headings=[],
            path_depth=2,
        ),
        page_type="solutions",
        raw_content="Banques privées Bourse en ligne Sociétés de gestion Épargne retraite et salariale",
    ) is True
