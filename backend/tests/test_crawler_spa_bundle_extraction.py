import asyncio

import httpx

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
