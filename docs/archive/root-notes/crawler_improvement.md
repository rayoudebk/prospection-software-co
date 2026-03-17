You’ve basically built two different crawlers:

crawler.py: async + binary URL filtering + URL-only page typing + very lossy text extraction (regex strip + 5k chars).

seed_processor.py: sitemap/homepage discovery + LLM/heuristic URL selection + trafilatura extraction (better), but tiny caps (3k/page, 12k/domain) and logic that drops exactly the “customers/logos” pages you care about.

If your goal is “precisely identify capabilities + services + customers + use cases”, the crawling phase has a bunch of failure modes that will keep biting you until you restructure it.

Weaknesses (and what to do about them)
A) Discovery is shallow and sometimes wrong

Weaknesses

Sitemap parsing treats every <loc> as a page URL. In a sitemap_index.xml, those <loc> entries are sitemap files, not pages. You’ll miss most URLs on sites that use an index.

No robots.txt sitemap discovery (your doc mentions it, code doesn’t do it).

.xml.gz handling is wrong in crawler.py (reads response.text instead of decompressing bytes).

Homepage link extraction is regex-only and one-hop. You miss “Resources/Docs/Security/Case studies” that are not directly linked from the homepage or are nested under menus.

Hard “descendant-of-starting-path” rule makes you miss /solutions, /customers, etc if the user starts from a subpath.

Fixes

Implement proper sitemap recursion:

parse root tag: urlset => extract URLs, sitemapindex => fetch child sitemaps and recurse

handle gzip (gzip.decompress(response.content))

Add robots.txt sitemap discovery (parse Sitemap: lines).

Do menu-aware discovery:

fetch homepage HTML

extract nav + footer links (DOM parse, not regex), plus 1–2 hop BFS from high-score pages

Default crawl scope to domain root, not the seed path. Only restrict to descendants if explicitly requested.

B) Selection logic is split and URL-only (so it’s brittle)

Weaknesses

crawler.py uses binary include/exclude; seed_processor.py uses numeric scoring; they’ll disagree and create inconsistent packs.

LLM selection sees only URLs, no titles/H1/meta/headings. That’s weak signal; it will pick wrong pages on modern sites.

Depth penalty (-2 * path_depth) pushes you away from the detailed pages where capability lists live.

Hard-excluding /press /news /blog kills customer proof (press releases often contain “X selects Y” + vertical + product).

MAX_PAGES_PER_DOMAIN = 6 is too low for “capabilities + customers + services” coverage across most B2B sites.

Fixes

Unify selection into one scorer used everywhere.

Replace hard excludes with:

hard-exclude only auth/legal/careers

soft-demote blog/press/news, but promote back if it smells like proof (case study, customer story, “selected by”, “partnered with”, integration announcement)

Add a preview stage (cheap fetch) and score on:

URL + title + H1 + meta description + first headings

Increase page budget, but make it coverage-based (see below) rather than a fixed 6.

C) Extraction destroys structure and drops “customers”

Weaknesses

crawler.py regex-strips HTML into one blob and truncates at 5k chars. You lose headings, bullet boundaries, tables — exactly where capabilities are.

seed_processor.py uses trafilatura (good), but:

include_images=False means you lose customer logo alt text / aria labels (common way customers are represented)

MIN_CONTENT_LENGTH=200 drops “Customers” pages that are mostly logos (low text)

MAX_CONTENT_PER_PAGE=3000 cuts off long feature lists and detailed platform pages

Neither stores “evidence spans” (where did this capability/customer come from?), so you can’t build a precise taxonomy later.

Fixes

For capabilities: preserve structure. At minimum:

keep headings and lists (convert </li>, </p>, </h2> to newlines before stripping)

don’t truncate at crawl time; truncate only when rendering markdown

For customers: do a customer-proof extractor that runs on raw HTML:

collect img[alt], svg[aria-label], *[aria-label] inside likely “logo wall” sections

capture nearby text (“Trusted by”, “Customers”, “Case study”)

keep this even if main-text extraction is short

Store evidence as {source_url, snippet, html_selector_or_offset} so later steps can cite.

D) You’re not crawling the “services rendered” pages you need

Weaknesses

Your priority paths don’t include common services vocabulary (/services, /implementation, /support, /migration, /training, /professional-services, /customer-success).

Page types are too limited (no pricing, security, services/implementation, docs/resources).

If the site uses “Resources / Whitepapers / Datasheets” you miss the best capability lists.

Fixes

Expand discovery + page types to include:

services_implementation, support, pricing_packaging, security_compliance, docs_resources, industries, use_cases, case_study (separate from generic customers)

Treat PDFs as first-class:

discover PDF links from sitemap + resources pages

extract text from PDFs and tag them as high-signal for capabilities

E) No coverage guarantees (so you get lopsided context packs)

Weaknesses

LLM/heuristics select “top N pages”, but you don’t enforce that you got at least one good page for customers, services, integrations, etc.

Result: a pack that looks fine but is missing customers or services (common).

Fixes

Select pages with quotas per category, e.g.:

1–2 about/company

3–6 product/platform

3–6 solutions/use-cases/industries

2–4 customers/case-studies/press-with-proof

1 pricing (if exists)

1 security/trust (if exists)

1 services/implementation (if exists)

plus 1–3 PDFs (datasheet/overview)

Keep crawling until coverage is satisfied or marginal gain is low.

F) Your context pack schema is too “document-like”

Weaknesses

You output a markdown summary grouped by page type, but your next steps need structured objects (capabilities/services/customers/use cases).

Once you collapse everything into markdown early, precision and traceability are gone.

Fixes

Keep markdown as a view, but make the primary artifact structured:

pages[] with preview + content_blocks

signals extracted per page: capability candidates, customer candidates, service candidates, integration candidates

evidence[] per signal (URL + snippet + selector/offset)

What the improved crawling phase should look like (practical and stable)
1) Discovery (deterministic)

Sitemaps:

robots.txt Sitemap: lines

/sitemap.xml, /sitemap_index.xml, .gz, plus /sitemap/sitemap.xml

recurse sitemap indexes

Navigation extraction:

DOM parse homepage

pull nav + footer links

Controlled BFS:

expand from high-score “hub” pages (platform/solutions/resources/customers) to depth 1–2

Normalize URLs:

remove fragments + common tracking params (utm_*, gclid, etc.)

dedupe by canonicalized URL

2) Preview fetch + scoring (cheap)

For each candidate URL, fetch and extract only:

title, meta description, H1, first ~10 H2/H3 headings

(optional) a small main-text sample

Score using:

keyword hits across url/title/h1/headings

page “proof signals” (customers/case study language)

penalty for obvious junk (legal/auth/careers)

gentle depth penalty only if no strong positives

3) LLM-assisted triage (optional but high ROI)

Feed the LLM a list of previews (not full pages) and ask it to output:

page_type

contains_tags: capabilities/services/customers/integrations/pricing/security/docs

priority score
Then enforce category quotas and select.

4) Deep fetch + structured extraction

Fetch selected URLs and extract:

content blocks (heading + paragraphs + bullets + tables)

customer logo evidence from HTML (alt/aria)

keep full blocks; do truncation only in UI rendering

5) Context pack output (built for taxonomy)

Generate:

pages[] (with blocks + metadata)

signals[] (capability candidates, service candidates, customer candidates, use case candidates)

raw_markdown as a presentation layer, not the source of truth

The fastest set of changes that will noticeably improve “capabilities + customers”

If you do nothing else, do these in order:

Fix sitemap index recursion + gzip + robots sitemaps

Remove hard excludes for blog/press/news; demote instead + promote proof-like pages

Add preview stage and score using title/H1/headings, not just URL

Stop dropping low-text pages (customers pages) and separately parse logo alt/aria evidence

Stop truncating at crawl time (3k/5k is killing feature lists)

Enforce coverage quotas (at least 1 services page + 1 customers proof page + 1 platform/features page)

That gets you from “pretty summary” to “reliable raw material” for capability/customer taxonomy.