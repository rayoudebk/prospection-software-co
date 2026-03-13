# Phase 1 Market Map Reasoning

## Recommended Sequence

Phase 1 reasoning should happen after crawl and extraction.

Recommended order:

1. crawl source + comparator sites
2. normalize into `ContextPackV2`
3. build taxonomy and lens seeds
4. run the market-map reasoning prompt
5. render the brief in the UI

Do not run the reasoning model directly on raw web pages.

## Input Contract

The reasoning model should receive:

- source company identity
- source-scoped `ContextPackV2`
- normalized taxonomy nodes and edges
- lens seeds
- crawl coverage summary

The prompt should not require full raw HTML.

## System Prompt Draft

```text
You are an M&A sourcing analyst generating a phase-1 Market Map Brief from normalized source-company evidence.

Your job is discovery-first, not transaction-first.

You must:
- describe what the source company appears to sell
- identify the strongest customer archetypes
- identify the strongest workflows
- identify the strongest capabilities
- identify delivery or integration traits separately from capabilities
- surface named customer proof separately from integration/partner proof
- recommend which market-map lenses are active
- write short adjacency hypotheses grounded in the evidence
- write open questions only when evidence is thin or conflicting

You must not:
- invent customer names, workflows, or capabilities not supported by the input
- mix workflow, capability, and delivery into the same field
- use comparator evidence to overwrite the source-company understanding
- infer ownership, valuation, or dealability unless explicitly present in the source evidence

Output valid JSON matching the MarketMapBrief schema exactly.

Prioritize:
1. source-company evidence
2. rendered product/solution page evidence
3. named customer proof
4. integration proof
5. generic summaries

When evidence is weak:
- say so directly
- keep the field sparse rather than filling it with generic abstractions

When multiple phrases overlap:
- prefer the sharper, more operator-meaningful phrase
- keep delivery/integration terms out of workflow and capability nodes

Adjacency hypotheses must be short and evidence-backed.
They should sound like:
"Private banks using front-office PMS/OMS software may also purchase adjacent compliance, reporting, and digital client-service capabilities."

Open questions must be concrete evidence gaps.
They should sound like:
"Named customer proof exists, but the source site does not clearly distinguish private-bank vs asset-manager adoption by product line."
```

## Output Expectations

The model should return:

- a short `source_summary`
- a selective top slice of taxonomy nodes
- high-confidence named customer proof
- high-confidence integration proof
- active lenses only when justified
- no unsupported facts

## Evaluation Fixtures

### `4TPM`

Expected strengths:

- strong source-company workflow evidence from `/platform/*`
- named customer logos
- partner / infrastructure proof
- multilingual source pages

What good output should contain:

- customer archetypes like `private bank`, `online brokerage`, `asset manager`
- workflows like `front office titres`, `back office titres`
- capabilities like `Décisions de gestion et arbitrages`, `Pré-trade, trading et post-trade`, `Connectivité et routage multi-marchés`
- delivery terms like `REST API`, `API documentation`

What bad output looks like:

- `APIs REST` listed as workflow
- named customers inside capability taxonomy
- long sentence fragments from payments pages dominating the brief

### `Hublo`

Expected strengths:

- cleaner buyer/workflow language
- useful careers-page context
- healthcare staffing workflow evidence

What good output should contain:

- customer archetypes like `hospital`, `care provider`, `operations team`
- workflows like `shift replacement`, `internal mobility`, `staffing operations`
- capabilities tied to workforce management rather than generic HR software

What bad output looks like:

- collapsing the company into generic `staffing`
- missing healthcare-specific customer context
- treating job-posting tech stack as core product capability

## Evaluation Checklist

For each fixture, check:

- Are customer/workflow/capability/delivery layers separated correctly?
- Are the top surfaced nodes more useful than the long tail?
- Are named customer and partner proofs separated?
- Are active lenses justified by evidence?
- Are open questions actually evidence gaps rather than generic filler?
- Would an associate understand where to search next from this brief?

## Next Implementation Step

Once the schema is stable, wire the reasoning model behind a dedicated phase-1 call that consumes only:

- `ContextPackV2`
- taxonomy nodes / edges
- lens seeds
- crawl coverage

That call should become the canonical generator for `market_map_brief`.
