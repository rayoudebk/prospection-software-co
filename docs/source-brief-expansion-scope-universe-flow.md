# Source Brief -> Expansion Brief -> Scope Review -> Universe

## Purpose

This document defines the recommended product flow for phase-1 market understanding and phase-2 target discovery.

The main change from the current product direction is:

- do not ask the user to author retrieval lanes as the primary abstraction
- instead, let the system build:
  - a `Source Brief`
  - then an `Expansion Brief`
  - then a user-reviewed `Scope`
- use that approved scope to drive internal retrieval plans for universe discovery

The goal is to keep the product evidence-first while still allowing the system to expand beyond only what the source company explicitly says on its own website.

## Canonical Build Chain

The intended chain is:

1. `Primary + secondary evidence crawl / extraction`
2. `Normalized graph`
3. `Source Brief` (visible in the UI)
4. `Deep research handoff`
5. `Expansion Brief`
6. `User scope review before universe discovery`

In more detail:

1. `Primary + secondary evidence crawl / extraction`
- this step gathers:
  - first-party evidence from the source company
  - secondary evidence about the source company from customers, partners, directories, trade pages, and other public sources
- this step may also crawl comparator websites, but comparator self-descriptions are not part of source-company truth; they are downstream expansion inputs

2. `Normalized graph`
- this step turns the extracted evidence into the canonical `CompanyContextGraph`
- the graph is about the source company, not the whole market
- it should preserve provenance, evidence tiers, publisher types, and confidence

3. `Source Brief`
- this is the visible UI artifact for source-company understanding
- it should stay bounded to:
  - what the source company says about itself
  - what others say about the source company
- it should not yet try to produce the one-hop adjacency map

4. `Deep research handoff`
- a separate deep-research agent should read:
  - a graph-derived structured packet
  - plus the `Source Brief`
- this is the handoff point between phase-1 source understanding and bounded adjacency research
- the graph-derived packet should stay explicitly segmented into:
  - `source_company_truth`
  - `secondary_context_about_source_company`
  - `adjacent_market_inputs`

5. `Expansion Brief`
- deep research should return the bounded one-hop expansion artifact
- this is where comparator websites, directories, and broader adjacent-market signals become central inputs

6. `User scope review before universe discovery`
- the user reviews expanded nodes, keeps/removes/deprioritizes them, and that approved scope replaces user-authored retrieval-lane editing

This document covers the whole chain, but the current implementation focus is steps `1`, `2`, and `3`.

## Core Product Idea

The product should answer three distinct questions in sequence:

1. What does the source company clearly do?
2. What is plausibly adjacent to that business, but not yet fully proven?
3. Which companies should we actually search for and validate?

These are different jobs and should produce different artifacts.

If they are collapsed into one step, the model will either:

- stay too narrow and miss important adjacencies
- or drift into generic market-map hallucination

The correct flow is to keep each job bounded.

## Recommended User Flow

1. `Source Setup`
2. `Source Brief`
3. `Expansion Brief`
4. `Scope Review`
5. `Universe`
6. `Candidate Enrichment`
7. `Report`

## Why This Flow Is Better Than User-Authored Retrieval Lanes

Internal query groups are still useful, but they should be an internal retrieval abstraction, not the main UI object.

Why:

- users think in companies, products, buyers, and geographies, not retrieval lanes
- the product can infer retrieval lanes from approved graph nodes more consistently than users can author them manually
- making the user edit lanes too early forces them to guess the market box before the product has done the adjacency expansion work

In the recommended flow:

- the product proposes nodes
- the user removes or approves nodes
- the system compiles those approved nodes into internal query packs and registry lookups behind the scenes

## Design Principles

### 1. Keep the source understanding bounded

The `Source Brief` should be source-grounded.

It should answer:

- what the source company appears to sell
- to which customer types
- in which workflows
- through which capabilities and integrations

It should not try to infer the full market from first principles.

It should be built from:

- `primary evidence`
  - what the source company says about itself
- `secondary evidence`
  - what other parties say about the source company

It should not flatten comparator self-descriptions into source-company truth.

### 2. Make expansion explicit

The product should have a separate `Expansion Brief` artifact whose job is:

- expand one hop around the source graph
- propose adjacent capabilities
- propose adjacent customer segments
- propose named-account anchors
- propose geography expansions

This is the right place for deeper research.

Comparator websites are useful here, but they are expansion inputs, not part of the canonical source-company truth graph.

### 3. Separate facts from hypotheses

Every surfaced node should carry an explicit status such as:

- `source_grounded`
- `corroborated_expansion`
- `hypothesis`
- `user_kept`
- `user_removed`

This avoids making weak adjacency ideas look equivalent to first-party source evidence.

### 4. Ask the user to prune scope, not author it from scratch

The highest-leverage user action is usually:

- "yes, keep this"
- "no, this is outside scope"

not:

- "please write the right retrieval lane structure for this market"

## Step 1: Source Setup

### User Input

The user provides:

- source company website
- optional comparator company websites
- optional supporting evidence URLs

### System Behavior

The system:

- crawls the source company deeply
- gathers secondary public evidence from external pages that talk about the source company
- may crawl comparator sites more lightly as downstream expansion inputs
- normalizes source-company evidence into the company-context graph

### Main Output Artifacts

- `ContextPackV2`
- `CompanyContextGraph`
- normalized taxonomy candidates
- evidence ledger
- source documents
- secondary public evidence proof
- graph-derived structured packet for downstream deep research

### 4TPM Example

Suppose the user enters:

- source company: `4tpm.com`
- comparators: two portfolio-management or wealth-tech vendors
- optional supporting links: a directory page or customer announcement

The crawl should prioritize:

- product / platform pages
- solution pages
- customer / case-study pages
- integration / API pages
- selected careers pages if they reveal workflow or product context

From that step alone, the product may extract signals like:

- customer archetypes:
  - `private bank`
  - `online brokerage`
  - `asset manager`
- workflows:
  - `front office titres`
  - `back office titres`
- capabilities:
  - `portfolio management`
  - `order management`
  - `pre-trade / post-trade operations`
  - `multi-market routing`
- delivery or integration:
  - `REST API`
  - `API documentation`
  - `partner ecosystem`

The system may also detect named customers or partners from logos, case studies, or integration pages.

## Step 2: Source Brief

### Purpose

The `Source Brief` is the product's answer to:

"What does this source company clearly appear to do, based on normalized evidence?"

### What It Should Do

It should produce:

- short source summary
- top customer segments
- top workflows
- top capabilities
- top delivery / integration traits
- named customer proof
- partner / integration proof
- secondary market context about the source company
- evidence gaps

### What It Should Not Do

It should not:

- invent adjacent capabilities that are not evidenced nearby
- infer the entire market map
- replace evidence with generic category guesswork
- act like a full analyst report
- turn comparator self-descriptions into source-company facts

### Why This Constraint Matters

If the `Source Brief` tries to solve adjacency expansion itself, two bad outcomes happen:

- it becomes too conservative and only repeats first-party marketing language
- or it becomes too speculative and starts inventing market structure

The `Source Brief` should stay narrow and trustworthy.

The downstream handoff from this step should be:

- `graph-derived structured packet`
- `Source Brief`

Those are the two inputs the deep-research agent should read before producing the `Expansion Brief`.

### 4TPM Example

A good `Source Brief` for 4TPM might say:

- 4TPM appears to provide front-office and back-office securities workflow software
- strongest customer evidence points to private banks, brokers, and asset managers
- strongest capability evidence points to portfolio management, order workflow, and multi-market connectivity
- delivery evidence suggests API and partner-led integration surfaces

It may also include open questions such as:

- does 4TPM primarily win in private banking or broader wealth / asset-servicing segments?
- how much of its product is portfolio management versus post-trade operations?
- is customer proof concentrated in France or broader European markets?

### What The Source Brief Should Not Claim Yet

A good `Source Brief` should not confidently say:

- `voting rights` is definitely a core adjacent lane
- `Belgium` is definitely a priority geography
- `BNP Paribas` should anchor the entire market box

Those are expansion questions, not source-understanding facts.

## Step 3: Expansion Brief

### Purpose

The `Expansion Brief` is the bounded deep-research artifact.

Its job is not to re-explain the source company.

Its job is to answer:

"Given what the source company clearly does, what are the most plausible one-hop expansions around that source graph?"

The `Expansion Brief` should be generated from:

- the graph-derived structured packet
- the `Source Brief`
- expansion-oriented external inputs such as comparator websites, directories, and broader market sources

### Expansion Axes

The recommended expansion axes are:

- adjacent capabilities
- adjacent customer segments
- named accounts / logos
- geographies

Do not keep `competitor categories` as a first-class axis.

In practice, `competitor categories` usually collapse into:

- capability language
- or customer-segment language

Category labels can still be stored as metadata for search-query generation, but they should not be a separate lane type.

### Axis 1: Adjacent Capabilities

This means:

- product modules or functional areas that are close to the source company's proven core
- often bought by the same buyer
- often used in the same workflow neighborhood

For 4TPM, examples might include:

- `voting rights`
- `proxy voting`
- `client reporting`
- `performance & attribution`
- `corporate actions`
- `pre-trade compliance`

Why these are useful:

- they expand the market box by product adjacency
- they help find companies that are complementary, not just directly similar

When they should be proposed:

- when there is evidence nearby in:
  - partner pages
  - customer references
  - competitor sites
  - directory labels
  - external deployment or workflow mentions

How `voting rights` fits:

The system should not add `voting rights` merely because it sounds plausible for financial software.

It should only propose it if there is at least some bounded support, such as:

- neighboring vendor category pages repeatedly coupling custody / portfolio operations with proxy-voting workflows
- customer or partner evidence around securities-servicing operations
- product neighborhoods that frequently sit next to holdings, governance, or corporate-actions workflows

Then it can appear as:

- `hypothesis` if evidence is weak but non-random
- `corroborated_expansion` if multiple sources support it

### Axis 2: Adjacent Customer Segments

This means:

- buyer groups neighboring the source company's proven customer base
- similar enough to share some capability demand
- different enough to open new market-map branches

For 4TPM, examples might include:

- `fund administrator`
- `custodian`
- `family office platform`
- `wealth platform`
- `fund services provider`

Why these are useful:

- the same capability stack often moves across neighboring buyer segments
- this helps find companies that are strong in similar software but marketed to a different segment

Example:

If the source evidence strongly says:

- `private bank`
- `online brokerage`
- `asset manager`

then an expansion brief might propose:

- `fund administrator`

because it is adjacent in operations, reporting, holdings, trade, and securities-processing needs, even if 4TPM does not explicitly market to that segment.

### Axis 3: Named Accounts / Logos

This means:

- use specific customer names as anchors for market expansion

This is often the sharpest expansion axis because named accounts are more concrete than generic buyer labels.

For 4TPM, examples might include:

- `BNP Paribas`
- `Milleis Banque Privee`
- another named bank, broker, or asset manager surfaced by source or secondary evidence

Why named accounts matter:

- they anchor the actual buying environment
- they help trace:
  - peer institutions
  - adjacent vendors serving the same account
  - surrounding modules bought by the same institution

Example:

If `BNP Paribas` appears in the evidence, the system can ask:

- what other vendors sell front-office or securities-operations tooling into BNP Paribas?
- what adjacent modules appear in evidence tied to BNP Paribas?
- what peer institutions should be searched because they look like BNP Paribas?

This is often how the product can discover neighboring capability areas without inventing them.

### Axis 4: Geographies

This means:

- expand the market box by geography after the capability box is bounded

This should not happen before the product has a decent understanding of the capability neighborhood.

For 4TPM, examples might include:

- `Belgium`
- `Luxembourg`
- `Switzerland`
- `United Kingdom`

Why this matters:

- regulated financial software categories are often fragmented by market and jurisdiction
- adjacent targets may be strongest in nearby countries, not the source company's home country

Example:

If the source brief is strongest in France, the expansion brief may propose:

- `Belgium`
- `Luxembourg`

because they are plausible adjacent markets for private-banking, wealth-tech, or securities-operations vendors.

### Expansion Brief Output Rules

The output should separate:

- `confirmed`: strongly evidenced in source or multiple public sources
- `corroborated`: not directly first-party, but supported by multiple external signals
- `hypothesis`: plausible one-hop expansion with some support, but not yet strong enough to act as fact

### Example Expansion Output For 4TPM

```json
{
  "adjacent_capabilities": [
    {
      "label": "Voting rights / proxy voting",
      "status": "hypothesis",
      "confidence": 0.58,
      "why_it_matters": "Adjacent to securities administration and portfolio operations workflows evidenced around the source company.",
      "evidence_urls": [
        "https://example-directory.com/wealth-operations-vendors",
        "https://example-partner.com/securities-operations"
      ]
    },
    {
      "label": "Client reporting",
      "status": "corroborated",
      "confidence": 0.77,
      "why_it_matters": "Frequently coupled with the same portfolio and front-office workflows evidenced in the source graph.",
      "evidence_urls": [
        "https://example-vendor.com/platform",
        "https://example-customer-announcement.com/reporting-stack"
      ]
    }
  ],
  "adjacent_customer_segments": [
    {
      "label": "Fund administrator",
      "status": "hypothesis",
      "confidence": 0.61,
      "why_it_matters": "Neighboring buyer group with overlapping positions, reporting, and securities-processing needs.",
      "evidence_urls": [
        "https://example-market-map.com/fund-admin-software"
      ]
    }
  ],
  "named_account_anchors": [
    {
      "label": "BNP Paribas",
      "status": "corroborated",
      "confidence": 0.81,
      "why_it_matters": "Useful anchor institution for tracing peer vendors and adjacent modules in the same buying environment.",
      "evidence_urls": [
        "https://example-source.com/customers",
        "https://example-third-party.com/announcement"
      ]
    }
  ],
  "geography_expansions": [
    {
      "label": "Belgium",
      "status": "hypothesis",
      "confidence": 0.56,
      "why_it_matters": "Adjacent search geography for wealth-tech and securities-workflow vendors near the current source market box.",
      "evidence_urls": [
        "https://example-directory.com/belgium-financial-software"
      ]
    }
  ]
}
```

## Step 4: Scope Review

### Purpose

This step should replace user-authored retrieval-lane editing as the primary review surface.

Instead of asking the user:

- "Please define the right retrieval lanes"

the product should ask:

- "Which proposed nodes belong in the scope for discovery?"

### What The User Reviews

The user sees grouped node sets:

- source-grounded capabilities
- source-grounded customer segments
- source-grounded workflows
- expanded adjacent capabilities
- expanded adjacent customer segments
- named-account anchors
- geography expansions

### Allowed User Actions

The user should be able to:

- keep
- remove
- maybe deprioritize

The user should not need to author complex lane logic unless they explicitly want advanced control.

### What The System Does After Review

The system compiles approved nodes into:

- internal retrieval plans
- internal query groups
- query packs
- registry lookups
- geo filters

### Example Scope Review For 4TPM

The product could show:

- kept:
  - `portfolio management`
  - `order management`
  - `private bank`
  - `asset manager`
  - `client reporting`
  - `BNP Paribas`
  - `Belgium`
- removed:
  - `proxy voting`
  - if the user thinks it is too speculative
- deprioritized:
  - `fund administrator`
  - if plausible but not important for the first pass

This is a much better interaction than asking the user to manually craft:

- core lane
- adjacent lane
- must-include terms
- must-exclude terms

at a stage when the adjacency map is still being formed.

## Step 5: Universe Discovery

### Purpose

`Universe Discovery` should answer:

"Which companies fit the approved scope, with enough evidence to justify review?"

### Inputs

It should consume:

- source-grounded nodes
- user-kept expansion nodes
- approved geographies
- approved named-account anchors

### System Behavior

The system:

- builds query packs from approved nodes
- runs web search and external retrieval
- runs registry identity expansion
- resolves official websites and aliases
- deduplicates candidates
- performs light first-party enrichment

### Light Candidate Enrichment

Before the user validates candidates, the system should already collect enough to show:

- what the candidate seems to sell
- likely customer segment
- 1-2 supporting citations
- official website
- country if available

This avoids asking the user to validate only names and domains.

### 4TPM Example

Suppose the approved scope is:

- capabilities:
  - `portfolio management`
  - `order management`
  - `client reporting`
- customer segments:
  - `private bank`
  - `asset manager`
- geography:
  - `France`
  - `Belgium`
- anchor:
  - `BNP Paribas`

The product might then search for companies that:

- market PMS/OMS or reporting modules to private banks or asset managers
- operate in France or Belgium
- appear in buyer, partner, or category evidence around large banking accounts

The result is a first-pass universe, not a final dossier set.

## Step 6: Candidate Enrichment

### Purpose

After the user reviews the universe, the system should deepen the kept companies.

### Two Enrichment Levels

Recommended:

- `light enrichment` before candidate validation
- `deep enrichment` after candidate validation

### Light Enrichment

Purpose:

- help the user decide whether a candidate is worth keeping

Should include:

- product summary
- likely modules
- basic customer proof if available
- basic geography / identity evidence
- a few citations

### Deep Enrichment

Purpose:

- create reusable structured dossiers for shortlisted companies

Should include:

- modules / capabilities
- mapped bricks
- customer references
- integrations and deployment claims
- hiring signals
- filing metrics where reliable

## Step 7: Report

The report step should operate on enriched, validated candidates.

It should not be the place where the market box is still being defined.

## Concrete Difference Between Source Brief And Expansion Brief

This distinction is the most important part of the design.

### Source Brief

Question:

- what does the source company clearly do?

Evidence posture:

- source-grounded
- normalized
- bounded
- built from primary evidence plus secondary evidence about the source company

For 4TPM:

- yes:
  - `front office titres`
  - `portfolio management`
  - `private bank`
  - `REST API`
- maybe:
  - `asset manager`
- no:
  - `voting rights` as an established lane unless directly evidenced

### Expansion Brief

Question:

- what is the most plausible one-hop market expansion around the source graph?

Evidence posture:

- bounded deep research
- explicit corroboration levels
- hypotheses allowed, but labeled
- may use comparator websites and broader adjacent-market signals

For 4TPM:

- yes:
  - `voting rights` as a hypothesis if supported by adjacent evidence
  - `Belgium` as a geography expansion if justified
  - `BNP Paribas` as an anchor if surfaced in evidence

## How This Fits The Current Product

### What Already Exists

The product already has most of the step-`1/2/3` source-understanding layer:

- source setup and crawl
- primary and secondary evidence ingestion
- normalized context-pack generation
- graph normalization for the source company
- source reasoning to produce a `Source Brief`
- company-context graph sync into Neo4j
- secondary public evidence graph layer

### What Exists But Should Change Position

Internal query groups already exist, but they should move from:

- primary user-facing step

to:

- internal compiled retrieval abstraction generated from approved scope nodes

### What Is Missing

The main missing pieces are:

- a clean graph-derived structured packet handoff into deep research
- a dedicated `Expansion Brief` artifact
- bounded deep-research logic for expansion
- graph support for expansion nodes and statuses
- a `Scope Review` UI step
- compilation from approved nodes to internal query packs

## Recommended Product Architecture

### User-Facing Artifacts

The user should see:

1. `Source Brief`
2. `Expansion Brief`
3. `Scope`
4. `Universe`
5. `Dossiers`

### Internal Artifacts

The system should maintain:

- `CompanyContextGraph`
- `ExpansionBrief`
- `DiscoveryScope`
- `QueryPack`
- `UniverseRun`
- `VendorDossier`

## Recommended Data Model Additions

### `ExpansionBrief`

Suggested fields:

```json
{
  "adjacent_capabilities": [],
  "adjacent_customer_segments": [],
  "named_account_anchors": [],
  "geography_expansions": [],
  "generated_at": "2026-03-15T12:00:00Z"
}
```

### Graph Node Status

Each expansion node should support:

- `source_grounded`
- `corroborated_expansion`
- `hypothesis`
- `user_kept`
- `user_removed`
- `user_deprioritized`

### `DiscoveryScope`

Suggested structure:

```json
{
  "kept_capabilities": [],
  "kept_customer_segments": [],
  "kept_named_accounts": [],
  "kept_geographies": [],
  "removed_node_ids": [],
  "deprioritized_node_ids": [],
  "confirmed_at": "2026-03-15T12:10:00Z"
}
```

## Recommended Screen Design

### Screen 1: Source

Inputs:

- source website
- comparator websites
- supporting evidence URLs

### Screen 2: Source Brief

Shows:

- summary
- top nodes
- customer / partner proof
- evidence gaps

### Screen 3: Expansion

Shows:

- proposed adjacent capabilities
- proposed adjacent customer segments
- proposed named-account anchors
- proposed geographies
- confidence and evidence

### Screen 4: Scope

Shows:

- all approved and proposed nodes together
- keep / remove / deprioritize controls

### Screen 5: Universe

Shows:

- candidate companies
- light rationale
- citations
- keep / remove

### Screen 6: Enrichment

Shows:

- deep dossiers for kept candidates

## End-To-End 4TPM Walkthrough

### Input

The user enters:

- source website: `4TPM`
- comparator websites: a few nearby portfolio / wealth / securities workflow vendors

### Source Brief Outcome

The system concludes:

- 4TPM appears strongest in portfolio-management and securities workflow software
- buyer evidence points to private banks, brokers, and asset managers
- workflow evidence points to front-office and back-office securities operations
- integration evidence points to APIs and partner surfaces

### Expansion Brief Outcome

The system proposes:

- adjacent capabilities:
  - `client reporting`
  - `voting rights`
  - `corporate actions`
- adjacent customer segments:
  - `fund administrator`
  - `custodian`
- named-account anchors:
  - `BNP Paribas`
- geography expansions:
  - `Belgium`
  - `Luxembourg`

Each is marked as:

- corroborated
- or hypothesis

### Scope Review Outcome

The user keeps:

- `client reporting`
- `BNP Paribas`
- `Belgium`

The user removes:

- `voting rights`

because it feels too speculative for the first pass.

The user deprioritizes:

- `fund administrator`

because the current focus is private banking, not fund servicing.

### Universe Outcome

The system builds retrieval plans from the kept nodes and returns a universe of candidate companies with:

- websites
- light product evidence
- likely fit to private banking / asset-management workflows
- citations

### Enrichment Outcome

The user validates a subset of candidates and the system deepens them into structured dossiers.

## Final Recommendation

The product should not ask users to define retrieval lanes before the system has completed adjacency expansion.

The correct flow is:

- crawl and extract primary + secondary evidence
- normalize into `CompanyContextGraph`
- generate `Source Brief`
- hand off graph-derived packet + `Source Brief` to deep research
- run bounded deep research to generate `Expansion Brief`
- present graph nodes to the user for `Scope Review`
- compile approved nodes into internal discovery plans
- run `Universe`
- then do `Candidate Enrichment`

This keeps the product:

- evidence-first
- explicit about uncertainty
- easier for users to operate
- less dependent on users inventing retrieval structure too early
