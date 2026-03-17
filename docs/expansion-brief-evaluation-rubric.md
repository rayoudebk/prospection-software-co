# Expansion Brief Evaluation Rubric

## Purpose

Use this rubric to evaluate whether an `Expansion Brief` is actually improving discovery quality for a workspace.

The goal is not to reward polished prose. The goal is to answer:

- did the expansion stage find the right adjacent lanes
- did it describe them precisely enough to matter
- did it generate useful company seeds and search lanes
- did it separate evidence from judgment well enough to trust downstream use

This rubric is designed to work across workspaces such as `Hublo`, `4TPM`, and future source companies.

## Scoring

Score each dimension from `1` to `5`.

- `1`: poor
- `3`: usable but weak
- `5`: strong and discovery-useful

## Dimensions

### 1. Adjacency Relevance

Does the report identify the right adjacent lanes around the source company?

Questions:

- Are the boxes genuinely one-hop adjacent to the source company?
- Are the most strategically important lanes present?
- Are narrow or weakly related adjacencies demoted appropriately?

Signals of a strong report:

- a small set of plausible lanes
- no obvious category sprawl
- priority ordering feels defensible

### 2. Workflow Specificity

Does each adjacency box describe an actual workflow rather than a vague category?

Questions:

- Can a human understand what operators do in this lane?
- Are triggers, actions, systems, and failure costs clear?
- Is day-to-day use distinguished from management or exception use?

Signals of a strong report:

- category labels are backed by workflow anatomy
- vague boxes like `VMS` or `compliance` become operationally concrete

### 3. Criticality Usefulness

Do the criticality judgments help rank discovery lanes?

Questions:

- Are workflow criticality and switching-cost judgments believable?
- Do the judgments explain why a lane matters for discovery?
- Are they framed as hypotheses rather than facts?

Signals of a strong report:

- `core_adjacent` lanes feel materially more important than `meaningful_adjacent`
- the justification is evidence-backed rather than generic

### 4. Evidence Quality

Is the reasoning well supported?

Questions:

- Are evidence items relevant and specific?
- Do they support the right claims?
- Is it possible to tell source-grounded observations from analyst inference?

Signals of a strong report:

- claim-level evidence, not just generic URLs
- multiple evidence types when needed
- low obvious slop or citation mismatch

### 5. Company Seed Usefulness

Do the named companies improve downstream discovery?

Questions:

- Are seeds relevant to the adjacency lanes?
- Are they varied enough to open the market map rather than duplicating one subsegment?
- Would you want to crawl these companies next?

Signals of a strong report:

- seeds clearly map to specific boxes
- at least some seeds are non-obvious but defensible
- the list is compact and useful

### 6. Retrieval Usefulness

Does the report help the universe step search better?

Questions:

- Are query seeds concrete and targeted?
- Do named accounts and geographies help narrow the search space?
- Would the report improve retrieval precision over the sourcing brief alone?

Signals of a strong report:

- query seeds are box-specific
- named accounts are actually useful anchors
- geographies are bounded and relevant

### 7. Trend / Shift Usefulness

Do any technology or market shifts actually improve discovery?

Questions:

- Are shifts tied to specific adjacency lanes?
- Do they change which companies or capabilities should be searched?
- Are they evidence-backed rather than trend theater?

Signals of a strong report:

- few but useful shifts
- each shift affects discovery priorities or seed selection

### 8. Multilingual Normalization

Does the report handle multilingual markets cleanly?

Questions:

- Are French and English concepts collapsed into the same lane when appropriate?
- Is local-language source richness preserved?
- Are canonical labels stable enough for graph use?

Signals of a strong report:

- no concept drift caused by translation
- local phrasing is preserved as alias/query context
- graph labels remain consistent

## Outcome Bands

### 36-40

Strong expansion brief.

Use it directly for scope review and universe planning.

### 28-35

Useful but needs review.

Use it, but expect manual pruning or clarification.

### 20-27

Weak expansion brief.

Some useful lanes may exist, but the artifact is too noisy or underspecified to trust directly.

### Below 20

Poor expansion brief.

Do not rely on it for discovery without re-running or revising prompt/scaffolding.

## Fast Review Checklist

For a quick pass, answer:

- Are the top 3 adjacency boxes actually the right ones?
- Can I explain what operators do inside each box?
- Are the company seeds good enough to crawl next?
- Did the report add something meaningful beyond the sourcing brief?
- Do the citations look believable and relevant?

If at least four of the five answers are `yes`, the expansion stage is probably adding real value.
