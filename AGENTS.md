# Repository Agent Rules

## Product Stage And Change Bias

- Treat this project as a solo-founder plus Codex workflow with minimal process overhead by default.
- Assume this product currently has `0` external users and is still being actively shaped.
- Do not add extra compatibility or deprecation precautions just to preserve old behavior when a cleaner replacement is available.
- Prioritize consistency across labels, field names, schemas, and product language to avoid data sprawl, naming drift, and obsolete concepts lingering in the system.
- If an existing field, label, or model name is misleading or incorrect, prefer renaming it to the clearer canonical name instead of adding another parallel variant.
- When renaming data fields or models, apply the corresponding migration and code updates so downstream consumers see the clearer name end to end.
- Prefer removing or replacing outdated structures rather than keeping legacy paths around unless there is a concrete reason they are still needed.
- When replacing a prompt, workflow, field, or taxonomy layer, remove the old path in the same change unless there is a documented reason to keep it.
- If a fallback path must remain, it must be explicit in both API and UI as a degraded mode. Do not leave silent fallbacks that look equivalent to the primary path.
- Do not keep parallel prompts or duplicate summary-generation paths for the same artifact. Each product artifact should have one canonical generation path.
- Keep coordination and delivery overhead low unless the user explicitly asks for a heavier process.

## CLI First For Platform Debugging

- When working with third-party platforms or hosted services, check for a CLI workflow first and prefer it over browser or dashboard debugging when it offers equivalent control.
- Use CLI tools as much as possible when debugging or operating Railway, Vercel, and Chrome-related workflows because they are faster, more reproducible, and easier to inspect.

## Git Workflow

- Do not introduce a PR-based workflow for this project right now.
- Do not create branches unless there is a specific reason the user asks for one.
- Default to working directly in the current branch and push or deploy to git directly when needed.

## Polling And Completion Effects

- Treat any effect that reacts to polled job state as an idempotent transition handler, not as a generic render side effect.
- When a job reaches a terminal state such as `completed`, `failed`, or `cancelled`, run completion logic at most once per stable identifier such as `jobId`.
- Do not rely on inline callback identity inside polling completion effects. If the caller passes a callback, use a stable callback primitive or guard the effect with a `useRef` keyed by the terminal resource id.
- If a completion effect invalidates queries, triggers mutations, or writes analytics, add an explicit once-only guard before shipping.
- Before merging polling code, check the route fan-out caused by a single completion path and confirm it cannot recursively trigger itself through invalidation and rerender.
