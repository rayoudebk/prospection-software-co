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

## Railway Deployment Discipline

- Do not stack Railway deploys for the same service.
- Before triggering a new Railway deploy, check the current service status/queue first.
- If a Railway deployment for that service is already `QUEUED` or `BUILDING`, do not start another deploy.
- Prefer one in-flight deploy per service at a time, then verify whether it reaches `SUCCESS` before retrying.
- When a deployment backlog exists, stop creating new deployments and wait for the queue to drain unless there is a clear way to cancel older ones.
- Treat repeated manual `railway deployment up` calls as a last resort because they can create a deploy queue that hides whether the latest code is actually serving traffic.
- Before deploying, prefer `railway deployment list --service <name>` or `railway service status --all` to confirm there is only one active in-flight deployment.
- Treat `railway down` as unreliable for clearing stuck builds. Verify the deployment list after using it instead of assuming the in-flight build was cancelled.
- Keep Railway service Dockerfile selection explicit via `RAILWAY_DOCKERFILE_PATH`; do not rely on Railway inferring the correct Dockerfile in a multi-service repo.
- Production Dockerfile mapping:
  `api` -> `backend/Dockerfile`
  `worker` -> `backend/Dockerfile` while `CHROME_MCP_ENABLED=false`
  `worker` -> `backend/Dockerfile.worker` only when production intentionally enables local Playwright/Chromium browser fallback
- Prefer the lean Dockerfile for production services unless browser rendering is explicitly required in that service. Heavy browser installs should be opt-in because they materially increase deploy time and make cache misses much more expensive.

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
