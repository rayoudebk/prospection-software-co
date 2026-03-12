# Repository Agent Rules

## Polling And Completion Effects

- Treat any effect that reacts to polled job state as an idempotent transition handler, not as a generic render side effect.
- When a job reaches a terminal state such as `completed`, `failed`, or `cancelled`, run completion logic at most once per stable identifier such as `jobId`.
- Do not rely on inline callback identity inside polling completion effects. If the caller passes a callback, use a stable callback primitive or guard the effect with a `useRef` keyed by the terminal resource id.
- If a completion effect invalidates queries, triggers mutations, or writes analytics, add an explicit once-only guard before shipping.
- Before merging polling code, check the route fan-out caused by a single completion path and confirm it cannot recursively trigger itself through invalidation and rerender.
