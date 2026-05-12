# Design notes

The library was built through eight stages of test-driven development,
each with explicit design conversations before code. The decisions
made along the way — the algebraic surfaces, the boundary between
stages, the failure modes preserved rather than smoothed over — are
collected here.

- **[Tenets](tenets.md)** — the design principles that shaped every
  module: Algebra-Driven Design, Normalized Systems Theory,
  Test-Driven Development, functional-core / imperative-shell,
  programming to interfaces. With concrete examples from the
  codebase showing each principle in action.
- **[Stage history](stages.md)** — a distilled timeline of stages
  1 through 8: what each delivered, the design decisions made at
  each, the cross-checks that pinned the work down. Cross-linked
  to the API reference and concept guide.

These pages are not strictly necessary for *using* the library — the
[tutorial](../tutorial.md) and [API reference](../api/index.md) are
enough for that. They exist to make the design legible to readers who
want to understand *why* the library is shaped the way it is, or who
want to add a new stage (Volterra, Blasius, multi-point Padé)
consistent with the existing architecture.
