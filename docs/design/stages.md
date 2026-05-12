# Stage history

The library was built in eight stages over four days
(2026-05-08 through 2026-05-12), each a fully-tested unit of
functionality with its own commit chain and design conversation.
This page is a distilled timeline. The detailed per-stage record
lives in the project's private PLAN.org.

## Stage 1 — Truncated power series in q

[`ham.series`](../api/series.md)

The substrate of every other module. A `QSeries` is a truncated formal
power series in \(q\) with sympy expressions as coefficients in the
independent variable.

- Polynomial-product semantics for multiplication (the Cauchy product
  grows the order), with explicit `trunc(n)` for working-order
  truncation. The alternative — modulo-\(q^{N+1}\) semantics — would
  have hidden every truncation site; explicit `trunc` keeps the
  authoring intent visible.
- `map_coeffs(f)` exposed as the functor map. Stage 2's
  `apply_series` and Stage 3's polynomial-N evaluator both compose
  on top of this.
- Abelian group + ring laws verified by Hypothesis property tests.

**Cross-check.** The Cauchy-product coefficient formula
`[q^k] (a · b) = Σ a.coeff(j) b.coeff(k-j)` is asserted directly
(not just commutativity).

## Stage 2 — Auxiliary linear operator \(L\)

[`ham.operator`](../api/operator.md)

The load-bearing piece for the solver loop: every order calls
`L.invert` once, so its design determines what HAM can actually
compute.

- **Boundary conditions are a first-class field**, not a per-call
  argument. Every `invert` call runs against the homogeneous BCs
  declared at construction time (since \(u_0\) already satisfies the
  originals).
- **Inversion strategy is pluggable**. Canonical hand-coded inverters
  (`antiderivative` for \(L = d/dt\) with \(u(0) = 0\)) live alongside
  a `sympy.dsolve`-backed fallback.

**Sub-stages.** 2a forward action + `apply_series`; 2b
`BoundaryCondition` type + canonical `antiderivative`; 2c
`sympy.dsolve`-backed default inverter. Three commits, each with
property tests for the algebraic laws (linearity, \(L \circ L^{-1} =
\text{id}\) on image, \(L^{-1} \circ L = u - u(0)\) modulo kernel).

## Stage 3 — Nonlinear operator \(N\)

[`ham.nonlinear`](../api/nonlinear.md)

A *compiler* from "sympy expression in \(u\) and its \(x\)-derivatives"
to "QSeries arithmetic against \(\varphi\)". Almost no new algebra —
the Cauchy product structure of Liao's \(R_m\) falls out of `QSeries`
multiplication automatically.

- **Polynomial-in-\(u\) regime** prioritised: tree-walker dispatches on
  `Add`, `Mul`, `Pow(int ≥ 0)`, `Derivative`, and the
  `dependent(indep)` node.
- **Eager truncation** to `phi.order` after each multiplication.
  Without it, a cubic-in-\(u\) operator on \(\varphi\) at order \(M\)
  would do Cauchy work bounded by \(3M \times 3M\) instead of
  \(M \times M\).
- **Transcendentals raise `NotImplementedError`** with the offending
  subexpression named — and the workaround (truncated Taylor in \(u\)
  before construction) suggested.

**Sub-stages.** 3a `apply_scalar`; 3b polynomial-in-\(u\); 3c
\(x\)-derivatives of \(u\); 3d transcendental rejection. Four
commits.

## Stage 4 — Deformation-equation builder

[`ham.deformation`](../api/deformation.md)

Bundle the HAM problem data into one noun, expose the right-hand side
of the m-th order deformation equation.

- **`HamProblem` carries five fields**: \(L\), \(N\), \(H\), \(\hbar\),
  \(u_0\). Math letters in field names so the implementation reads as
  Liao's notation.
- **Taylor-coefficient form of \(R_m\)** chosen over the unnormalised
  \(q\)-derivative form. The PLAN.org note that conflated the two
  conventions was fixed here.
- **Validation at the boundary.** `r_m(phi, m)` raises for
  \(m < 1\) and for \(\text{phi.order} < m - 1\) — silent zeros from
  an under-built \(\varphi\) would be a quiet bug.

**Cross-check.** \(R_1 = N[u_0]\) for *any* polynomial \(N\) — Liao's
m=1 closed form, asserted directly.

## Stage 5 — HAM solver loop

[`ham.solver`](../api/solver.md)

The forward sweep: iterate `m = 1..M`, build \(\varphi\) incrementally,
solve `L.invert(rhs)` per step, return the bundled `HamSolution`.

- **`HamSolution` retains the problem and the QSeries**, so Stages 6
  and 8 can access both per-order coefficients and the ℏ symbol.
- **`partial_sum()` keeps ℏ symbolic**; `evaluate_at_hbar(value)`
  substitutes. A single solve can be evaluated at any ℏ.

**End-to-end cross-checks.** The first cross-stage validation:
\(u' = u, u(0) = 1\) matches `exp(x)` Taylor to order 4, and
\(u' = u^2, u(0) = 1\) matches `1/(1-x)` Taylor to order 3, both at
\(\hbar = -1\). A single error anywhere in Stages 1-4 would
propagate to a Taylor mismatch here.

## Stage 6 — Convergence diagnostics

[`ham.diagnostics`](../api/diagnostics.md)

Operational tools for defending a convergence claim: residual, two
norm flavors, ℏ-curve, optimal-ℏ.

- **Both L² flavors** ship (symbolic-interval and discrete-sample),
  per the PLAN.org "decide per example" directive.
- **The ℏ-curve is returned as a polynomial**, not a plot. The
  polynomial *is* the curve; rendering is the caller's job.
- **`optimal_hbar` grid-searches** a caller-supplied norm function.
  Robust against degenerate sympy minimisation that brittle
  continuous methods would hit.
- **Validity gate deferred to Stage 7**, where each worked example
  composes the primitives into its own threshold predicate.

**Sub-stages.** 6a residual primitive; 6b norm flavors; 6c ℏ-curve +
optimal-ℏ. Three commits.

## Stage 7 — Worked examples

`examples/` and `tests/examples/`. See the
[worked-examples gallery](../examples/index.md) for the narrative.

- **Quadratic drag** (`v' = 1 - v^2`, exact \(\tanh\)) — surfaces the
  HAM adaptive-ℏ phenomenon: \(\hbar = -1/2\) sometimes beats
  \(\hbar = -1\) in L² on \([0, 1]\).
- **Logistic** (\(u' = u(1-u)\), \(u_0 = 1/2\), exact sigmoid) — first
  example with a non-trivial initial guess.

Volterra and Blasius deferred. Blasius is gated on a library
extension to represent asymptotic BCs in `LinearOperator`.

## Stage 8 — Homotopy-Padé acceleration

[`ham.pade`](../api/pade.md)

Pure post-processing on `HamSolution.phi`. Build a Padé approximant
in \(q\), evaluate at \(q = 1\).

- **Linear-system Padé construction** via `sympy.Matrix.LUsolve`.
- **Substitute ℏ early**, not late. Late substitution can trade a
  vanishing symbolic determinant for a silent `nan`; early
  substitution surfaces the degeneracy as
  `NonInvertibleMatrixError`. This bug surfaced during the
  red-green-refactor of `test_pade_singular_denominator_propagates_sympy_error`.
- **Geometric-series cross-check.** \([0/1]\) Padé recovers
  \(1/(1-x)\) from two HAM coefficients on \(u' = u^2, u(0) = 1\).
  The headline result.

## Where to go next

Three obvious extension directions are *not yet implemented* and
remain in scope:

- **Volterra worked example.** Polynomial \(N\), no library changes
  needed — fits the existing `examples/<name>.py` template.
- **Blasius BVP.** Requires extending `LinearOperator.bcs` to admit
  asymptotic conditions like \(f'(\infty) = 1\), and probably a
  dedicated inverter that knows how to solve the resulting BVP.
- **Multi-point Padé and Hermite-Padé.** Generalisations of Stage 8
  that compose on top of `HamSolution.phi` without touching the
  upstream stages.

Each follows the pattern Stages 1-8 established: a design conversation
in the PR description, sub-stages with red-green-refactor commits,
algebraic identities pinned by property tests, and PLAN.org-style
record of what was decided and why.
