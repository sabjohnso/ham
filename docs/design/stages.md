# Stage history

The library's algebraic core was built in eight stages
(2026-05-08 through 2026-05-12); five further stages extended
the worked-example surface and the Blasius pipeline through
2026-05-13. A post-build defensive-correctness pass on
2026-05-13/14 added the [`ham.contracts`](../api/contracts.md)
module and tightened two test/library guards.

Each entry below is a distilled timeline. The detailed per-stage
record lives in the project's private PLAN.org.

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

## Stage 9 — Volterra integro-differential support

[`ham.nonlinear`](../api/nonlinear.md) (Integral branch) +
[`examples/volterra.py`](../examples/volterra.md)

The first stage where the `NonlinearOperator` compiler grew a new
node type rather than refining an existing one.

- **Integral branch in `_compile_integral`** handles
  `sp.Integral(f(u, ...), (s, 0, indep))` — the canonical Volterra
  form. Lower bound must be 0; upper bound must be the indep
  variable; the integrand must depend on the dummy only through
  `dependent(dummy)`. Other shapes raise `NotImplementedError`.
- **Substrate identity preserved.** The new branch is a thin
  wrapper around the existing `map_coeffs` machinery — the
  integro-differential character flows through the same Cauchy
  arithmetic as polynomial-\(N\), with no algebra invented.
- **Worked example.** Volterra single-species population model
  with `u' = κ u (1 - u - ∫_0^t u ds)`. HAM polynomial degree at
  order \(M\) is *2M* for this problem (one degree from \(L^{-1}\),
  one from the integral). The residual norm is /not/ strictly
  monotone in \(M\) here; the test honestly relaxes to "M=6 at
  least 100× smaller than M=1".

## Stage 10 — Blasius (truncated-domain, polynomial basis)

[`examples/blasius.py`](../examples/blasius.md)

The first worked example where \(\hbar = -1\) /diverges/ — the
pedagogical bite of Liao's Rule of Solution Expression. The
asymptotic BC \(f'(\infty) = 1\) is replaced with
\(f'(\eta_{\max}) = 1\) at a large finite cap, so library
support for asymptotic BCs is still deferred.

- **u_0 = η²/(2 η_max)** satisfies all three truncated BCs by
  construction.
- **Validity gate is closeness to Howarth's f''(0) ≈ 0.4696**,
  not L² residual norm — the latter has a false plateau at small
  positive \(\hbar\) where the partial sum collapses near \(u_0\)
  but the residual is tiny.

## Stage 11 — Blasius via exponential basis (true asymptotic BC)

[`examples/blasius_exponential.py`](../examples/blasius-exponential.md)

The Liao-canonical setup: \(L = d^3/d\eta^3 - \alpha^2 \cdot d/d\eta\),
\(u_0 = \eta - 1/\alpha + e^{-\alpha\eta}/\alpha\), and a custom
inverter that handles the resonant `η·exp(-αη)` RHS.

- **`LinearOperator` extended to accept `BoundaryCondition(point=sp.oo)`**.
  Sympy's `dsolve` resolves the asymptotic BC for cheap RHSes; a
  zero-free-constants workaround handles the resonant case.
- **\(\alpha = 1\) (fixed) gave \(|f''(0) - \text{Howarth}| \approx 5\times 10^{-3}\)** at \(M = 2\).

## Stage 12 — Multi-parameter optimisation

[`ham.diagnostics.optimal_parameters`](../api/diagnostics.md) +
two-parameter (ℏ, α) tuning in
[`examples/blasius_exponential.py`](../examples/blasius-exponential.md)

\(\alpha\) becomes a sympy symbol carried through the solve;
`optimal_parameters` generalises `optimal_hbar` to grid-search
substitution dictionaries.

- **At \(M = 2\) the 2D optimum gave error ≈ 4×10⁻⁴** — an order
  of magnitude tighter than the Stage 11 single-parameter best.

## Stage 13 — Closed-form basis-aware Blasius inverter

[`examples/blasius_inverter.py`](../examples/blasius-exponential.md#the-closed-form-basis-aware-inverter-stage-13)

The Stage-11 `sympy.dsolve` inverter scaled poorly with \(M\) (M=3
≈ 12 s, M=4 prohibitive). Stage 13 replaces it with a decompose-
and-cache strategy that brings M=3 down to ~3 s and unlocks M=4.

- **Decompose** RHS into basis terms \(c \cdot \eta^j \cdot e^{-k\alpha\eta}\).
- **Cache** \(L^{-1}\) of each basis element via `functools.cache`.
- **Assemble** the inverse by linearity.
- **At \(M = 3\) the 2D optimum gives \(|error| \approx 1.6\times 10^{-4}\)** —
  six times tighter than the Stage 12 \(M = 2\) best, and over
  100× tighter than the polynomial-basis Stage 10 \(M = 5\) best.

## Post-build evolution (2026-05-13/14)

Driven by a comprehensive review documented in `Review.org`
at the project root. Five defensive-correctness items landed
serially:

| # | Concern | Commit | Effect |
| --- | --- | --- | --- |
| 1 | Integral branch silently miscompiled non-canonical integrands | `7c747b5` | `_compile_integral` now refuses integrands with explicit `indep` dependence outside `dependent(dummy)` |
| 2 | Causality property test only perturbed the top coefficient | `c8c3a85` | Strict per-index Hypothesis property test (compiler was already causal; the law is now asserted) |
| 3 | No CI pipeline | `f9a450e` | GitHub Actions runs ruff / format / mypy / pytest on push and PR |
| 4 | Linearity of `L.action` was contract-only | `8d0886a` | New module [`ham.contracts`](../api/contracts.md) with `verify_linearity` + `LinearityViolation(ValueError)` |
| 5 | `HamProblem` did not validate `u_0` against original BCs | `5e9590f` | `ham.contracts.verify_initial_guess` + `InitialGuessViolation`; each worked example exposes `ORIGINAL_BCS` |

The post-build pass added one new public module
(`ham.contracts`), no new examples, and no breaking changes.
Cumulative test count grew 226 → 252.

## Substrate-parametrisation arc (S0-S9, 2026-05-13/17)

The Stage 1-13 line settled the algebraic core for the sympy
substrate. To make the same scaffolding serve the **Spectral HAM**
substrate (numpy arrays on a Chebyshev grid) without forking the
library, the next nine stages refactored every module to be generic
over the coefficient type, then added the spectral substrate on top.

The architectural target was set in `PLAN.org`: rather than building
a parallel `ham.spectral` package, parametrise each existing module
over a `Backend[C]` Protocol. Property tests covering each algebraic
law would then run unchanged across substrates, with only the
equality comparator changing at the verification site.

| Stage | Headline change | Module(s) touched |
| --- | --- | --- |
| S0 | `Backend[C]` Protocol (six ops: `zero`, `one`, `lift_xonly`, `diff_x`, `integrate_x`, `normalize`) + `SympyBackend`. Equality stays out of the Backend (PLAN.org D-4). | new [`ham.backend`](../api/backend.md) |
| S1 | `Series[C: SupportsCoefficientArith]` generic; `QSeries` becomes a sympy-baked subclass; methods use `Self` returns and `type(self)` construction so subclasses propagate. | [`ham.series`](../api/series.md) |
| S2 | `NonlinearOperator._compile` reads `phi.backend` and dispatches all leaves through it. Solver constructs `SympyBackend(problem.L.var)` so examples on `t`/`eta` integrate w.r.t. the right variable. | [`ham.nonlinear`](../api/nonlinear.md), [`ham.solver`](../api/solver.md) |
| S3 | `LinearOperator[C]` generic; `sympy_dsolve_inverter` exposed as a public factory; `verify_linearity` takes an injectable `equal` comparator (D-4). | [`ham.operator`](../api/operator.md), [`ham.contracts`](../api/contracts.md) |
| S4 | `Backend.normalize` (sympy = `sp.expand`, spectral = identity); the solver's two former `sp.expand` sites route through it. | [`ham.backend`](../api/backend.md), [`ham.solver`](../api/solver.md) |
| S5 | `Grid` Protocol + `ChebGLGrid` (Trefethen `cheb.m` + `clencurt.m`); `SpectralBackend(grid, indep, scalar)` over `scalar ∈ {"float", "sympy"}` (D-1's dual-scalar architecture). | new [`ham.grids`](../api/grids.md), new [`ham.spectral`](../api/spectral.md) |
| S6 | `spectral_linear_operator` parses linear-in-\(u\) sympy expressions into the dense \(L\) matrix; `spectral_inverter` imposes BCs by row replacement. `verify_linearity` with `equal=np.allclose` is the D-4 demonstration. | [`ham.spectral`](../api/spectral.md) |
| S7 | `HamProblem[C]` and `HamSolution[C]` generic; `solve(problem, order, backend=...)` is the end-to-end substrate-agnostic entry. End-to-end smoke tests on `u' = u, u(0) = 1` pass on both scalars. | [`ham.deformation`](../api/deformation.md), [`ham.solver`](../api/solver.md) |
| S8 | Diagnostics dispatch on backend: `residual` via apply_series order-0 trick on the spectral side; `residual_l2_squared(grid=...)` uses Clenshaw-Curtis; `hbar_curve_at_sweep` builds the float-substrate ℏ-curve by re-running the solver per ℏ. | [`ham.diagnostics`](../api/diagnostics.md) |
| S9 | `homotopy_pade` ships sympy-only for the spectral release; spectral inputs raise `NotImplementedError` with a pointer to the block-structured-Padé follow-up. | [`ham.pade`](../api/pade.md) |

### Design decisions recorded along the way

PLAN.org tracked four design questions that were resolved early and
shaped the rest of the arc:

- **D-1 — Dual-scalar `SpectralBackend`.** The spectral substrate
  supports both `scalar="float"` (classical SHAM; ℏ-curves via
  external sweep) and `scalar="sympy"` (ℏ inline inside every grid
  entry as an `ndarray[object]` of sympy expressions). The two
  paths share every line of code except the linear solver in
  `integrate_x` / `spectral_inverter`.
- **D-2 — Single generic `Series[C]`.** Reviewing the original
  `ham.series` line by line showed every method was either pure
  q-arithmetic or routed through Python `+ - * /int`, which both
  substrates support. One generic class with a `QSeries` back-compat
  shim was cleaner than forking the type.
- **D-3 — Grid Protocol with identity-pinned instances.** Each
  `HamProblem` carries one `Grid` instance shared across the
  `SpectralBackend`, the `spectral_linear_operator`, and the
  diagnostic quadrature. Cached \(D^k\) and quadrature weights are
  computed once per grid.
- **D-4 — Equality is an injected comparator, not a Backend method.**
  `verify_linearity` takes an `equal: Callable[[C, C], bool]` kwarg
  defaulting to the sympy `sp.expand`-based comparator. Spectral
  callers pass `np.allclose` (float scalar) or a sympy element-wise
  closure (sympy scalar). `LinearOperator` stays substrate-agnostic.

### What the arc delivered

- Two substrates ship from one solver loop: 338 tests pass; ruff +
  ruff format + mypy strict clean across `ham/`, `tests/`,
  `examples/`.
- Algebra-driven design dividend: the substrate laws (zero / one
  identities, diff∘integrate = id, linearity of `L`, the commuting
  diagram `[q^k] N.apply_series(phi) = [q^k] N.apply_scalar(Σ
  phi.coeff(j) q^j)`) are pinned by Hypothesis property tests against
  every Backend fixture. Adding a new substrate runs the same laws
  for free.
- The S7 smoke test (`u' = u, u(0) = 1` on `[0, 1]`) confirms the
  three backends produce the same answer to substrate-appropriate
  tolerance, with the sympy-scalar spectral run reproducing the
  float-scalar result after `subs(ℏ, -1)`.

## Where to go next

The Stage-1 to Stage-13 line and the S0-S9 substrate-parametrisation
arc together cover the public surface as it ships. Remaining
extension directions live above:

- **Rational-Chebyshev grid** for Blasius-class problems on
  \([0, \infty)\). The Grid Protocol is the plug-in point; the
  finite-domain spectral redo of Blasius is deferred per the S7
  plan note.
- **Block-structured spectral Padé.** S9's deferred follow-up:
  the linear system has grid-vector entries; `homotopy_pade` would
  need a substrate-aware Padé construction.
- **Multi-point Padé and Hermite-Padé** on the sympy substrate —
  generalisations of Stage 8 that compose on top of `HamSolution.phi`
  without touching the upstream stages.
- **Additional contracts.** [`ham.contracts`](../api/contracts.md)
  has room for nonlinear-polynomial form verification and
  invertibility-on-image checks for `L`; see the module docstring's
  extension policy.

Each new direction follows the pattern Stages 1-13 and S0-S9
established: a design conversation in `PLAN.org`, sub-stages with
red-green-refactor commits, algebraic identities pinned by property
tests, and a PLAN.org-style record of what was decided
and why.
