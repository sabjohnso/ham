# Tenets

The library follows five guiding principles, each chosen for a
specific reason and each visible at concrete points in the code.

## 1. Algebra-Driven Design

> Model data and operations as algebras: define the laws they must
> satisfy, then verify those laws with property-based tests.

**Why it matters here.** HAM is fundamentally an algebra over formal
power series with sympy coefficients. Modelling each layer
(`QSeries`, `LinearOperator`, `NonlinearOperator`, ...) as an
algebraic type with explicit laws makes the dependencies between
layers obvious and the failure modes precise.

**Where you see it.**

- [`ham.series.QSeries`](../api/series.md) is a ring: addition is
  abelian, multiplication is the Cauchy product. The Hypothesis
  property tests in `tests/test_qseries.py` verify commutativity,
  associativity, distributivity, the multiplicative identity, and the
  q-differentiation/integration mutual-inverse laws.
- [`ham.nonlinear.NonlinearOperator.apply_series`](../api/nonlinear.md)
  satisfies *causality in q*: perturbing
  `phi.coeff(M)` leaves `apply_series(phi).coeff(0..M-1)`
  invariant. The test
  `test_apply_series_causality_in_q` asserts this directly.
- [`ham.deformation.HamProblem.r_m`](../api/deformation.md) at
  \(m = 1\) reduces to `N.apply_scalar(u_0)` — Liao's textbook
  closed-form for \(R_1\). Pinned by
  `test_r_m_at_one_is_n_of_u0`.

The tests don't just check "this code returns the value it returned
yesterday"; they check that the algebraic identity holds. A
refactor that breaks an identity breaks a test with a message that
names the law violated.

## 2. Normalized Systems Theory

> Small, well-defined interfaces; action and data version transparency;
> separation of concerns.

**Why it matters here.** Eight stages of HAM, each adding capability
on top of the previous, with no inter-stage coupling beyond the
public surfaces. A future Stage 7c (Blasius BVP) should extend
`LinearOperator` to support asymptotic BCs *without touching* the
solver, diagnostics, or Padé layers — and the architecture is
arranged so that's true.

**Where you see it.**

- The eight modules under `ham/` correspond to the build stages
  (Stage 7 is the `examples/` directory, not a `ham/` module; the
  later `ham.contracts` checker module was added after the staged
  build, as a defensive-correctness extension).
  Each module's surface is small: usually one or two top-level
  functions or classes plus their methods.
- [`ham.solver.solve_step`](../api/solver.md) composes
  `problem.rhs_m`, `problem.L.invert`, and `chi_m * phi.coeff(m-1)`.
  Each of those is a Stage-4 or Stage-2 public function; the solver
  reaches into none of their internals.
- The bug found while implementing Stage 8 — that late-substituting
  ℏ silently produced `nan` for degenerate Padé orders — was a
  *version transparency* violation. The fix (substitute ℏ early so
  `LUsolve` sees a concrete singular matrix) restored the contract
  that downstream consumers can trust the function's error mode.

## 3. Test-Driven Development

> Red → green → refactor, with failing tests as evidence that new
> code is demanded before any implementation lands.

**Why it matters here.** A symbolic algebra library has many internal
laws and few external observable behaviours. Without TDD discipline,
the tests would have lagged the code, and the algebraic identities
would have been claims rather than verifications.

**Where you see it.**

- Every stage's commit chain follows red-green-refactor. Look at the
  history for Stage 3's four sub-stages, each carrying a
  hand-derivation in the commit body that says *"this is the law
  the next commit must satisfy."*
- The Stage 8 commit body documents a subtle algorithmic bug found
  *during TDD*: the singular-matrix failure mode of `[2/2]` Padé on
  the geometric series only manifested *after* ℏ substitution.
  Without a test asserting the failure mode, the fix would have
  been a silent change with no record of why.

## 4. Functional core, imperative shell

> Pure functions for symbolic manipulation; isolate I/O at the edges.

**Why it matters here.** The whole library is post-hoc analysis of a
HAM solve — readers want to compute, substitute, plot, decide. Keeping
the core pure means the same `HamSolution` can be analysed at
multiple ℏ values, with multiple norms, in multiple notebooks, with
zero risk of shared mutable state.

**Where you see it.**

- Every algebraic type is a `@dataclass(frozen=True)`:
  `BoundaryCondition`, `LinearOperator`, `NonlinearOperator`,
  `HamProblem`, `HamSolution`. Modifications return new values.
- [`ham.diagnostics`](../api/diagnostics.md) contains zero I/O.
  `hbar_curve_at` returns a sympy polynomial; *plotting is the
  caller's job*. `optimal_hbar` does grid evaluation; *the choice of
  norm and grid is the caller's*. No `print`, no plot, no file write.
- ℏ is kept symbolic through every stage, substituted at the boundary
  by `HamSolution.evaluate_at_hbar` or the diagnostic functions'
  `hbar_value` parameter. A single `solve` output can be
  re-evaluated at any ℏ.

## 5. Program to interfaces, not implementations

> Code against abstract types, not concrete types.

**Why it matters here.** HAM has many degrees of freedom in the
choice of \(L\), \(N\), \(H\). The library's job is to provide the
*deformation machinery* without dictating *which* operators users
must pick.

**Where you see it.**

- [`ham.operator.LinearOperator`](../api/operator.md) accepts any
  `Callable[[sp.Expr], sp.Expr]` for its `action` field. The library
  does not require that action to be linear; *property tests verify
  the canonical d/dx case*, but a misbehaving caller-supplied action
  is the caller's problem.
- The `inverter` field on `LinearOperator` is also a callable. The
  library ships an explicit `antiderivative` inverter for
  \(L = d/dt\) with \(u(0) = 0\), and falls back to `sympy.dsolve`
  when no inverter is supplied. Users with their own fast inverter
  for a specific \(L\) plug it in.
- [`ham.diagnostics.optimal_hbar`](../api/diagnostics.md) takes a
  `norm_fn: Callable[[HamSolution, sp.Expr], sp.Expr]` rather than
  prescribing a norm. Bind `residual_l2_squared` with an interval, or
  `residual_discrete_sum_of_squares` with sample points, or write
  your own composition — the optimisation doesn't care.

## What these tenets cost

A few non-trivial costs are worth naming:

- **Verbosity.** A user who wants "give me the HAM solution at order
  M" has to assemble a `HamProblem` from five fields rather than
  calling `ham.solve(equation, initial)`. The verbose form pays off
  by making every knob explicit (and by letting users keep the
  symbolic structure of intermediate steps).
- **No global state.** The library has no module-level config — no
  `set_default_inverter`, no `set_default_norm`. Reasoned choice;
  global state would couple modules in ways the tenets explicitly
  reject.
- **No "validity gate" in the library.** Stage 6 deliberately refused
  to ship a generic `is_convergent(solution) -> bool`. Each worked
  example writes its own gate, calibrated to its own residual
  magnitudes and interval. See
  [convergence diagnostics](../concepts/convergence.md) for the
  rationale.

These costs were deliberate trade-offs, documented in the design
conversations recorded in the source-stage commit messages and in
PLAN.org.
