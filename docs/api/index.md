# API reference

Each page is auto-extracted from its module's docstrings via
`mkdocstrings`. After the S0-S9 parametrisation, the library is
organised around a **coefficient substrate** abstraction
(`ham.backend.Backend[C]`) over which every other module is generic:
the classical symbolic-HAM substrate (`C = sympy.Expr`) and the
spectral SHAM substrate (`C = numpy.ndarray`) share the same algebraic
core, solver loop, and diagnostic surface.

## The substrate

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.backend`](backend.md) | The substrate Protocol — six operations (`zero`, `one`, `lift_xonly`, `diff_x`, `integrate_x`, `normalize`) that bridge the algebraic core to any concrete coefficient type. Equality lives at the verification site, not on the backend. | `Backend`, `SympyBackend` |

## Algebraic core (substrate-generic)

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.series`](series.md) | Truncated formal power series in \(q\), generic over the coefficient substrate via `Series[C: SupportsCoefficientArith]`. `QSeries` is the sympy-baked back-compat shim that pre-binds a default sympy backend. | `Series`, `QSeries`, `SupportsCoefficientArith` |
| [`ham.operator`](operator.md) | The auxiliary linear operator \(L\), generic via `LinearOperator[C]`. First-class boundary conditions; the sympy-substrate fallback inverter is the public `sympy_dsolve_inverter` factory. | `LinearOperator`, `BoundaryCondition`, `sympy_dsolve_inverter`, `antiderivative` |
| [`ham.nonlinear`](nonlinear.md) | The nonlinear operator \(N\) — a sympy-tree compiler whose tree-walker dispatches through `phi.backend` so the leaves use whichever substrate `phi` carries. | `NonlinearOperator` |

## Deformation and solver (substrate-generic)

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.deformation`](deformation.md) | The HAM problem statement, generic via `HamProblem[C]`. `rhs_m` lifts `hbar · H` through `phi.backend.lift_xonly` so the product stays in `C` for both substrates. | `HamProblem`, `chi_m` |
| [`ham.solver`](solver.md) | The forward sweep, generic via `solve(problem, order, backend=...)`. When `backend` is omitted, defaults to `SympyBackend(problem.L.var)` for back-compat. | `solve`, `solve_step`, `HamSolution` |

## Diagnostics and acceleration

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.diagnostics`](diagnostics.md) | Residual, L² / discrete norms, Liao's ℏ-curve, grid-search parameter optimisation. Dispatches on backend: sympy uses `sp.integrate` for L², spectral uses `grid.quadrature_weights`. `hbar_curve_at_sweep` builds the float-spectral ℏ-curve by re-running the solver per ℏ value. | `residual`, `residual_l2_squared`, `residual_discrete_sum_of_squares`, `hbar_curve_at`, `hbar_curve_at_sweep`, `optimal_hbar`, `optimal_parameters` |
| [`ham.pade`](pade.md) | Homotopy-Padé acceleration in \(q\). Currently sympy-only; spectral substrate is rejected with a tracking pointer to the block-structured follow-up. | `homotopy_pade` |

## Algebraic contracts

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.contracts`](contracts.md) | Opt-in algebraic-contract checkers users can call at problem-construction time. `verify_linearity` takes an injectable `equal` comparator (sympy `sp.expand`-based default; spectral callers pass `np.allclose` or a sympy element-wise checker). | `verify_linearity`, `verify_initial_guess`, `LinearityViolation`, `InitialGuessViolation` |

## Spectral substrate (SHAM)

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.grids`](grids.md) | The `Grid` Protocol — nodes, differentiation matrix, quadrature weights, domain, memoised \(D^k\). Two concrete implementations ship: `ChebGLGrid` for finite intervals, and `RationalChebGrid` for \([0, \infty)\) via the algebraic map (minimal: differentiation only; rational-Cheb quadrature is a tracked follow-up). The protocol is small so further families (Fourier, Legendre, mixed bases) plug in without touching consumers. | `Grid`, `ChebGLGrid`, `RationalChebGrid` |
| [`ham.spectral`](spectral.md) | `SpectralBackend(grid, indep, scalar)` is the `Backend[np.ndarray]` for SHAM, generic over `scalar ∈ {"float", "sympy"}`: float for classical SHAM, sympy-scalar for the symbolic-ℏ variant. `spectral_linear_operator` parses a linear-in-\(u\) sympy expression into the dense \(L\) matrix; `spectral_inverter` imposes BCs by row replacement, displacing additional BCs at the same boundary node to adjacent rows (Trefethen Program 30 convention). | `SpectralBackend`, `Scalar`, `spectral_linear_operator`, `spectral_inverter` |

## What you won't find here

- **Plotting.** The library produces sympy expressions and numpy
  arrays; render externally.
- **Numerical drivers.** `optimal_hbar` evaluates the supplied
  `norm_fn` at concrete `hbar` values via Python `float`; if the
  problem demands `scipy.optimize`-grade tooling, compose externally.
  For the spectral float backend, build the ℏ-curve via
  `hbar_curve_at_sweep` and reduce with `min`.
- **Validity gates.** Convergence claims belong to the application —
  each example provides its own `is_convergent` predicate using the
  primitives above; the library does not ship a generic one. See
  [Convergence](../concepts/convergence.md) for the rationale.

## Conventions across every module

- The coefficient substrate `C` is generic over
  `SupportsCoefficientArith` (a Protocol defined in `ham.series`)
  naming the arithmetic operations the algebraic core uses. Both
  `sympy.Expr` and `numpy.ndarray` satisfy it structurally.
- The convergence-control parameter `hbar` is **kept symbolic**
  throughout `solve` on the sympy / sympy-scalar substrates.
  Substitute via `solution.evaluate_at_hbar(value)` (sympy only) or
  by passing `hbar_value=value` to diagnostics / Padé functions.
  On the spectral float substrate, `hbar` is pre-substituted at
  `HamProblem` construction and the ℏ-curve is built externally via
  `hbar_curve_at_sweep`.
- Frozen `@dataclass(frozen=True)` is used for every algebraic
  type (`BoundaryCondition`, `LinearOperator[C]`, `NonlinearOperator`,
  `HamProblem[C]`, `HamSolution[C]`, `Backend[C]`). Construction is
  the only mutation; threading state through transformations means
  building a new value.
- All public functions are type-annotated and pass `mypy --strict`.
