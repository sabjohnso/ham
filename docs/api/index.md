# API reference

Each page is auto-extracted from its module's docstrings via
`mkdocstrings`. The library splits cleanly along the eight stages
documented in [Design notes: Stage history](../design/stages.md);
the dependency order is bottom-up.

## The eight modules

### Algebraic core

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.series`](series.md) | Truncated formal power series in \(q\) with sympy coefficients. The substrate of every other module. | `QSeries` |
| [`ham.operator`](operator.md) | The auxiliary linear operator \(L\) with first-class boundary conditions and a pluggable inverter. | `LinearOperator`, `BoundaryCondition`, `antiderivative` |
| [`ham.nonlinear`](nonlinear.md) | The nonlinear operator \(N\) — a sympy-tree compiler that turns a sympy expression in \(u\) and its \(x\)-derivatives into `QSeries` arithmetic against the homotopy series. | `NonlinearOperator` |

### Deformation and solver

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.deformation`](deformation.md) | The HAM problem statement and the m-th order deformation equation builder. | `HamProblem`, `chi_m` |
| [`ham.solver`](solver.md) | The forward sweep iterating \(L^{-1}\) per step, returning a `HamSolution` carrying the partial sum and the per-order coefficients. | `solve`, `solve_step`, `HamSolution` |

### Diagnostics and acceleration

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.diagnostics`](diagnostics.md) | Residual, two flavors of L² norm, Liao's ℏ-curve, and grid-search parameter optimisation (single- and multi-parameter). Pure functional core — no plotting, no I/O. | `residual`, `residual_l2_squared`, `residual_discrete_sum_of_squares`, `hbar_curve_at`, `optimal_hbar`, `optimal_parameters` |
| [`ham.pade`](pade.md) | Homotopy-Padé acceleration of the partial sum in \(q\), evaluated at \(q = 1\). | `homotopy_pade` |

### Algebraic contracts

| Module | Responsibility | Headline surface |
| --- | --- | --- |
| [`ham.contracts`](contracts.md) | Opt-in algebraic-contract checkers users can call at problem-construction time. Linearity of \(L\) and consistency of \(u_0\) with the original boundary conditions are both contract-only on the data types; this module exposes them as runtime assertions raising `ValueError` subclasses. | `verify_linearity`, `verify_initial_guess`, `LinearityViolation`, `InitialGuessViolation` |

## What you won't find here

- **Plotting.** The library produces sympy expressions; render externally.
- **Numerical drivers.** `optimal_hbar` evaluates the supplied
  `norm_fn` at concrete `hbar` values via Python `float`; if the
  problem demands `scipy.optimize`-grade tooling, compose externally.
- **Validity gates.** Convergence claims belong to the application —
  each example provides its own `is_convergent` predicate using the
  primitives above; the library does not ship a generic one. See
  [Convergence](../concepts/convergence.md) for the rationale.

## Conventions across every module

- Every sympy expression is a `sp.Expr`. Functions accept
  `Symbol`, `Integer`, `Rational`, etc., as long as they participate
  in the algebra correctly.
- The convergence-control parameter `hbar` is **kept symbolic**
  throughout `solve`. Substitute it in user code via
  `solution.evaluate_at_hbar(value)` or by passing
  `hbar_value=value` to diagnostics / Padé functions.
- Frozen `@dataclass(frozen=True)` is used for every algebraic
  type (`BoundaryCondition`, `LinearOperator`, `NonlinearOperator`,
  `HamProblem`, `HamSolution`). Construction is the only mutation;
  threading state through transformations means building a new
  value.
- All public functions are type-annotated and pass `mypy --strict`.
