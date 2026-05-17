# `ham`: the Homotopy Analysis Method in Python

A library for nonlinear differential and algebraic equations, built on
Liao's *Homotopy Analysis Method* (HAM). The algebraic core operates
on formal power series in the embedding parameter `q` with
substrate-generic coefficients: classical symbolic HAM (`sympy.Expr`
in the independent variable) or the **Spectral HAM** (SHAM) variant
(`numpy.ndarray` of nodal values on a Chebyshev grid). Same solver
loop, same diagnostics, same algebraic laws across both substrates.

## What HAM is, briefly

Given a nonlinear equation `N[u(x)] = 0` with boundary conditions, HAM
constructs a homotopy

```
(1 - q) L[φ(x; q) - u_0(x)] = q · ℏ · H(x) · N[φ(x; q)]
```

between an initial guess `u_0` (which need only satisfy the BCs, not the
ODE) and the true solution `u`. Expanding `φ(x; q) = Σ u_k(x) q^k` and
matching powers of `q` yields a sequence of linear ODEs for the
`u_k`, all involving the same auxiliary linear operator `L`. The
convergence-control parameter `ℏ` is a free knob: chosen well, it makes
the partial sum `Σ u_k(x)` converge to the true solution faster and over
a wider domain than classical perturbation methods.

The **Spectral** variant (SHAM) keeps the same homotopy and the same
deformation chain but discretises every `u_k` on a Chebyshev-Gauss-
Lobatto grid. `L.invert` becomes a dense linear solve with BCs imposed
by row replacement; `N[φ]` is evaluated via the differentiation matrix
and element-wise products. SHAM trades the symbolic ℏ-curve for fast
linear algebra and stronger numerical convergence behaviour at
moderate truncation orders.

References:

- Liao, *Beyond Perturbation* (Chapman & Hall/CRC 2003) — the
  symbolic HAM reference.
- Liao, *Notes on the homotopy analysis method: Some definitions and
  theorems*, Comm. Nonlinear Sci. Numer. Simul. 14 (2009) 983-997.
- Motsa, Sibanda, Shateyi, *A new spectral-homotopy analysis method
  for solving a nonlinear second order BVP*, Comm. Nonlinear Sci.
  Numer. Simul. 15 (2010) 2293-2302 — the original SHAM paper.

## Installation

Python ≥ 3.12, managed with Poetry.

```sh
poetry install                                  # runtime (sympy + numpy) + dev deps
poetry run pytest                               # 338 tests, ~3 min
```

## Quick example — sympy substrate (the logistic equation)

`u'(t) = u(t)·(1 - u(t))` with `u(0) = 1/2`, exact `u(t) = 1/(1 + e^{-t})`.

```python
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import solve

t = sp.Symbol("t")
u = sp.Function("u")
hbar = sp.Symbol("hbar")

problem = HamProblem(
    L=LinearOperator(
        var=t,
        action=lambda e: sp.diff(e, t),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    ),
    N=NonlinearOperator(
        expr=u(t).diff(t) - u(t) + u(t)**2,
        dependent=u,
        indep=t,
    ),
    H=sp.Integer(1),
    hbar=hbar,
    u0=sp.Rational(1, 2),
)

solution = solve(problem, order=5)
print(solution.evaluate_at_hbar(sp.Integer(-1)))
# t**5/480 - t**3/48 + t/4 + 1/2   (the sigmoid Taylor expansion)
```

## Quick example — spectral substrate (the same problem on a grid)

```python
from ham.grids import ChebGLGrid
from ham.spectral import SpectralBackend, spectral_linear_operator

grid = ChebGLGrid(N=16, domain=(0.0, 1.0))
backend = SpectralBackend(grid, indep=t, scalar="float")

problem = HamProblem(
    L=spectral_linear_operator(
        u(t).diff(t), dependent=u, indep=t, grid=grid, scalar="float",
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    ),
    N=NonlinearOperator(expr=u(t).diff(t) - u(t) + u(t)**2, dependent=u, indep=t),
    H=sp.Integer(1),
    hbar=sp.Float(-1.0),    # pre-substituted on the float spectral backend
    u0=sp.Rational(1, 2),
)

solution = solve(problem, order=5, backend=backend)
print(solution.partial_sum())
# numpy array of length 17 — the sigmoid Taylor truncation at the grid nodes
```

The two paths share the `HamProblem` / `NonlinearOperator` / `solve`
shape exactly; only `L`'s construction and the `backend=` kwarg
change.

## Homotopy-Padé acceleration: exact answers from two coefficients

For problems where the formal series in `q` has radius of convergence less
than one, the bare partial sum at `q = 1` diverges. The `[L/M]` Padé
approximant in `q` often analytically continues past that radius. For
`u' = u²`, `u(0) = 1` (exact `1/(1 - x)`):

```python
from ham.pade import homotopy_pade
# ... same problem setup with N = u' - u**2, u_0 = 1 ...
sol = solve(problem, order=1)               # just u_0 and u_1
print(sol.evaluate_at_hbar(sp.Integer(-1)))
# x + 1   (truncated, diverges at x = 1)
print(homotopy_pade(sol, 0, 1, sp.Integer(-1)))
# 1/(1 - x)   (the exact closed form, from two coefficients)
```

Padé currently runs on the sympy substrate only; the block-structured
spectral analogue is a tracked follow-up.

## Library layout

```
ham/
  backend.py       Backend[C] Protocol — six ops bridging the algebraic core
                   to any coefficient substrate. SympyBackend included.
  series.py        Series[C] — substrate-generic truncated power series in q.
                   QSeries is the sympy-baked back-compat shim.
  operator.py      LinearOperator[C] with first-class BCs and pluggable invert.
                   sympy_dsolve_inverter is the sympy-substrate factory.
  nonlinear.py     NonlinearOperator — sympy-tree compiler whose tree-walker
                   dispatches through phi.backend.
  deformation.py   HamProblem[C] + r_m / rhs_m for the m-th deformation equation.
  solver.py        solve(problem, order, backend=...) → HamSolution[C].
  diagnostics.py   residual, L² / discrete norms, ℏ-curve, optimal-ℏ.
                   Substrate-aware dispatch on grid= kwarg.
  pade.py          Homotopy-Padé acceleration (sympy substrate only).
  contracts.py     Opt-in algebraic-contract checkers (verify_linearity with
                   injectable equal=; verify_initial_guess).
  grids.py         Grid Protocol + ChebGLGrid (Trefethen cheb.m + clencurt.m).
  spectral.py      SpectralBackend(grid, indep, scalar) + spectral_linear_operator
                   + spectral_inverter (SHAM substrate).

examples/
  quadratic_drag.py        v' = 1 - v², v(0) = 0                   (exact tanh)
  logistic.py              u' = u(1 - u), u(0) = 1/2               (exact sigmoid)
  volterra.py              u' = κ·u·(1 - u - ∫u dτ), u(0) = α      (Taylor recurrence)
  blasius.py               f''' + ½ f f'' = 0, polynomial basis    (Howarth f''(0) ≈ 0.4696)
  blasius_exponential.py   same Blasius BVP, exponential basis     (joint (ℏ, α) optimum)
  blasius_inverter.py      closed-form basis-aware L⁻¹ for blasius_exponential
```

Each module's docstring is the design document. Each example is both an
importable module (factory functions) and a runnable script.

## Diagnostics surface

Once you have a `HamSolution`, the rest of the library is composition:

| call | sympy substrate | spectral substrate |
| --- | --- | --- |
| `sol.partial_sum()` | `Σ u_k(x)`, ℏ symbolic | grid vector, length N+1 |
| `sol.evaluate_at_hbar(value)` | partial sum with ℏ substituted | sympy substrate only |
| `residual(sol, hbar)` | `N` applied to the partial sum | grid vector via apply_series order-0 |
| `residual_l2_squared(sol, hbar, interval=..., grid=...)` | `∫ residual² dx` | `Σ wᵢ · residual²` |
| `residual_discrete_sum_of_squares(sol, hbar, samples)` | `Σ residual(x_i)²` | sympy substrate only |
| `hbar_curve_at(sol, x_star, grid=...)` | polynomial in ℏ | nearest-node entry |
| `hbar_curve_at_sweep(factory, x_star, hbar_grid, order, grid, backend)` | — | sweep solver per ℏ |
| `optimal_hbar(sol, grid, norm_fn)` | grid value minimising `norm_fn` | use sweep + `min` |
| `homotopy_pade(sol, L, M, hbar)` | `[L/M]` Padé at `q = 1` | not yet — sympy only |

On the sympy and sympy-scalar spectral substrates ℏ is kept symbolic
through every stage; downstream code substitutes when ready. On the
float spectral substrate ℏ is a numeric parameter set at problem
construction.

## Development

```sh
poetry run pytest                               # full test suite (338 tests)
poetry run pytest tests/test_x.py::test_name    # single test
poetry run ruff check ham/ tests/ examples/     # lint
poetry run ruff format                          # format
poetry run mypy                                 # type-check (strict)
poetry run pre-commit run --all-files           # pre-commit hooks
```

The library follows Algebra-Driven Design: each module exposes algebraic
laws (linearity of `L`, Cauchy structure of `N`, causality in `q`, the
diff/integrate inverse on substrate-resolved polynomials, etc.) and
Hypothesis property tests assert those laws directly across substrates.
Worked examples under `tests/examples/` pin end-to-end output against
closed-form Taylor expansions (tanh, sigmoid), Taylor recurrences
(Volterra), and published numerical references (Howarth's Blasius
`f''(0) ≈ 0.4696`); `tests/test_spectral_solve.py` and
`tests/test_diagnostics_spectral.py` pin the SHAM substrate end to end
against the same problems.

## Documentation

The narrative docs (tutorial, concept pages, worked-example
walk-throughs, design notes, auto-extracted API reference) are built
with [`mkdocs`](https://www.mkdocs.org/) +
[`mkdocs-material`](https://squidfunk.github.io/mkdocs-material/) +
[`mkdocstrings`](https://mkdocstrings.github.io/). The doc tooling is
in the `docs` Poetry group, separate from the runtime install so
end-users don't pull it by default.

```sh
poetry install --with docs                      # one-time: install doc tooling
poetry run mkdocs serve                         # live-reload at http://127.0.0.1:8000
poetry run mkdocs build --strict                # static build to site/; fail on broken links
```

`mkdocs serve` watches `docs/`, `mkdocs.yml`, and the `ham/` package
(so `mkdocstrings` re-extracts docstrings on edit) and rebuilds on
every save. Use `--strict` on CI-shaped builds to catch broken
cross-references before they ship.

## Continuous integration

`.github/workflows/test.yml` runs on push to `main` and on pull requests.
The job installs Poetry via `pipx`, caches the Poetry virtualenv, then
runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`
against Python 3.12 on `ubuntu-latest`. A green CI run is the baseline
expectation before merging any PR.
