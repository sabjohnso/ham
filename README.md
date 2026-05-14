# `ham`: the Homotopy Analysis Method in Python

A symbolic-math library for nonlinear differential and algebraic equations,
built on Liao's *Homotopy Analysis Method* (HAM). Pure `sympy` — operates on
formal power series in the embedding parameter `q` with symbolic
coefficients in the independent variable, then accelerates the resulting
series at `q = 1` either directly or via Padé.

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

References: Liao, *Beyond Perturbation* (Chapman & Hall/CRC 2003); Liao,
*Notes on the homotopy analysis method: Some definitions and theorems*,
Comm. Nonlinear Sci. Numer. Simul. 14 (2009) 983-997.

## Installation

Python ≥ 3.12, managed with Poetry.

```sh
poetry install                                  # runtime + dev dependencies
poetry run pytest                               # 166 tests, ~50s
```

## Quick example: the logistic equation

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

## Library layout

```
ham/
  series.py        QSeries — truncated formal power series in q
  operator.py      LinearOperator with first-class BCs and pluggable invert
  nonlinear.py     NonlinearOperator — sympy-tree compiler to QSeries arithmetic
  deformation.py   HamProblem + r_m / rhs_m for the m-th deformation equation
  solver.py        solve(problem, order) → HamSolution + per-step solve_step
  diagnostics.py   residual, L² / discrete norms, ℏ-curve, optimal-ℏ
  pade.py          homotopy-Padé acceleration

examples/
  quadratic_drag.py    v' = 1 - v², v(0) = 0  (exact tanh)
  logistic.py          u' = u(1 - u), u(0) = 1/2  (exact sigmoid)
```

Each module's docstring is the design document. Each example is both an
importable module (factory functions) and a runnable script.

## Diagnostics surface

Once you have a `HamSolution`, the rest of the library is composition:

| call | result |
| --- | --- |
| `sol.partial_sum()` | `Σ u_k(x)`, ℏ symbolic |
| `sol.evaluate_at_hbar(value)` | partial sum with ℏ substituted |
| `residual(sol, hbar)` | `N` applied to the partial sum |
| `residual_l2_squared(sol, hbar, interval)` | `∫ residual² dx` |
| `residual_discrete_sum_of_squares(sol, hbar, samples)` | `Σ residual(x_i)²` |
| `hbar_curve_at(sol, x_star)` | partial sum at `x*` as a polynomial in ℏ |
| `optimal_hbar(sol, grid, norm_fn)` | grid value minimising `norm_fn` |
| `homotopy_pade(sol, L, M, hbar)` | `[L/M]` Padé at `q = 1` |

ℏ is kept symbolic through every stage; downstream code substitutes when
ready. The same `solve()` output can be evaluated at any number of ℏ
values without re-running the loop.

## Development

```sh
poetry run pytest                               # full test suite
poetry run pytest tests/test_x.py::test_name    # single test
poetry run ruff check ham/ tests/ examples/     # lint
poetry run ruff format                          # format
poetry run mypy                                 # type-check (strict)
poetry run pre-commit run --all-files           # pre-commit hooks
```

The library follows Algebra-Driven Design: each module exposes algebraic
laws (linearity of `L`, Cauchy structure of `N`, causality in `q`, etc.)
and Hypothesis property tests assert those laws directly. Worked examples
under `tests/examples/` pin end-to-end output against closed-form Taylor
expansions.

## Continuous integration

`.github/workflows/test.yml` runs on push to `main` and on pull requests.
The job installs Poetry via `pipx`, caches the Poetry virtualenv, then
runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`
against Python 3.12 on `ubuntu-latest`. A green CI run is the baseline
expectation before merging any PR.
