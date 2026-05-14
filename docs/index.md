# `ham`

A symbolic-math library for solving nonlinear differential and algebraic
equations via Liao's **Homotopy Analysis Method** (HAM). Pure
[sympy](https://www.sympy.org/) — operates on formal power series in the
embedding parameter \(q\) with symbolic coefficients in the independent
variable, then accelerates the resulting series either by direct
summation or via Padé.

## What you get

- A tiny algebraic core in eight modules.
- Five end-to-end worked examples: closed-form benchmarks
  (`tanh(t)`, the sigmoid `1/(1 + e^{-t})`), the integro-differential
  Volterra single-species population model, and the Blasius
  boundary-layer BVP in two basis choices (polynomial, exponential).
- A convergence-diagnostics toolkit: residual, L² / discrete norms,
  Liao's ℏ-curve, grid-search optimal-ℏ.
- Homotopy-Padé acceleration in \(q\) that recovers `1/(1-x)` exactly
  from just two HAM coefficients for the geometric-series problem.

## A minute-long sample

The logistic equation \(u'(t) = u(t)(1 - u(t))\) with \(u(0) = 1/2\),
exact \(u(t) = 1/(1 + e^{-t})\):

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
        expr=u(t).diff(t) - u(t) + u(t) ** 2,
        dependent=u,
        indep=t,
    ),
    H=sp.Integer(1),
    hbar=hbar,
    u0=sp.Rational(1, 2),
)

solution = solve(problem, order=5)
print(solution.evaluate_at_hbar(sp.Integer(-1)))
# t**5/480 - t**3/48 + t/4 + 1/2   ← the sigmoid Taylor expansion
```

## Where to go next

- **[Tutorial](tutorial.md)** — a guided walk through the full solve →
  diagnose → accelerate workflow on the logistic problem.
- **[Concepts](concepts/index.md)** — the HAM math from scratch:
  homotopy equation, deformation chain, convergence, Padé.
- **[Worked examples](examples/index.md)** — narrative walk-throughs of
  the five examples that ship with the library, from the closed-form
  tanh/sigmoid benchmarks to the Blasius BVP in two different bases.
- **[API reference](api/index.md)** — auto-extracted from module
  docstrings.
- **[Design notes](design/index.md)** — the architectural decisions and
  tenets behind the codebase.

## Installation

Python ≥ 3.12, managed with [Poetry](https://python-poetry.org/):

```sh
poetry install            # runtime + dev dependencies
poetry install --with docs  # adds mkdocs/material/mkdocstrings
poetry run pytest         # 252 tests, ~2 min
poetry run mkdocs serve   # local docs preview at http://127.0.0.1:8000
```

## References

- Liao, S.-J. (2003). *Beyond Perturbation: Introduction to the
  Homotopy Analysis Method*. Chapman & Hall/CRC.
- Liao, S.-J. (2009). Notes on the homotopy analysis method: Some
  definitions and theorems. *Comm. Nonlinear Sci. Numer. Simul.* 14:
  983–997.
