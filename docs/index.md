# `ham`

A library for solving nonlinear differential and algebraic equations
via Liao's **Homotopy Analysis Method** (HAM). The algebraic core
operates on formal power series in the embedding parameter \(q\) with
substrate-generic coefficients: classical symbolic HAM ([sympy][sympy])
with sympy expressions in the independent variable, or the **Spectral
HAM** ([SHAM][sham-paper]) variant with numpy arrays of nodal values on
a Chebyshev grid. Same solver loop, same diagnostics, same algebraic
laws — the substrate is the only difference, threaded through a
`Backend[C]` Protocol.

[sympy]: https://www.sympy.org/
[sham-paper]: https://doi.org/10.1016/j.cnsns.2009.09.015

## What you get

- A small algebraic core in eleven modules, generic over the coefficient
  substrate. Adding a new substrate (Fourier on the circle,
  rational-Chebyshev on \([0, \infty)\), ...) means writing a `Backend[C]`
  and the matching `Grid`; the solver loop, deformation builder,
  diagnostics, and contract checkers all come for free.
- Two substrates ready to use:
    - **Symbolic.** `sympy.Expr` coefficients; ℏ stays symbolic
      through every stage; the partial sum is a polynomial in ℏ that
      downstream code substitutes when ready.
    - **Spectral.** `numpy.ndarray` coefficients on a Chebyshev-Gauss-
      Lobatto grid, in two scalar modes — `float` for classical SHAM
      (fast linear solves, ℏ-curves via external sweep) and `sympy`
      for the symbolic-ℏ variant that keeps ℏ inside every grid entry
      (slower; ℏ-curves work directly).
- Five end-to-end worked examples on the symbolic substrate:
  closed-form benchmarks (`tanh(t)`, the sigmoid `1/(1 + e^{-t})`),
  the integro-differential Volterra single-species population model,
  and the Blasius boundary-layer BVP in two basis choices (polynomial,
  exponential).
- A convergence-diagnostics toolkit shared across substrates:
  residual, L² / discrete norms, Liao's ℏ-curve, grid-search
  optimal-ℏ, and the spectral `hbar_curve_at_sweep` for the
  float-substrate ℏ-curve.
- Homotopy-Padé acceleration in \(q\) (sympy substrate) that recovers
  `1/(1-x)` exactly from just two HAM coefficients for the
  geometric-series problem.

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

The same problem with the spectral substrate (float scalar, ℏ
pre-substituted to -1) — note that only the construction of \(L\) and
the `backend=` kwarg on `solve` change:

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
    N=NonlinearOperator(
        expr=u(t).diff(t) - u(t) + u(t) ** 2,
        dependent=u, indep=t,
    ),
    H=sp.Integer(1),
    hbar=sp.Float(-1.0),
    u0=sp.Rational(1, 2),
)

solution = solve(problem, order=5, backend=backend)
print(solution.partial_sum())     # numpy array of length 17 — the sigmoid Taylor
                                  # truncation evaluated at the Chebyshev nodes
```

## Where to go next

- **[Tutorial](tutorial.md)** — a guided walk through the full solve →
  diagnose → accelerate workflow on the logistic problem (symbolic
  substrate); the closing section runs the same problem under the
  spectral backend.
- **[Concepts](concepts/index.md)** — the HAM math from scratch:
  homotopy equation, deformation chain, convergence, Padé.
- **[Worked examples](examples/index.md)** — narrative walk-throughs of
  the five examples that ship with the library, from the closed-form
  tanh/sigmoid benchmarks to the Blasius BVP in two different bases.
- **[API reference](api/index.md)** — auto-extracted from module
  docstrings, organised by substrate / algebraic-core / spectral
  layer.
- **[Design notes](design/index.md)** — the architectural decisions
  and tenets behind the codebase, including the S0-S9 substrate-
  parametrisation arc.

## Installation

Python ≥ 3.12, managed with [Poetry](https://python-poetry.org/):

```sh
poetry install              # runtime (sympy + numpy) + dev dependencies
poetry install --with docs  # adds mkdocs/material/mkdocstrings
poetry run pytest           # full test suite, ~3 min
poetry run mkdocs serve     # local docs preview at http://127.0.0.1:8000
```

## References

- Liao, S.-J. (2003). *Beyond Perturbation: Introduction to the
  Homotopy Analysis Method*. Chapman & Hall/CRC.
- Liao, S.-J. (2009). Notes on the homotopy analysis method: Some
  definitions and theorems. *Comm. Nonlinear Sci. Numer. Simul.* 14:
  983–997.
- Motsa, S. S., Sibanda, P., & Shateyi, S. (2010). A new spectral-
  homotopy analysis method for solving a nonlinear second order BVP.
  *Comm. Nonlinear Sci. Numer. Simul.* 15: 2293-2302. — the original
  SHAM paper.
- Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM. — source
  of the differentiation matrix (`cheb.m`) and Clenshaw-Curtis weights
  (`clencurt.m`) ported into `ham.grids`.
