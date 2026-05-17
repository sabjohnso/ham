# Logistic equation

## The problem

The unit-rate logistic equation models population growth saturated at
carrying capacity 1:

\[
u'(t) = u(t)\,(1 - u(t)), \qquad u(0) = \tfrac{1}{2}.
\]

The closed-form solution is the standard sigmoid
\(u(t) = 1/(1 + e^{-t})\), with Taylor expansion about \(t = 0\)

\[
u(t) = \tfrac{1}{2} + \tfrac{t}{4} - \tfrac{t^3}{48} + \tfrac{t^5}{480}
       - \tfrac{17 t^7}{80640} + O(t^9).
\]

The sigmoid is odd about its centre value \(\tfrac{1}{2}\) — only odd
powers of \(t\) appear as corrections to the constant term.

## What makes this example different

Every prior cross-check (`exp(x)`, `1/(1-x)`, `tanh(t)`) starts from
\(u_0 = 0\) or \(u_0 = 1\) — initial guesses that are either trivially
zero or a multiplicative identity. The logistic problem demands
\(u(0) = 1/2\), so \(u_0 = 1/2\) is the simplest valid initial guess —
the first **non-trivial constant** initial guess that the library has
to handle. That exercises a code path none of the prior examples
touched.

## Liao's three rules in this example

1. **Solution expression.** The sigmoid is analytic at \(t = 0\) with
   only odd-power corrections about the centre value \(1/2\). A
   polynomial base via \(L = d/dt\) with \(u_k(0) = 0\) for \(k \ge 1\)
   reflects that structure.
2. **Coefficient ergodicity** (base coefficients in \(u = \sum_n c_n t^n\),
   *not* the \(q\)-coefficients \(u_k\); see
   [Rule 2](../concepts/convergence.md)). The deformation BCs are
   homogeneous, the polynomial degree grows with \(k\), and \(H = 1\)
   — no terms are structurally blocked.
3. **Solution existence.** \(u_0 = 1/2\) satisfies the original BC
   \(u(0) = 1/2\) exactly. The library's separation of "original BC
   satisfied by \(u_0\)" from "homogeneous deformation BCs on \(L\)"
   makes this a one-line setup.

## HAM setup

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
    u0=sp.Rational(1, 2),     # ← non-trivial initial guess
)
```

\(N[u] = u' - u + u^2\) so that the exact sigmoid gives
\(N[u_{\text{exact}}] = u'(1 - u) - u + u^2 = u - u^2 - u + u^2 = 0\).

## Solve and verify

```python
solution = solve(problem, order=7)
print(solution.evaluate_at_hbar(sp.Integer(-1)))
# -17*t**7/80640 + t**5/480 - t**3/48 + t/4 + 1/2
```

That is the sigmoid Taylor expansion to order 7, term for term. The
regression test in `tests/examples/test_logistic.py` pins this match
at orders 5 and 7; another test asserts no even-power corrections
appear above \(t^0\) in the partial sum at \(\hbar = -1\) — the
even-power constraint that sigmoidal odd-about-centre structure
implies.

## Diagnostics

```python
from examples.logistic import analyze
analyze(solution)
```

```
residual_at_neg_one:    289*t**14/6502809600 - 17*t**12/19353600
                        + 127*t**10/9676800 - 31*t**8/161280
hbar_curve_at_t_eq_1:   2537*hbar**7/80640 - 35*hbar**6/96 - 427*hbar**5/160
                        - 105*hbar**4/16 - 385*hbar**3/48 - 21*hbar**2/4
                        - 7*hbar/4 + 1/2
optimal_hbar:           -1
l2_norm_at_optimal:     ≈ 1.93e-09
convergent_at_neg_one:  True
```

### Residual orders of magnitude

The L² norm² of the residual on \([0, 1]\) for this problem is **two
to three orders of magnitude smaller** than for the quadratic-drag
example at the same working order. The sigmoid's Taylor expansion has
fast-decaying coefficients (\(1/4, -1/48, 1/480, -17/80640, \ldots\)),
so each additional HAM order buys substantial residual reduction.

This is why the example's default validity-gate threshold is *tighter*
than the quadratic-drag example's: \(1/100\) on the L² norm here, vs
\(1/10\) for quadratic drag.

### Optimal ℏ *is* \(-1\)

```python
optimal_hbar(sol, [-3/2, -1, -1/2, 0]) == -1
```

Unlike the quadratic-drag example, the L² advantage of plain Taylor
truncation is clean and unambiguous on this problem. The L² norm²
values across the grid:

| \(\hbar\) | L² norm² |
| --- | --- |
| \(-3/2\) | \(\approx 3.08 \times 10^{-4}\) |
| \(-1\) | \(\approx 1.93 \times 10^{-9}\) |
| \(-1/2\) | \(\approx 8.78 \times 10^{-6}\) |
| \(0\) | \(6.25 \times 10^{-2}\) |

\(\hbar = -1\) is *five orders of magnitude better* than the runner-up.

The contrast with quadratic-drag is worth noting: both problems are
polynomial nonlinearity, both use the same \(L\), both Taylor-converge
at \(\hbar = -1\). But for tanh the truncation residual on \([0, 1]\)
is intrinsically larger than for the sigmoid (the tanh Taylor
coefficients decay slower), so adaptive ℏ has more to win. For the
sigmoid, plain Taylor is already close enough that no other grid
choice can compete.

### Validity gate

```python
from examples.logistic import is_convergent
is_convergent(solution, sp.Integer(-1))   # True
is_convergent(solution, sp.Integer(0))    # False
```

Threshold \(1/100\) on the L² norm over \([0, 1]\). At \(\hbar = -1\)
the norm is \(\sqrt{1.93 \times 10^{-9}} \approx 4.4 \times 10^{-5}\)
— easily below threshold. At \(\hbar = 0\) the partial sum is
\(u_0 = 1/2\), the residual is \(N[1/2] = -1/4\) (constant), and the
norm is \(1/4 > 1/100\).

## On the spectral substrate (SHAM)

The same logistic HamProblem on a Chebyshev-Gauss-Lobatto grid, ℏ
pre-substituted to \(-1\):

```python
from examples.logistic import solve_to_spectral
from ham.grids import ChebGLGrid
import numpy as np

grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
sol = solve_to_spectral(7, grid=grid)

exact = 1.0 / (1.0 + np.exp(-grid.nodes))
print(np.max(np.abs(sol.partial_sum() - exact)))
# ≈ 4e-4 — well below the next-Taylor-term bound (~6.6e-3 at t=1)
```

Note the **non-zero initial guess** \(u_0 = 1/2\): the lift to the
grid is a constant vector of `0.5`s, not a varying function. The
homogeneous deformation BC \(u_k(0) = 0\) is imposed by
`spectral_inverter` per step, so the partial sum stays consistent
with the original `u(0) = 1/2` boundary condition.

The ℏ-curve on the spectral substrate has two flavours:

```python
from ham.diagnostics import hbar_curve_at, hbar_curve_at_sweep
from examples.logistic import HBAR, build_spectral_problem

# sympy-scalar mode: ℏ stays symbolic inside every grid entry,
# so the partial sum at any grid node is a polynomial in ℏ.
sol_sym = solve_to_spectral(3, grid=grid, scalar="sympy", hbar_value=HBAR)
curve = hbar_curve_at(sol_sym, sp.Float(0.5), grid=grid)
# `curve` is a sympy expression in ℏ; plot externally or pass to
# optimal_hbar via a closure.

# float-scalar mode: ℏ is numeric per problem, so the ℏ-curve at
# a point is built by sweeping the solver across ℏ values.
hbar_grid = [sp.Float(h) for h in (-1.5, -1.0, -0.5, 0.0)]
pairs = hbar_curve_at_sweep(
    lambda hbar: build_spectral_problem(grid, scalar="float", hbar_value=hbar),
    x_star=sp.Float(0.5), hbar_grid=hbar_grid,
    order=5, grid=grid,
    backend=...,    # SpectralBackend(grid, indep=T, scalar="float")
)
# pairs is [(ℏ_i, partial_sum_at_0.5_under_ℏ_i)] — the discrete
# version of the symbolic ℏ-curve, expensive (one full solve per
# point) but available at machine precision.
```

The dual-scalar trade-off is documented in the
[substrates concept page](../concepts/substrates.md): pick sympy
scalar when the ℏ-curve interpretation is worth the
`sp.Matrix.LUsolve` cost, float scalar for fast convergence at
moderate orders.

## Running the example as a script

```sh
poetry run python examples/logistic.py
```

prints the same diagnostic bundle as `analyze()` plus the validity
gate result, and ends with the spectral substrate's L∞ error vs the
sigmoid on the grid.
