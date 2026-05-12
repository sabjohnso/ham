# Tutorial

This walkthrough takes one nonlinear ODE end to end: assemble a
`HamProblem`, solve it to a working order, inspect the resulting series,
verify it against a closed-form reference, run convergence diagnostics,
search for an optimal convergence-control parameter, and finally apply
the homotopy-Padé acceleration on a related problem where it does
something striking.

## The problem

The unit-rate logistic equation

\[
u'(t) = u(t)\,(1 - u(t)), \qquad u(0) = \tfrac{1}{2},
\]

has closed-form solution \(u(t) = 1/(1 + e^{-t})\) (the standard
sigmoid). Its Taylor expansion around \(t = 0\) reads

\[
u(t) = \tfrac{1}{2} + \tfrac{t}{4} - \tfrac{t^3}{48} + \tfrac{t^5}{480}
       - \tfrac{17\,t^7}{80640} + O(t^9).
\]

The logistic problem is small enough to follow by hand and rich enough
to exercise every part of the HAM stack.

## Step 1 — Build the `HamProblem`

The library models the input data of HAM as a single frozen dataclass.
You supply five things:

- The auxiliary linear operator \(L\), with the boundary conditions the
  HAM deformations should respect.
- The nonlinear operator \(N\), written so that \(N[u_{\mathrm{exact}}] = 0\).
- The auxiliary function \(H(x)\), a sympy expression (not an operator).
- The convergence-control parameter \(\hbar\), kept symbolic so you can
  re-evaluate the partial sum at any \(\hbar\) without re-solving.
- The initial guess \(u_0(x)\), which only has to satisfy the original
  boundary condition.

```python
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator

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
```

Three things deserve a comment:

- `LinearOperator.action` is *any* callable `Expr → Expr` you claim is
  linear. The library does not enforce linearity; the test suite
  verifies it on the canonical `d/dt` case. The `bcs` field is a tuple
  of `BoundaryCondition(point, derivative_order, value=0)` items —
  here, the single homogeneous BC `u(0) = 0` that every deformation
  step \(u_k\) must satisfy.

- `NonlinearOperator.expr` is the sympy expression that defines
  \(N[u]\). The compiler in
  [`ham.nonlinear`](api/nonlinear.md) walks this tree and emits
  `QSeries` arithmetic against the homotopy series — polynomial in
  \(u\) with \(x\)-derivatives is supported, transcendentals raise
  with a clear message.

- `u0 = 1/2` satisfies the original BC `u(0) = 1/2` exactly. The
  deformation BCs are then the homogeneous `u_k(0) = 0` for \(k \ge 1\),
  which compose without conflict.

## Step 2 — Solve

```python
from ham.solver import solve

solution = solve(problem, order=5)
```

`solution` is a `HamSolution`: a frozen dataclass bundling the
originating problem and a `QSeries` whose `coeff(k)` is the HAM
coefficient \(u_k(t)\). The driver iterates `m = 1..order`, building
`phi` incrementally and calling `L.invert` once per step.

```python
for k in range(solution.order + 1):
    print(f"u_{k}(t) =", solution.phi.coeff(k))
```

```
u_0(t) = 1/2
u_1(t) = -hbar*t/4
u_2(t) = -hbar**2*t/4 - hbar*t/4
u_3(t) = hbar**3*t**3/48 - hbar**3*t/4 - hbar**2*t/2 - hbar*t/4
u_4(t) = hbar**4*t**3/16 - hbar**4*t/4 + hbar**3*t**3/16 - 3*hbar**3*t/4 - 3*hbar**2*t/4 - hbar*t/4
u_5(t) = -hbar**5*t**5/480 + hbar**5*t**3/8 - hbar**5*t/4 + hbar**4*t**3/4 - hbar**4*t + hbar**3*t**3/8 - 3*hbar**3*t/2 - hbar**2*t - hbar*t/4
```

Every \(u_k\) for \(k \ge 1\) carries `hbar` symbolically. The
convergence-control parameter is a free variable until you decide what
to do with it.

## Step 3 — Evaluate the partial sum

Two flavors:

```python
solution.partial_sum()            # ℏ kept symbolic
solution.evaluate_at_hbar(sp.Integer(-1))   # substituted
```

At \(\hbar = -1\) the HAM partial sum collapses to the truncated Taylor
expansion of the exact solution:

```python
>>> solution.evaluate_at_hbar(sp.Integer(-1))
t**5/480 - t**3/48 + t/4 + 1/2
```

That matches the sigmoid Taylor expansion shown above, term for term.
This is not a coincidence — for problems where the HAM iteration
recovers the Taylor series exactly, \(\hbar = -1\) is the natural
choice.

## Step 4 — Compute the residual

The library's convergence diagnostics live in
[`ham.diagnostics`](api/diagnostics.md). The fundamental primitive is
`residual(solution, hbar_value)`, which applies \(N\) to the partial
sum:

```python
from ham.diagnostics import residual

print(residual(solution, sp.Integer(-1)))
# t**10/230400 - t**8/11520 + 17*t**6/11520
```

If the partial sum were the exact solution, the residual would vanish.
What you see is the truncation error fed through \(N\), starting at
\(O(t^6)\) for this order \(M = 5\) run.

## Step 5 — Norm the residual

Two flavors of L² norm are available; pick by problem character:

```python
from ham.diagnostics import (
    residual_l2_squared,
    residual_discrete_sum_of_squares,
)

interval = (sp.Integer(0), sp.Integer(1))
norm_l2 = residual_l2_squared(solution, sp.Integer(-1), interval)
print(f"L^2 norm squared on [0, 1] at hbar=-1: {float(norm_l2):.3e}")
# ≈ 1.516e-07

samples = [sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(3, 4), sp.Integer(1)]
norm_d = residual_discrete_sum_of_squares(solution, sp.Integer(-1), samples)
print(f"discrete sum of squares at {samples}: {float(norm_d):.3e}")
```

`residual_l2_squared` uses `sp.integrate` and is exact for polynomial
residuals; `residual_discrete_sum_of_squares` evaluates and sums at
sample points and is what you reach for when the residual is
transcendental or the domain is unbounded.

## Step 6 — Sample the ℏ-curve

At a fixed point \(t = t^\star\) the partial sum is a polynomial in
\(\hbar\) — Liao's *ℏ-curve*. A horizontal plateau in that curve
indicates a candidate convergence region for \(\hbar\):

```python
from ham.diagnostics import hbar_curve_at

curve = hbar_curve_at(solution, sp.Integer(1))
print(curve)
# -61*hbar**5/480 - 15*hbar**4/16 - 55*hbar**3/24 - 5*hbar**2/2 - 5*hbar/4 + 1/2
```

That polynomial *is* the curve. Render it externally if you want a
picture; the library stays a functional core.

## Step 7 — Grid-search the optimal ℏ

```python
from functools import partial
from ham.diagnostics import optimal_hbar

def norm(sol, h):
    return residual_l2_squared(sol, h, interval)

grid = [sp.Rational(-3, 2), sp.Integer(-1), sp.Rational(-1, 2), sp.Integer(0)]
best = optimal_hbar(solution, grid, norm)
print(f"optimal hbar over {grid}: {best}")
# -1
```

For the logistic problem at \(M = 5\), \(\hbar = -1\) is the winner.
The norm at \(\hbar = -1\) is about \(1.5 \times 10^{-7}\); at
\(\hbar = 0\) the partial sum collapses to \(u_0 = 1/2\) and the norm
jumps to about \(6.25 \times 10^{-2}\) — six orders of magnitude
worse. `optimal_hbar` picks the grid value minimising the
caller-supplied norm.

!!! note "When does the grid not pick \(-1\)?"
    On the quadratic-drag problem covered in the
    [examples gallery](examples/quadratic-drag.md), HAM at
    \(\hbar = -1/2\) gives a *smaller* L² residual than at
    \(\hbar = -1\) at order 7. That is the *adaptive-ℏ advantage* of
    HAM: tuning \(\hbar\) can distribute truncation error across the
    interval better than plain Taylor truncation. The library does not
    assume \(-1\) is special.

## Step 8 — A validity gate

Liao's convergence theorem (see [Concepts: Convergence](concepts/convergence.md))
only guarantees that the HAM partial sum equals the exact solution
*given* convergence at \(q = 1\). A *validity gate* is a per-problem
predicate that composes residual norm and threshold into a yes/no
convergence claim. Each worked example provides one:

```python
from examples.logistic import is_convergent

is_convergent(solution, sp.Integer(-1))  # True
is_convergent(solution, sp.Integer(0))   # False
```

The library deliberately keeps gates application-specific — a baked-in
threshold would risk false comfort.

## Step 9 — Homotopy-Padé acceleration

For problems where the formal series in \(q\) converges slowly or
diverges at \(q = 1\), the bare partial sum is the wrong tool. The
*homotopy-Padé* approximant — Padé in \(q\), evaluated at \(q = 1\) —
analytically continues the series past its radius of convergence.

The most striking demonstration is the geometric problem
\(u' = u^2, u(0) = 1\), exact \(1/(1-x)\):

```python
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.pade import homotopy_pade
from ham.solver import solve

x = sp.Symbol("x")
u = sp.Function("u")
hbar = sp.Symbol("hbar")

geometric = HamProblem(
    L=LinearOperator(
        var=x,
        action=lambda e: sp.diff(e, x),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    ),
    N=NonlinearOperator(expr=u(x).diff(x) - u(x) ** 2, dependent=u, indep=x),
    H=sp.Integer(1),
    hbar=hbar,
    u0=sp.Integer(1),
)

sol = solve(geometric, order=1)
print("partial sum at hbar=-1:", sol.evaluate_at_hbar(sp.Integer(-1)))
# x + 1   ← truncated, diverges at x = 1

print("[0/1] homotopy-Padé:   ", homotopy_pade(sol, 0, 1, sp.Integer(-1)))
# 1/(1 - x)   ← exact closed form from two coefficients
```

Two HAM coefficients, fed through Padé in \(q\), recover the exact
closed-form solution. That kind of result is the reason
[Liao §2.3.7](concepts/pade-acceleration.md) treats homotopy-Padé as
an indispensable companion to the bare partial sum.

## Where to go next

- [Concepts](concepts/index.md): how the homotopy equation works, why
  the m-th order deformation has the form it does, what Liao's
  convergence theorem actually says.
- [Worked examples](examples/index.md): full annotated runs of the two
  examples shipped with the library, including the
  adaptive-ℏ observation on quadratic drag.
- [API reference](api/index.md): the public surface of every module,
  extracted from docstrings.
