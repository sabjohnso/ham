# Quadratic drag

## The problem

Take a falling object subject to quadratic air drag, with gravity and
the drag coefficient non-dimensionalised so that \(g/k = 1\). The
velocity \(v(t)\) of the object satisfies

\[
v'(t) = 1 - v(t)^2, \qquad v(0) = 0,
\]

whose closed-form solution is \(v(t) = \tanh(t)\) — terminal velocity
1 (in the dimensionless units) as \(t \to \infty\). The Taylor
expansion about \(t = 0\) is

\[
\tanh(t) = t - \tfrac{t^3}{3} + \tfrac{2 t^5}{15}
           - \tfrac{17 t^7}{315} + O(t^9).
\]

## Liao's three rules in this example

1. **Solution expression.** \(\tanh(t)\) is analytic at \(t = 0\) with
   a pure-polynomial Taylor expansion. The polynomial base implied by
   \(L = d/dt\) and \(u_0 = 0\) reflects that structure.
2. **Coefficient ergodicity.** Each HAM step raises the polynomial
   degree by one (the inverse of \(d/dt\)), so every power \(t^k\)
   eventually appears in some \(u_k(t)\). No structural blockers.
3. **Solution existence.** \(u_0 = 0\) satisfies the original BC
   \(v(0) = 0\) exactly, so the deformation BCs are the homogeneous
   \(u_k(0) = 0\) for \(k \ge 1\) — and \(L = d/dt\) with that BC has
   the explicit `antiderivative` inverter from
   [`ham.operator`](../api/operator.md).

## HAM setup

```python
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import solve

t = sp.Symbol("t")
v = sp.Function("v")
hbar = sp.Symbol("hbar")

problem = HamProblem(
    L=LinearOperator(
        var=t,
        action=lambda e: sp.diff(e, t),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    ),
    N=NonlinearOperator(
        expr=v(t).diff(t) - sp.Integer(1) + v(t) ** 2,
        dependent=v,
        indep=t,
    ),
    H=sp.Integer(1),
    hbar=hbar,
    u0=sp.Integer(0),
)
```

`N[v] = v' - 1 + v^2` so that `N[tanh] = 0` (checking: \(\tanh'(t) =
\operatorname{sech}^2(t) = 1 - \tanh^2(t)\), so \(\tanh' - 1 +
\tanh^2 = 0\)). \(u_0 = 0\) is the simplest function satisfying
\(v(0) = 0\); the deformation chain handles every higher-order term.

## Solve and verify

```python
solution = solve(problem, order=7)
print(solution.evaluate_at_hbar(sp.Integer(-1)))
# -17*t**7/315 + 2*t**5/15 - t**3/3 + t   ← tanh Taylor to order 7
```

At \(\hbar = -1\) the HAM partial sum to order 7 equals the truncated
Taylor expansion of \(\tanh(t)\) term for term. The regression test in
`tests/examples/test_quadratic_drag.py` pins this match at orders 5
and 7; an additional test asserts that every *even* power of \(t\)
vanishes in the partial sum at \(\hbar = -1\) (since \(\tanh\) is odd).

## Diagnostics

```python
from examples.quadratic_drag import analyze
analyze(solution)
```

```
residual_at_neg_one:    289*t**14/99225 - 68*t**12/4725 + 254*t**10/4725
                        - 62*t**8/315
hbar_curve_at_t_eq_1:   647*hbar**7/315 + 35*hbar**6/3 + 91*hbar**5/5
                        - 70*hbar**3/3 - 21*hbar**2 - 7*hbar
optimal_hbar:           -1/2
l2_norm_at_optimal:     ≈ 3.44e-04
convergent_at_neg_one:  True
```

Three observations:

### The residual

\(N\) applied to the partial sum gives a polynomial starting at
\(t^8\) (the next missing term of the Taylor expansion fed through
\(v' - 1 + v^2\)). The norm of this on the unit interval is the
quantitative measure of how far we are from satisfying the ODE.

### The ℏ-curve at \(t = 1\)

The partial sum at \(t = 1\) is a degree-7 polynomial in \(\hbar\). Its
graph against \(\hbar\) is Liao's *ℏ-curve* — a plateau near
\(\hbar = -1\) would indicate convergence. For this problem the curve
shape is:

\[
\varphi(1; \hbar) = \tfrac{647}{315}\hbar^7 + \tfrac{35}{3}\hbar^6
                    + \tfrac{91}{5}\hbar^5 - \tfrac{70}{3}\hbar^3
                    - 21 \hbar^2 - 7 \hbar
\]

(no \(\hbar^4\) term, no constant — \(v_0(1) = 0\) so the curve passes
through the origin). Sketch it from \(\hbar \in [-1.5, 0]\) and the
plateau region is around \(\hbar \in [-1, -1/2]\) — exactly where the
grid search finds the optimum.

### Optimal ℏ is *not* \(-1\)

This is the headline observation. On the grid \([-3/2, -1, -1/2, 0]\)
at order 7 the L² norm² values are:

| \(\hbar\) | L² norm² |
| --- | --- |
| \(-3/2\) | \(\approx 3.44\) |
| \(-1\) | \(\approx 1.47 \times 10^{-3}\) |
| \(-1/2\) | \(\approx 3.44 \times 10^{-4}\) |
| \(0\) | \(1.0\) |

The L² norm at \(\hbar = -1/2\) is about *four times smaller* than at
\(\hbar = -1\). The HAM partial sum at \(\hbar = -1/2\) is *not* the
truncated Taylor of \(\tanh\); it is a different polynomial that
happens to distribute its error across the unit interval more
favourably than plain Taylor truncation does.

**This is the HAM adaptive-ℏ advantage.** Tuning \(\hbar\) trades
accuracy at one point for accuracy across an interval. For
applications that only care about the integral norm or a region of
interest away from \(t = 0\), the non-Taylor choice can be the right
one.

The behavioural test
`test_optimal_hbar_returns_grid_minimum` in
`tests/examples/test_quadratic_drag.py` deliberately does not pin
\(\hbar = -1\); it asserts only that the grid minimum is what the
function returns and that the obviously-bad endpoints (\(-3/2\),
\(0\)) are never selected.

### The validity gate

```python
from examples.quadratic_drag import is_convergent
is_convergent(solution, sp.Integer(-1))   # True
is_convergent(solution, sp.Integer(0))    # False
```

The threshold is \(1/10\) on the L² norm over \([0, 1]\) — calibrated
so that an underconverged \(\hbar = 0\) case (where the partial sum
is just \(u_0 = 0\) and the residual is \(-1\) constant, norm² = 1)
fails clearly, while the converged \(\hbar = -1\) case (norm² ≈
\(1.5 \times 10^{-3}\) at order 7) passes.

## Running the example as a script

```sh
poetry run python examples/quadratic_drag.py
```

prints the full diagnostic bundle and ends with the validity-gate
result. Use it as a sanity check or as a template for new examples.
