# Volterra population model

## The problem

The single-species Volterra population model with cumulative resource
depletion (Liao, *Beyond Perturbation*, Ch. 10):

\[
u'(t) = \kappa \, u(t) \left[\, 1 - u(t) - \int_0^t u(\tau)\,d\tau \,\right],
\qquad u(0) = \alpha.
\]

The integral term represents *accumulated* resource consumption: as
the population grows, it depletes the resources that have already
been consumed, eventually forcing decay. With \(\kappa > 0\) and
\(0 < \alpha < 1\), \(u(t)\) initially grows, peaks, and decays
asymptotically to 0.

We pick \(\kappa = 1\) and \(\alpha = 1/10\) — a small initial
population that grows substantially before being depleted.

## What makes this example different

This is the **first worked example with an integro-differential
\(N\).** The compiler's `Integral` branch (Stage 9a in
[`ham.nonlinear`](../api/nonlinear.md)) turns
`sp.Integral(u(s), (s, 0, t))` into coefficient-wise integration of
the homotopy series — the integro-differential structure flows
through the same `apply_series` machinery as polynomial-in-u \(N\),
no special-case path needed at the solver layer.

The example also surfaces a structural property absent from the four
prior worked examples: the HAM partial sum is a polynomial of degree
**twice** the working order, not the same as the working order. The
integral inside \(N\) adds a degree per HAM step on top of the degree
that \(L^{-1}\) adds.

## Liao's three rules in this example

1. **Solution expression.** \(u(t)\) is analytic at \(t = 0\) with a
   polynomial Taylor expansion (derivable directly from the
   integro-differential equation; see `taylor_reference` below). The
   polynomial base via \(L = d/dt\) reflects that structure.
2. **Coefficient ergodicity.** Each HAM step raises the polynomial
   degree of \(u_k(t)\) by **two** — one from \(L^{-1}\), one from
   the integral inside \(N\) — so every power \(t^k\) eventually
   appears in some \(u_m\).
3. **Solution existence.** \(u_0 = \alpha\) is a constant function
   satisfying the original BC \(u(0) = \alpha\) exactly. The
   deformation BCs are the homogeneous \(u_m(0) = 0\) for \(m \ge 1\).

## HAM setup

```python
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import solve

t = sp.Symbol("t")
s = sp.Symbol("s")
u = sp.Function("u")
hbar = sp.Symbol("hbar")
kappa = sp.Integer(1)
alpha = sp.Rational(1, 10)

problem = HamProblem(
    L=LinearOperator(
        var=t,
        action=lambda e: sp.diff(e, t),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    ),
    N=NonlinearOperator(
        expr=u(t).diff(t) - kappa * u(t) * (
            sp.Integer(1) - u(t) - sp.Integral(u(s), (s, 0, t))
        ),
        dependent=u,
        indep=t,
    ),
    H=sp.Integer(1),
    hbar=hbar,
    u0=alpha,
)
```

The `sp.Integral(u(s), (s, 0, t))` node is what the Stage 9a Integral
branch in `_compile` recognises. The integration variable `s` is a
fresh dummy; the upper limit is `t` (the HAM independent variable);
the lower limit must be `0`. Other shapes raise
`NotImplementedError` with a clear message.

## The Taylor reference

There is no closed-form analytic solution to the Volterra equation,
so the cross-check is against a Taylor expansion derived directly
from the integro-differential form. Substituting
\(u(t) = \sum_{k \ge 0} a_k t^k\) and matching powers of \(t\) gives
the recurrence

\[
(n + 1)\,a_{n+1} = \kappa\,a_n
    - \kappa \sum_{i + j = n} a_i\,a_j
    - \kappa \sum_{\ell = 0}^{n - 1} \frac{a_{n - 1 - \ell}\,a_\ell}{\ell + 1},
\qquad a_0 = \alpha.
\]

The `taylor_reference(order)` factory function in the example module
unwinds this recurrence. For \(\alpha = 1/10, \kappa = 1\) the first
few coefficients are

| \(k\) | \(a_k\) |
| --- | --- |
| 0 | \(1/10\) |
| 1 | \(9/100\) |
| 2 | \(31/1000\) |
| 3 | \(2/1875\) |
| 4 | \(-1291/400000\) |
| 5 | \(-929/750000\) |

The population peaks early (positive coefficients at \(t, t^2\))
then begins its decline (negative coefficients from \(t^4\)).

## The HAM-Taylor match

```python
from examples.volterra import solve_to, taylor_reference

sol = solve_to(5)
print(sol.evaluate_at_hbar(sp.Integer(-1)))
```

```
-691/226800000000*t**10 + 6541/25920000000*t**9
- 272621/40320000000*t**8 + 291139/5040000000*t**7
+ 1637/24000000*t**6 - 929/750000*t**5 - 1291/400000*t**4
+ 2/1875*t**3 + 31/1000*t**2 + 9/100*t + 1/10
```

The first six coefficients (\(t^0\) through \(t^5\)) **exactly
match** the Taylor reference at order 5. The library's regression
test `test_partial_sum_coefficients_match_taylor_up_to_order_m`
pins this for every \(M\) from 1 to 5.

Beyond \(t^5\), HAM produces **extra terms** (here \(t^6\) through
\(t^{10}\)) that do *not* match higher-order Taylor coefficients —
they are spurious approximations that get superseded as the working
order grows. At order \(M+1\) the \(t^{M+1}\) coefficient is filled
in correctly; the \(t^{M+2}\) through \(t^{2(M+1)}\) tail shifts to a
new spurious approximation. The polynomial degree of the HAM partial
sum is exactly \(2M\) for this problem, pinned by
`test_ham_polynomial_degree_is_twice_working_order`.

## Diagnostics

```python
from examples.volterra import analyze
analyze(solve_to(5))
```

```
residual_at_neg_one:    polynomial of degree 21 in t
hbar_curve_at_t_eq_1:   polynomial of degree 5 in hbar
optimal_hbar:           -1
l2_norm_at_optimal:     ≈ 8.80e-09
convergent_at_neg_one:  True
```

A few notes on what this output tells you:

### The residual is high-degree

The residual `N[partial sum]` is a polynomial of degree \(2 \cdot 2M -
1 + 1 = 4M - 1\) for \(N\) quadratic in \(u\) with a single integral.
At \(M = 5\) that's degree 19; the actual output is degree 21 because
of the extra terms from the `u·integral` cross product.

### Residual decrease is *not* strictly monotone

Unlike the polynomial-N worked examples, Volterra's residual norm is
not strictly monotone in \(M\): at \(M = 3\) the L² norm on \([0, 1]\)
is slightly *higher* than at \(M = 2\), before resuming its decrease.

| \(M\) | L² norm² on \([0, 1]\) | Δ vs prev |
| --- | --- | --- |
| 1 | \(8.94 \times 10^{-4}\) | — |
| 2 | \(4.96 \times 10^{-6}\) | \(-8.9 \times 10^{-4}\) |
| 3 | \(6.19 \times 10^{-6}\) | \(+1.2 \times 10^{-6}\) |
| 4 | \(9.71 \times 10^{-7}\) | \(-5.2 \times 10^{-6}\) |
| 5 | \(8.80 \times 10^{-9}\) | \(-9.6 \times 10^{-7}\) |
| 6 | \(2.96 \times 10^{-9}\) | \(-5.8 \times 10^{-9}\) |

Liao's *Theorem 2.1* only guarantees convergence in the limit, not
strict monotone decrease step-by-step. The regression test pins the
overall trend (the norm at \(M = 6\) is at least an order of
magnitude smaller than at \(M = 1\)) rather than asserting strict
monotonicity. The bump at \(M = 3\) is a real feature of the
Volterra deformation chain, not a numerical artifact.

### Optimal ℏ is \(-1\)

```python
optimal_hbar(sol, [-3/2, -1, -1/2, 0]) == -1
```

The L² advantage of \(\hbar = -1\) is clear on this problem; no
other grid value comes close. Numerical values across the grid:

| \(\hbar\) | L² norm² |
| --- | --- |
| \(-3/2\) | large (`> 10`) |
| \(-1\) | \(\approx 8.80 \times 10^{-9}\) |
| \(-1/2\) | larger than at \(-1\) but small |
| \(0\) | \(\approx 8.1 \times 10^{-3}\) |

### Validity gate

```python
from examples.volterra import is_convergent
is_convergent(solution, sp.Integer(-1))   # True
is_convergent(solution, sp.Integer(0))    # False
```

The default threshold is **\(1/1000\)** on the L² norm — tighter than
the quadratic-drag example's \(1/10\) because the initial population
\(\alpha = 1/10\) is small, which scales the residual magnitudes
down. At \(\hbar = 0\), the partial sum is \(u_0 = 1/10\) (constant),
giving residual \(N[1/10] = -0.09\) (constant), and L² norm² on
\([0, 1]\) is \(0.0081\) — well above the threshold.

## Running the example as a script

```sh
poetry run python examples/volterra.py
```

prints the partial sum, Taylor reference, residual, ℏ-curve at
\(t = 1\), grid-search result, and validity-gate value. Use it as
a sanity check or template for adding new integro-differential
worked examples.
