# Blasius boundary-layer equation

## The problem

The Blasius flat-plate boundary-layer ODE describes the velocity
profile in a viscous boundary layer over a flat plate:

\[
f'''(\eta) + \tfrac{1}{2}\,f(\eta)\,f''(\eta) = 0,
\qquad f(0) = 0, \; f'(0) = 0, \; f'(\infty) = 1.
\]

The third condition is asymptotic — \(f'\) tends to 1 as
\(\eta \to \infty\), modeling the outer-flow boundary. There is no
closed-form solution; Howarth's high-precision numerical value
\(f''(0) \approx 0.469600\) is the standard convergence benchmark
for any HAM Blasius solver.

## What makes this example different

The previous worked examples either converge at \(\hbar = -1\)
(quadratic drag, logistic, Volterra match the Taylor reference
directly) or use the integro-differential pathway (Volterra).
Blasius is the first example where:

1. **\(\hbar = -1\) fails outright.** The polynomial-basis HAM series
   oscillates and diverges as \(M\) grows.
2. **The L² residual norm has a false plateau.** The standard
   diagnostic (`optimal_hbar` over a residual norm) finds a small
   positive \(\hbar\) where the residual is tiny but \(f''(0)\) is
   nowhere near Howarth's reference — the partial sum has collapsed
   close to \(u_0\), not converged to Blasius.
3. **The validity gate must be custom.** Closeness of \(f''(0)\) to
   Howarth's reference value is the per-example convergence test.

These are real features of the library's behaviour on a hard problem,
not workarounds.

## The truncated-domain trick

The asymptotic BC \(f'(\infty) = 1\) is not directly representable in
the library's `BoundaryCondition`, which expects a finite point.
This example replaces \(\infty\) with a large finite cutoff
\(\eta_{\max} = 10\), well past the boundary layer (which decays
around \(\eta \approx 5\)). The third BC becomes
\(f'(\eta_{\max}) = 1\), a regular point BC.

Liao's canonical HAM treatment (*Beyond Perturbation*, Ch. 14) uses
an **exponential basis** that encodes the asymptotic decay
directly — the library does not support that base out of the box,
which is why this polynomial-basis truncated form converges so
slowly. Adding exponential-basis support is on the roadmap; for now
this example documents the truncated workaround and its
characteristic failure modes.

## Liao's three rules in this example

1. **Solution expression.** Polynomial base is **not** the natural
   choice for Blasius — Liao explicitly uses an exponential base.
   The slow convergence we see here is Rule 1 making its argument:
   a wrong base costs orders.
2. **Coefficient ergodicity.** Each HAM step raises the polynomial
   degree of \(u_k\) by 3 (from \(L^{-1}\) of \(d^3/d\eta^3\)).
3. **Solution existence.** \(u_0 = \eta^2 / (2\eta_{\max})\)
   satisfies all three truncated BCs exactly:
   \(u_0(0) = 0\), \(u_0'(0) = 0\), \(u_0'(\eta_{\max}) = 1\). The
   deformation BCs are the homogeneous versions of these three.

## HAM setup

```python
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import solve

eta = sp.Symbol("eta")
f = sp.Function("f")
hbar = sp.Symbol("hbar")
eta_max = sp.Integer(10)

problem = HamProblem(
    L=LinearOperator(
        var=eta,
        action=lambda e: sp.diff(e, eta, 3),
        bcs=(
            BoundaryCondition(point=sp.Integer(0), derivative_order=0),
            BoundaryCondition(point=sp.Integer(0), derivative_order=1),
            BoundaryCondition(point=eta_max, derivative_order=1),
        ),
    ),
    N=NonlinearOperator(
        expr=f(eta).diff(eta, 3) + sp.Rational(1, 2) * f(eta) * f(eta).diff(eta, 2),
        dependent=f,
        indep=eta,
    ),
    H=sp.Integer(1),
    hbar=hbar,
    u0=eta**2 / (sp.Integer(2) * eta_max),
)
```

The three `BoundaryCondition` entries on `LinearOperator.bcs` are the
homogeneous versions of the three Blasius BCs — every deformation
step \(u_k\) for \(k \ge 1\) inherits them as the BCs for
\(L^{-1}\). `sympy.dsolve` (the default inverter) handles this 3rd-
order BVP cleanly.

## ℏ = -1 fails

```python
from examples.blasius import solve_to, f_double_prime_at_zero
sol = solve_to(5)
f_double_prime_at_zero(sol, sp.Integer(-1))
# ≈ -3.7422   (Howarth: 0.4696)
```

Wrong sign, wrong magnitude, and the error grows with \(M\):

| \(M\) | \(f''(0)\) at \(\hbar = -1\) |
| --- | --- |
| 1 | \(0.308\) |
| 2 | \(0.631\) |
| 3 | \(0.760\) |
| 4 | \(-0.321\) |
| 5 | \(-3.742\) |

This is the classical HAM divergence at the "wrong" \(\hbar\),
documented in [Concepts: Convergence](../concepts/convergence.md):
Liao's Theorem 2.1 only guarantees correctness *given* convergence,
and at \(\hbar = -1\) the series here does not converge.

## Finding a working ℏ

Sample the ℏ-curve of \(f''(0)\) across a grid of negative \(\hbar\)
values:

| \(\hbar\) | \(f''(0)\) at \(M = 5\) |
| --- | --- |
| \(-1\) | \(-3.742\) (diverges) |
| \(-0.7\) | \(-0.418\) |
| \(-0.5\) | \(+0.319\) |
| **\(-0.4\)** | **\(+0.418\)** — *closest to 0.4696* |
| \(-0.3\) | \(+0.405\) |
| \(-0.2\) | \(+0.324\) |
| \(-0.1\) | \(+0.212\) |
| \(0\) | \(+0.100\) (\(= u_0''(0) = 1/10\)) |

The plateau where the curve is near-flat sits around
\(\hbar \in [-0.5, -0.3]\). The closest grid value to Howarth's
reference is \(\hbar = -0.4\), giving \(f''(0) \approx 0.4178\) — an
11% relative error, which is acceptable given the polynomial-basis
slowdown.

## Why the L² residual norm misleads

The standard `optimal_hbar` grid search over `residual_l2_squared`
*does not* find \(\hbar = -0.4\). Instead it finds small positive
\(\hbar\) values where the residual norm is many orders smaller —
because the partial sum at small positive \(\hbar\) collapses near
\(u_0 = \eta^2 / 20\), whose residual on \([0, 1]\) is just
\((\eta^2/400)^2\) integrated, which is tiny.

This is a real characteristic of slow-converging HAM: the residual
norm can be small without the partial sum being close to the
solution. The library exposes the primitives; the validity gate is
the user's responsibility per problem. For Blasius the right gate
is closeness of \(f''(0)\) to Howarth's reference, not closeness of
the residual to zero.

## The example's custom validity gate

```python
def is_convergent(solution, hbar_value, tolerance=sp.Rational(1, 10)):
    """|f''(0) - Howarth| < tolerance."""
    fdd = f_double_prime_at_zero(solution, hbar_value)
    return bool(sp.Abs(fdd - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < tolerance)
```

| Call | Result |
| --- | --- |
| `is_convergent(sol, -1)` | `False` (divergent) |
| `is_convergent(sol, -2/5)` | `True` (within 0.052) |

Pinned by the regression tests
`test_validity_gate_fails_at_hbar_neg_one` and
`test_validity_gate_passes_at_best_hbar`.

## What this example confirms

Three properties hold by construction at every working order and at
every \(\hbar\), and the tests pin them:

1. **\(f(0) = 0\).** Trivially: \(u_0(0) = 0\), every \(u_k(0) = 0\).
2. **\(f'(0) = 0\).** Same reason: BC inherited.
3. **\(f'(\eta_{\max}) = 1\).** Exactly 1 symbolically in \(\hbar\):
   \(u_0\) contributes 1, every \(u_k\) for \(k \ge 1\) contributes 0
   (homogeneous BC). The library's regression test
   `test_partial_sum_satisfies_truncated_asymptotic_bc` verifies
   this symbolic identity rather than checking numerically at a few
   \(\hbar\).

What does *not* hold: closeness of \(f''(0)\) to Howarth depends on
\(\hbar\), and finding the right \(\hbar\) is the per-problem
diagnostic work.

## Where to go next

The library now ships an **[exponential-basis Blasius
example](blasius-exponential.md)** (Stage 11) that solves the same
problem using Liao's recommended basis. It:

- Uses `BoundaryCondition(point=sp.oo, derivative_order=1)` for the
  true asymptotic BC \(f'(\infty) = 1\), no domain truncation.
- Plugs in a custom inverter that filters growing-exp free
  constants left by `sympy.dsolve` when the resonance pattern
  defeats symbolic asymptotic-BC application.
- Reaches \(|f''(0) - \text{Howarth}| < 0.01\) at \(M = 3\) — an
  order of magnitude better than this polynomial-basis version
  achieves at \(M = 5\).

That example is the natural follow-up reading and demonstrates
Liao's Rule 1 (solution expression) by direct contrast with the
slow convergence here.

## Running the example as a script

```sh
poetry run python examples/blasius.py
```

prints \(f''(0)\) at \(\hbar = -1\) (showing divergence), the best
\(\hbar\) on the sweep grid, \(f''(0)\) there, the absolute error
vs Howarth, and the validity-gate result.
