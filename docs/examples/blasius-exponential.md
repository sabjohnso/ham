# Blasius via exponential basis

## The problem

Same as the [polynomial-basis Blasius](blasius.md) example:

\[
f'''(\eta) + \tfrac{1}{2}\,f(\eta)\,f''(\eta) = 0,
\qquad f(0) = 0,\;f'(0) = 0,\;f'(\infty) = 1.
\]

But solved using Liao's recommended **exponential basis** (Ch. 14):

- True asymptotic BC `f'(∞) = 1`, no domain truncation.
- Initial guess \(u_0(\eta) = \eta - 1/\alpha + e^{-\alpha\eta}/\alpha\)
  satisfies all three original BCs by construction.
- \(L = d^3/d\eta^3 - \alpha^2 \cdot d/d\eta\) — kernel
  \(\{1, e^{\alpha\eta}, e^{-\alpha\eta}\}\), constructed to absorb
  the exponential structure of the solution.
- Second free parameter \(\alpha\) (fixed at \(\alpha = 1\) in this
  example; Liao's standard baseline).

## What this example demonstrates

This is the library's most sophisticated worked example. It
demonstrates four properties that none of the earlier examples does:

1. **Asymptotic boundary conditions** via `BoundaryCondition(point=sp.oo, ...)`.
   The library's `LinearOperator` already accepts \(sp.oo\) as a BC
   point (verified by the Stage 11a regression test); the
   dsolve-backed inverter handles it for simple RHS shapes.
2. **A custom inverter** plugged into `LinearOperator.inverter` to
   work around a sympy.dsolve limitation. dsolve cannot symbolically
   apply \(f'(\infty) = 0\) when the result contains growing-exp
   kernel components — see the next section.
3. **An exponential basis** via the coefficient ring of `QSeries`.
   QSeries was always polymorphic in the coefficient ring (any
   `sp.Expr`), so this example exercises a basis the library was
   never restricted to.
4. **A second free parameter** \(\alpha\) alongside \(\hbar\). The
   `HamProblem` data type carries only one named convergence
   parameter (`hbar`), but additional sympy symbols ride through
   the deformation chain naturally. The example keeps \(\alpha\) as a
   sympy symbol through the entire HAM solve; the Stage 12 library
   extension
   [`ham.diagnostics.optimal_parameters`](../api/diagnostics.md)
   does a two-parameter grid search over \((\hbar, \alpha)\) and
   typically finds an optimum that beats fixing \(\alpha = 1\) by an
   order of magnitude on the same working order.

## The sympy.dsolve workaround

The kernel of \(L = d^3/d\eta^3 - d/d\eta\) is
\(\{1, e^{\eta}, e^{-\eta}\}\). When the HAM RHS contains terms like
\(\eta\,e^{-\eta}\) (resonance with the kernel element \(e^{-\eta}\)),
sympy.dsolve produces a particular solution plus an unresolved free
constant \(C_3\) parametrising the **growing-exponential** branch
\(e^{\eta}\). It cannot apply the asymptotic BC \(f'(\infty) = 0\)
symbolically because \(e^{\eta}.subs(\eta, \infty)\) is not a finite
expression.

Workaround: solve the ODE with only the two point BCs at \(\eta = 0\),
then **zero out any free constants** in the result:

```python
def _blasius_exponential_inverter(rhs):
    u = sp.Function("_u")(ETA)
    ode = sp.Eq(u.diff(ETA, 3) - ALPHA**2 * u.diff(ETA), rhs)
    ics = {u.subs(ETA, 0): 0, u.diff(ETA).subs(ETA, 0): 0}
    sol = sp.dsolve(ode, u, ics=ics)
    result = sol.rhs
    free = result.free_symbols - rhs.free_symbols - {ETA}
    if free:
        result = result.subs(dict.fromkeys(free, sp.Integer(0)))
    return sp.simplify(result)
```

For this \(L\) the free constant *always* multiplies the growing-exp
kernel direction \(e^{\alpha\eta}\), so zeroing it correctly enforces
\(f'(\infty) = 0\). Verified algebraically:

- Hand-derive \(u_1\) by setting \(C_3 = 0\) in the dsolve output.
- Verify the result satisfies all three BCs (`u(0) = 0`,
  `u'(0) = 0`, `u'(∞) = 0`) and the ODE \(L[u_1] = \hbar\,N[u_0]\).
- The regression test `test_inverter_zeros_growing_exp_branch_on_resonant_rhs`
  pins this property on a resonant RHS.

The inverter is plugged in via
`LinearOperator(..., inverter=_blasius_exponential_inverter)`, so the
library uses it instead of the default `_dsolve_invert` for this
problem.

## Convergence: dramatically faster than polynomial basis

The Stage 12 library extension exposes \(\alpha\) as a sympy symbol
and provides a two-parameter grid search via `optimal_parameters`.
The example's `analyze()` sweeps the (ℏ, α) plane and reports the
joint optimum:

```text
Best (ℏ, α) found by grid search at each working order:
  M  best ℏ   best α    f''(0)   |error|   gate
  1   -4/5    7/10     +0.46762  0.00198   True
  2   -1/2    7/10     +0.47003  0.00043   True
```

Tuning \(\alpha\) alongside \(\hbar\) is a **substantial**
improvement over the Stage 11 single-parameter search with α=1:

| Strategy | M | f''(0) | |error| | vs Howarth |
| --- | ---: | ---: | ---: | --- |
| Polynomial basis (Stage 10) | 5 | 0.4178 | 0.0518 | 11.0% off |
| Exp basis, α=1 only (Stage 11) | 2 | 0.4644 | 0.0052 | 1.1% off |
| **Exp basis, (ℏ, α) jointly (Stage 12)** | **2** | **0.4700** | **0.00043** | **0.09% off** |

Two parameters and two HAM iterations beat the polynomial basis at
M=5 by more than two orders of magnitude. This is Liao's Rule 1
(solution expression) **plus** a free parameter that the library now
fully exposes — choose the right base and tune the right knobs, and
the convergence rate transforms.

The regression test
`test_two_parameter_beats_polynomial_basis_at_higher_order` pins
this comparison: exponential-basis with the joint optimum at M=2 is
asserted at least 50x better than polynomial-basis at M=5.

## Why \((\hbar, \alpha)\) shifts with order

The "best \((\hbar, \alpha)\)" varies with the working order — both
parameters drift as more HAM terms are summed:

\[
(\hbar^*, \alpha^*)(M=1) = (-4/5, 7/10), \quad
(\hbar^*, \alpha^*)(M=2) = (-1/2, 7/10).
\]

This is normal HAM behaviour. The partial sum is an approximation of
the homotopy series at \(q = 1\); at finite truncation, the optimal
convergence-control parameter depends on truncation order. As
\(M \to \infty\) the optimal \(\hbar\) typically stabilises.

Practical workflow:

1. Pick a working order \(M\) your solve budget supports.
2. Sweep \(\hbar\) on a grid and pick the value minimising your
   convergence criterion (here, \(|f''(0) - \text{Howarth}|\)).
3. If the result is good enough, stop. Otherwise raise \(M\) and
   re-sweep.

The `analyze(solution)` factory in the example does steps 2–3
automatically over the grid \(\{-1.5, -1.4, \ldots, 0\}\).

## What holds by construction

The three BCs hold for the partial sum at *every* working order and
*every* \(\hbar\) — verified symbolically by tests:

| BC | Test |
| --- | --- |
| \(f(0) = 0\) | `test_partial_sum_satisfies_f_zero_is_zero` |
| \(f'(0) = 0\) | `test_partial_sum_satisfies_f_prime_zero_is_zero` |
| \(f'(\infty) = 1\) | `test_partial_sum_satisfies_asymptotic_bc` |

Each \(u_k\) for \(k \ge 1\) has \(f'(\infty) = 0\) (the homogeneous
deformation BC, enforced by the custom inverter), so the partial
sum's limit at infinity equals \(u_0'(\infty) = 1\) exactly,
regardless of \(\hbar\). The Stage 10 polynomial-basis Blasius could
only achieve this in the truncated sense \(f'(\eta_{\max}) = 1\);
this stage achieves it as a true asymptotic limit.

## The validity gate

```python
def is_convergent(solution, hbar_value, tolerance=sp.Rational(1, 50)):
    """|f''(0) - Howarth| < tolerance."""
    fdd = f_double_prime_at_zero(solution, hbar_value)
    return bool(sp.Abs(fdd - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < tolerance)
```

Default tolerance \(1/50 = 0.02\) — five times tighter than the
polynomial-basis Blasius's \(1/10\). The exponential basis converges
fast enough to justify the tighter gate.

## Running as a script

```sh
poetry run python examples/blasius_exponential.py
```

prints \(f''(0)\), absolute error, and gate result at the best
\(\hbar\) for each \(M\) from 1 to 4. Roughly 10 seconds total.

## What remains scope-deferred

- **Higher-order solves.** Each HAM step calls sympy.dsolve on a
  progressively more complex RHS with symbolic \(\alpha\); M = 2 takes
  ≈ 3 s, M = 3 takes ≈ 12 s. Liao reports \(f''(0) = 0.469600\)
  accurate to 6 decimals at M ≈ 30; reaching that with our library
  would benefit from a faster custom inverter that decomposes the RHS
  over the basis \(\{\eta^j e^{-k\alpha\eta}\}\) and applies a
  closed-form L^{-1} formula per basis element. The current dsolve-
  backed inverter is correct but generic.
