# Convergence

The HAM partial sum

\[
u^{(M)}(x) \;=\; u_0(x) + \sum_{m=1}^{M} u_m(x)
\;=\; \varphi(x; 1)\;\Big|_{\text{truncated at } q^M}
\]

is meaningful only when the underlying formal series in \(q\) converges
at \(q = 1\). If it does, *Liao's Theorem 2.1* guarantees it converges
to the exact solution. If it does not, the partial sum is a meaningless
truncation and acting on it would be wrong.

This page covers the convergence guarantee, the three fundamental rules
that shape it in practice, and the diagnostics the library provides for
defending a convergence claim on a specific problem.

## Liao's Theorem 2.1 — the conditional guarantee

**Theorem 2.1** (Liao, *Beyond Perturbation*, restated). *If the
homotopy series*

\[
\varphi(x; q) \;=\; \sum_{k=0}^{\infty} u_k(x)\, q^k
\]

*converges at \(q = 1\) — that is, the series \(\sum_{k=0}^{\infty}
u_k(x)\) converges — then it converges to the exact solution \(u(x)\)
of the original problem.*

The proof is short: at \(q = 1\) the zeroth-order deformation equation
gives \(\hbar H(x) N[\varphi(x; 1)] = 0\), and since \(\hbar \neq 0\)
and \(H \neq 0\) the partial sum at \(q = 1\) satisfies
\(N[\varphi(x; 1)] = 0\) — the original equation.

The theorem is **conditional**. There is no general guarantee of
convergence; the convergence has to be established per problem, either
by analytical argument or by the diagnostics below.

## The three fundamental rules

Liao codifies three rules for choosing the auxiliary operator \(L\),
the function \(H(x)\), and the initial guess \(u_0\) so that
convergence is *possible*. They do not guarantee convergence, but
violating them tends to make it impossible.

### Rule 1 — solution expression

The base functions implied by \(L\) and \(u_0\) should reflect the
structure of the expected solution.

- Polynomial in \(x\) → polynomial base, \(L = d^k/dx^k\), polynomial
  \(u_0\).
- Exponential decay → exponential base, \(L\) with constant
  coefficients, exponential \(u_0\).
- Periodic → Fourier base; \(L\) accommodating sines and cosines.

The library's examples illustrate the polynomial case. Both
`tanh(t)` (in the quadratic-drag example) and the sigmoid
`1/(1 + e^{-t})` (in the logistic example) have analytic
Taylor expansions about \(t = 0\), so a polynomial base via
\(L = d/dt\) with \(u_k(0) = 0\) is the natural choice.

### Rule 2 — coefficient ergodicity

Here "coefficient" means the **base coefficient** \(c_n\) in the
assumed solution expression \(u(x) = \sum_n c_n B_n(x)\), *not* the
\(q\)-coefficient \(u_k = \varphi.\text{coeff}(k)\). The two are
related — each base coefficient is built up from contributions across
the \(u_k\) — but the rule constrains the *base* coefficients, not the
HAM iterates.

For the deformation chain to *reach* every term of the expected
solution expression, \(H(x)\) and the structure of \(L\) must be
arranged so that no element of the base is identically zero in every
\(u_k\). If a term in the base is structurally unreachable, the chain
can never produce it and the partial sum cannot represent the exact
solution.

For polynomial \(N\) and a polynomial base, \(H(x) = 1\) typically
suffices — each \(u_k\) is a polynomial of degree growing with \(k\),
and every monomial \(x^j\) is reachable.

### Rule 3 — solution existence (of \(L^{-1}\))

The auxiliary linear operator \(L\) must be invertible on the function
space the deformation chain works in, with boundary conditions that
match \(u_0\)'s satisfaction of the original BCs.

In the library, \(L\) carries its boundary conditions as a first-class
field — every `LinearOperator.invert` call runs against the
**homogeneous** BCs (since \(u_0\) already satisfies the originals).
The inverter is pluggable: a canonical fast-path (`antiderivative` for
\(d/dt\) with \(u(0) = 0\)) is available, and `sympy.dsolve` is the
fallback for general \(L\). See
[`ham.operator`](../api/operator.md) for the surface.

## Diagnostics — defending a convergence claim

Given that the theorem is conditional, *every* claim of "the HAM
partial sum is the solution" must be backed by evidence. The library
provides four diagnostic primitives, all in
[`ham.diagnostics`](../api/diagnostics.md).

### The residual

The first cross-check is to apply \(N\) to the partial sum:

\[
\rho(x; \hbar, M) \;=\; N\!\left[\,u^{(M)}(x; \hbar)\,\right].
\]

If the partial sum *were* the exact solution, \(\rho\) would vanish
identically. In practice \(\rho\) is non-zero and measures how far
the truncated series is from satisfying the original equation.

```python
from ham.diagnostics import residual
residual(solution, hbar_value)
```

### Norms of the residual

Reducing \(\rho\) to a scalar requires a norm. Two flavors:

- **L² over an interval** — exact for polynomial residuals via
  `sp.integrate`:

  \[
  \lVert \rho \rVert_{L^2[a,b]}^2 \;=\; \int_a^b \rho(x)^2 \, dx.
  \]

- **Discrete sample sum-of-squares** — cheaper and more robust for
  transcendental residuals or unbounded domains:

  \[
  \lVert \rho \rVert_{\text{disc}}^2 \;=\; \sum_i \rho(x_i)^2.
  \]

```python
from ham.diagnostics import residual_l2_squared, residual_discrete_sum_of_squares
residual_l2_squared(solution, hbar_value, interval=(a, b))
residual_discrete_sum_of_squares(solution, hbar_value, samples=[x1, x2, ...])
```

### The ℏ-curve

At a fixed point \(x = x^\star\) and working order \(M\), the partial
sum is a polynomial in \(\hbar\) of degree \(\le M\). Liao's
*ℏ-curve* is the graph of this polynomial against \(\hbar\). A
*plateau* — a region of \(\hbar\) where the curve is roughly
horizontal — is a candidate convergence region: small changes in
\(\hbar\) produce small changes in the partial sum, suggesting the
series is converging there.

```python
from ham.diagnostics import hbar_curve_at
hbar_curve_at(solution, x_star)
# returns a sympy polynomial in hbar
```

The library stays a functional core. The polynomial *is* the curve;
plotting is the caller's job.

### Optimal-ℏ grid search

The "best" \(\hbar\) is the one minimising the residual norm over a
candidate plateau. The library provides a grid search that takes a
caller-supplied norm function (built by binding `residual_l2_squared`
or `residual_discrete_sum_of_squares` to a fixed interval or sample
set):

```python
from ham.diagnostics import optimal_hbar
def norm(s, h):
    return residual_l2_squared(s, h, (sp.Integer(0), sp.Integer(1)))
optimal_hbar(solution, hbar_grid=[-1.5, -1, -0.5], norm_fn=norm)
```

The grid-search approach is intentionally simple. Continuous
optimisation via `sp.solve(d/dhbar = 0)` is brittle for high-order
polynomials; grid search is robust and fast for the typical 5-to-20
grid points needed in practice.

### Validity gates — per-problem

The library deliberately does **not** ship a generic
`is_convergent(solution)` predicate. A baked-in threshold would risk
false comfort — convergence claims should be made per-problem with a
threshold and norm chosen for that problem's character. Each worked
example defines its own gate:

```python
# examples/quadratic_drag.py
def is_convergent(solution, hbar_value,
                  interval=(0, 1),
                  threshold=1/10):
    norm_squared = residual_l2_squared(solution, hbar_value, interval)
    return bool(norm_squared - threshold**2 < 0)
```

Use these as templates; adjust the threshold to your problem's
expected residual magnitude.

## When the bare partial sum doesn't converge

The diagnostics above ask "is the partial sum at \(q = 1\) a good
approximation of the exact solution?" For some problems the answer is
*no, but the formal series in \(q\) is fine* — the series just has
radius of convergence less than 1 in \(q\). In that case, you
analytically continue past the radius via
[homotopy-Padé](pade-acceleration.md), and the diagnostics on the
Padé-accelerated sum can pass even when the bare partial sum fails.
