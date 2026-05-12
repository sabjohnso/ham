# Padé acceleration

## The problem with summing at \(q = 1\)

The bare HAM partial sum

\[
u^{(M)}(x) \;=\; \sum_{k=0}^{M} u_k(x)
\;=\; \varphi(x; q)\bigg|_{q = 1,\,\text{truncated}}
\]

evaluates the truncated power series in \(q\) at \(q = 1\). If the
formal series

\[
\varphi(x; q) \;=\; \sum_{k=0}^{\infty} u_k(x)\, q^k
\]

has *radius of convergence in \(q\) less than 1*, then summing at
\(q = 1\) is summing past the radius of convergence — the series
diverges as \(M \to \infty\), and the truncated partial sum sits
somewhere far from the exact solution.

This is not a HAM-specific pathology; it is the standard problem of
summing slowly- or non-convergent series. Padé approximants are the
classical fix.

## The Padé approximant in \(q\)

Given the truncated series

\[
\varphi(x; q) \;\approx\; c_0(x) + c_1(x)\, q + c_2(x)\, q^2 + \cdots + c_N(x)\, q^N,
\]

with \(c_k(x) = u_k(x)\) (the HAM coefficients), the **[L/M] Padé
approximant in \(q\)** is the rational function

\[
\frac{P(q)}{Q(q)}
\;=\;
\frac{p_0 + p_1 q + \cdots + p_L q^L}{1 + q_1 q + q_2 q^2 + \cdots + q_M q^M}
\]

with \(L + M = N\) (or \(\le N\)), chosen so that
\(P(q)/Q(q) - \varphi(x; q) = O(q^{N+1})\). The normalisation
\(Q(0) = 1\) makes the system uniquely determined when it is.

The coefficients \(p_i\) and \(q_j\) (themselves expressions in \(x\)
and possibly \(\hbar\)) are determined by a linear system: from
\(P(q) = Q(q)\,\varphi(x; q) + O(q^{N+1})\), matching coefficients
gives a square \(M \times M\) system for the denominator coefficients
followed by direct evaluation of the numerator.

The **homotopy-Padé sum** is the Padé approximant evaluated at
\(q = 1\):

\[
u^{(M)}_{\text{Padé}, [L/M]}(x) \;=\; \frac{P(1)}{Q(1)}
\;=\; \frac{p_0 + p_1 + \cdots + p_L}{1 + q_1 + q_2 + \cdots + q_M}.
\]

## Why this analytically continues

A Padé approximant of a truncated power series matches the series term-
by-term up to order \(N\), but as a *rational function* its
analytic behaviour extends past the radius of convergence of the
formal series. If \(\varphi(x; q)\) has a finite pole at \(q = q_0\)
with \(|q_0| < 1\), the rational \(P(q)/Q(q)\) can place a pole there
too and remain finite (and correct) at \(q = 1\).

The bare partial sum cannot do this: a truncated polynomial cannot
have a pole. Padé in \(q\) is the cheapest way to give the partial
sum that structural freedom.

## The library's encoding

The [`ham.pade.homotopy_pade`](../api/pade.md) function takes a
`HamSolution` and the two Padé orders:

```python
from ham.pade import homotopy_pade

homotopy_pade(solution, numerator_degree=L, denominator_degree=M,
              hbar_value=hbar_val)
```

The function:

1. Reads \(c_k = \text{solution.phi.coeff}(k)\) for \(k = 0, \ldots, N\).
2. Substitutes \(\hbar = \text{hbar\_value}\) into each \(c_k\)
   **before** building the Padé linear system. Late substitution can
   trade a vanishing symbolic determinant for a silent `nan`; early
   substitution surfaces the degeneracy as
   `NonInvertibleMatrixError` instead.
3. Solves the \(M \times M\) system for the denominator coefficients
   via `sympy.Matrix.LUsolve`.
4. Computes the numerator coefficients by direct substitution.
5. Returns \(P(1)/Q(1)\) as a sympy expression.

## The geometric-series cross-check

For the geometric problem \(u'(x) = u(x)^2\) with \(u(0) = 1\) and
exact solution \(u(x) = 1/(1 - x)\), HAM at \(\hbar = -1\) produces
\(u_k(x) = x^k\) — exactly the truncated geometric series. The bare
partial sum at \(q = 1\) is

\[
u^{(M)}(x) \;=\; 1 + x + x^2 + \cdots + x^M,
\]

which has radius of convergence 1 in \(x\) and diverges at \(x = 1\)
as \(M \to \infty\).

The **[0/1] homotopy-Padé** approximant in \(q\) is

\[
\frac{P(q)}{Q(q)}
\;=\;
\frac{p_0}{1 + q_1 q}
\;=\;
\frac{1}{1 - x q}.
\]

Evaluating at \(q = 1\) gives \(\dfrac{1}{1 - x}\) — the exact
closed-form solution. *From two HAM coefficients*: \(u_0 = 1\) and
\(u_1 = x\). The bare partial sum needs infinitely many; Padé gets
there in one denominator coefficient.

```python
sol = solve(geometric_problem, order=1)
homotopy_pade(sol, 0, 1, sp.Integer(-1))
# 1/(1 - x)
```

This is the headline demonstration: HAM + homotopy-Padé can produce
*exact closed-form* solutions for problems where the bare partial sum
would diverge. Liao §2.3.7 collects more examples; the library's
regression tests in `tests/test_pade.py` pin the cross-checks for the
geometric problem at orders 1 through 4 and the exp problem at order
4 (where \([2/2]\) Padé equals the classical
\((1 + x/2 + x^2/12)/(1 - x/2 + x^2/12)\) approximant of \(e^x\)).

## Degenerate orders

Not every \([L/M]\) is well-defined. For the geometric series, the
\([0/1]\) and any \([L/1]\) for \(L \ge 0\) all give \(1/(1-x)\), and
\([0/M]\) for \(M \ge 1\) does too — but \([1/2]\), \([2/2]\),
\([1/3]\), \([2/3]\), \(\ldots\) are *degenerate*: the denominator
linear system is singular because the geometric pole already lives at
one specific Padé order and higher denominators introduce redundant
equations.

The library propagates `NonInvertibleMatrixError` for those cases —
silent fallback would risk wrong answers, so the failure is loud:

```python
homotopy_pade(sol, 2, 2, sp.Integer(-1))
# raises NonInvertibleMatrixError
```

The caller picks a different \([L/M]\), or interprets the singularity
as evidence that the problem already has a small Padé representation
at lower orders.

## Choosing the orders

A reasonable starting heuristic for a partial sum of order \(N\):

- **\([N/0]\)** — the bare partial sum, no acceleration. Use as a
  baseline.
- **\([\lfloor N/2 \rfloor / \lceil N/2 \rceil]\)** — the symmetric
  choice. Often gives the best convergence in practice when no
  structural information about the solution is available.
- **\([0/N]\), \([1/(N-1)]\), \([L/1]\) for various \(L\)** —
  diagonal sweeps to find a non-degenerate order with low residual.

The residual diagnostics from [Convergence](convergence.md) apply to
the Padé-accelerated sum just as they do to the bare partial sum.
Substitute the Padé expression for the partial sum in your validity
gate; the rest of the workflow is unchanged.
