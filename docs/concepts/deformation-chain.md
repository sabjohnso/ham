# The deformation chain

The zeroth-order deformation equation

\[
(1 - q)\, L\!\left[\,\varphi(x; q) - u_0(x)\,\right]
\;=\;
q\, \hbar\, H(x)\, N\!\left[\,\varphi(x; q)\,\right]
\]

determines \(\varphi(x; q)\) implicitly. Expanding

\[
\varphi(x; q) \;=\; \sum_{k=0}^{\infty} u_k(x)\, q^k
\]

and matching powers of \(q\) on both sides gives an explicit recurrence
for the \(u_k\). This page derives that recurrence and shows how the
library extracts it.

## The expansion, term by term

The LHS:

\[
(1 - q)\, L\!\left[\varphi - u_0\right]
\;=\;
(1 - q)\, L\!\left[\sum_{k=1}^{\infty} u_k\, q^k\right]
\;=\;
(1 - q)\, \sum_{k=1}^{\infty} L[u_k]\, q^k
\]

by linearity of \(L\). Expanding the \((1 - q)\) factor:

\[
(1 - q)\, \sum_{k=1}^{\infty} L[u_k]\, q^k
\;=\;
L[u_1]\, q
\;+\;
\sum_{m=2}^{\infty} \bigl(L[u_m] - L[u_{m-1}]\bigr)\, q^m.
\]

The RHS:

\[
q\, \hbar\, H(x)\, N[\varphi]
\;=\;
\hbar\, H(x)\, \sum_{m=1}^{\infty} \!\bigl[q^{m-1}\bigr]\!\,N[\varphi]\, q^m,
\]

where \([q^{m-1}] N[\varphi]\) denotes the Taylor coefficient of
\(N[\varphi]\) at \(q^{m-1}\).

Matching coefficient of \(q^m\):

- For \(m = 1\): \(\quad L[u_1] \;=\; \hbar\, H(x)\, [q^0] N[\varphi]
  \;=\; \hbar\, H(x)\, N[u_0]\),
  since \(\varphi(x; 0) = u_0\) and so \([q^0] N[\varphi] = N[u_0]\).
- For \(m \ge 2\): \(\quad L[u_m] - L[u_{m-1}] \;=\;
  \hbar\, H(x)\, [q^{m-1}] N[\varphi]\).

These two cases combine into the **m-th order deformation equation**:

\[
\boxed{\;
L\!\left[\,u_m - \chi_m\, u_{m-1}\,\right]
\;=\;
\hbar\, H(x)\, R_m,
\qquad
R_m \;\equiv\; \bigl[q^{m-1}\bigr]\, N[\varphi],
\;}
\]

with the *deformation heaviside*

\[
\chi_m \;=\; \begin{cases} 0 & m \le 1, \\ 1 & m \ge 2. \end{cases}
\]

Crucially, this equation is *linear* in the unknown \(u_m\) — the
nonlinearity sits inside \(R_m\), which depends only on the lower-order
\(u_0, \ldots, u_{m-1}\) that have already been solved for at this
point. The chain runs **strictly forward**: to find \(u_m\), invert
\(L\) once.

## A convention note

Liao's textbook often writes \(R_m\) as the unnormalised \(q\)-derivative

\[
R_m^{\text{Liao}} \;=\; \frac{\partial^{m-1} N[\varphi]}{\partial q^{m-1}}\bigg|_{q=0}
\;=\; (m-1)!\, \bigl[q^{m-1}\bigr] N[\varphi],
\]

so the factor of \((m-1)!\) appears in his statement of the m-th order
equation. The library uses the **Taylor-coefficient** form
\(R_m = [q^{m-1}] N[\varphi]\) throughout. The two are equivalent up to
the prefactor, which gets absorbed into a different scaling of the
\(u_k\) in Liao's convention (\(u_m^{\text{Liao}} = m!\,u_m^{\text{us}}\)).
Same partial sum at \(q = 1\); same physics; cleaner code.

## Why \(R_m\) is *causal in q*

Look at the dependency chain in \(R_m\):

\[
N[\varphi] \;=\; N\!\left[\,\sum_{k=0}^{\infty} u_k\, q^k\,\right].
\]

For \(N\) polynomial of degree \(d\) in \(u\) and its \(x\)-derivatives,
\(N[\varphi]\) is itself a formal series in \(q\) whose coefficient at
\(q^{m-1}\) is a sum of products of the \(u_k\), with the index
constraint \(\sum k_i \le m - 1\). The high-\(q\) tail of \(\varphi\)
*cannot* leak into low-\(q\) coefficients of \(N[\varphi]\) — the
multiplication is a Cauchy product, structurally forward.

This is the property the library calls
**causality in q**, asserted by property tests in `tests/test_pade.py`
and `tests/test_deformation.py`: perturbing
`phi.coeff(k)` for \(k \ge m\) leaves `r_m(phi, m)` invariant.

## The library's encoding

`R_m` and the m-th equation's right-hand side are methods on
[`HamProblem`](../api/deformation.md):

```python
problem.r_m(phi, m)          # → R_m = [q^{m-1}] N[phi]
problem.rhs_m(phi, m)        # → ℏ · H(x) · R_m
```

The heaviside is a free function:

```python
from ham.deformation import chi_m
chi_m(0), chi_m(1), chi_m(2), chi_m(7)
# (0, 0, 1, 1)
```

The full one-step solve, including the \(L\)-inversion, lives in
`ham.solver.solve_step`:

```python
def solve_step(problem, phi, m):
    rhs = problem.rhs_m(phi, m)
    v   = problem.L.invert(rhs)
    return sp.expand(v + chi_m(m) * phi.coeff(m - 1))
```

`solve(problem, order)` iterates this from \(m = 1\) up to the working
order, growing the partial sum one coefficient at a time. See
[`ham.solver`](../api/solver.md) for the driver.

## Reading the coefficient chain

For the logistic problem \(u'(t) = u(t)(1 - u(t))\) with
\(u(0) = \tfrac{1}{2}\):

| \(k\) | \(u_k(t)\) (ℏ symbolic) | \(u_k(t)\) at \(\hbar = -1\) |
| --- | --- | --- |
| 0 | \(1/2\) | \(1/2\) |
| 1 | \(-\hbar t / 4\) | \(t / 4\) |
| 2 | \(-\hbar^2 t/4 - \hbar t/4\) | \(0\) |
| 3 | \(\hbar^3 t^3/48 - \hbar^3 t/4 - \hbar^2 t/2 - \hbar t/4\) | \(-t^3/48\) |

Adding the rows at \(\hbar = -1\) up to order 3 gives the truncated
sigmoid Taylor expansion
\(1/2 + t/4 - t^3/48\) — every even-order coefficient \(u_2, u_4, \ldots\)
vanishes at \(\hbar = -1\) because the sigmoid has only odd corrections
about its centre value.

The same pattern shows up in every worked example; the
[examples gallery](../examples/index.md) walks through it.
