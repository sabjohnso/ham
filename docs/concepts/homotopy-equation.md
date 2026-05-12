# The homotopy equation

## The setting

We have a nonlinear differential or algebraic equation

\[
N\!\left[u(x)\right] = 0, \qquad x \in \Omega
\]

with appropriate boundary or initial conditions. \(N\) is some
nonlinear operator on functions of \(x\); the goal is to find
\(u\) satisfying both the equation and the BCs.

Classical perturbation methods assume the nonlinear part is "small"
and expand in a physical parameter that measures that smallness. HAM
makes no such assumption — there is no small parameter; the
expansion is along a *topological* parameter that the author of the
solve introduces by hand.

## The deformation

Pick an *initial guess* \(u_0(x)\) that satisfies the BCs (but
generically not the equation). Pick an *auxiliary linear operator*
\(L\), an *auxiliary function* \(H(x) \neq 0\), and a *convergence-
control parameter* \(\hbar \neq 0\). The **zeroth-order deformation
equation** is

\[
(1 - q)\, L\!\left[\,\varphi(x; q) - u_0(x)\,\right]
\;=\;
q\, \hbar\, H(x)\, N\!\left[\,\varphi(x; q)\,\right]
\]

with BCs \(\varphi(x; q) = \text{(original BCs)}\) and the auxiliary
embedding parameter \(q \in [0, 1]\).

This is a one-parameter family of equations indexed by \(q\):

- At \(q = 0\) the LHS reads \(L[\varphi(x; 0) - u_0(x)] = 0\), so by
  invertibility of \(L\) (modulo its kernel pinned by the BCs)
  \(\varphi(x; 0) = u_0(x)\). The deformation starts at the initial
  guess.
- At \(q = 1\) the LHS factor \((1 - q)\) vanishes; the RHS reads
  \(\hbar H(x) N[\varphi(x; 1)] = 0\). Since \(\hbar \neq 0\) and
  \(H \neq 0\), this forces \(N[\varphi(x; 1)] = 0\), i.e.,
  \(\varphi(x; 1) = u(x)\) — the deformation ends at the exact
  solution.

So as \(q\) sweeps from 0 to 1, \(\varphi(x; q)\) traces out a path in
function space from the initial guess to the exact solution. **If
that path is analytic in \(q\) at \(q = 0\)**, expand:

\[
\varphi(x; q)
\;=\;
\sum_{k=0}^{\infty} u_k(x)\, q^k,
\qquad
u_k(x) = \tfrac{1}{k!}\,\frac{\partial^k \varphi}{\partial q^k}\bigg|_{q=0}.
\]

\(u_0\) is the initial guess by the \(q = 0\) condition above. The
higher \(u_k\) are determined by matching powers of \(q\) on both
sides of the deformation equation — this is the
[deformation chain](deformation-chain.md), the next page.

## What the free knobs buy you

Four objects are at the author's disposal:

- The **base functions** through the choice of \(L\) and the structure
  of \(u_0\). Liao's first fundamental rule is that the base should
  reflect the structure of the expected solution. Polynomial base for
  polynomial-like solutions; exponential base for boundary-layer
  problems; Fourier base for periodic solutions.
- The **auxiliary function** \(H(x)\). Its job is to make every term
  of the original equation reachable by some \(u_k\) (Liao's second
  rule, *coefficient ergodicity*). For \(N\) polynomial in \(u\), and
  a polynomial base, \(H = 1\) usually works.
- The **convergence-control parameter** \(\hbar\). Tuning \(\hbar\)
  adjusts the rate and region of convergence of the partial sum at
  \(q = 1\). The optimal choice is problem-specific; the
  [convergence diagnostics](convergence.md) page covers the standard
  tools for finding it.
- The **initial guess** \(u_0(x)\). It must satisfy the original BCs
  (so the deformation chain only ever has to enforce homogeneous BCs
  on \(u_k\) for \(k \ge 1\)) but is otherwise free.

## What HAM is not

- HAM is not a perturbation method. There is no small parameter; the
  expansion in \(q\) is topological, and \(\hbar\) is a free knob,
  not a "small physical quantity."
- HAM is not a particular series expansion. The shape of the
  expansion comes from the chosen base, which the user picks.
- HAM is not guaranteed to converge. Liao's *Theorem 2.1* (the
  [convergence theorem](convergence.md)) says only that **if** the
  partial sum at \(q = 1\) converges, **then** it converges to the
  exact solution. Demonstrating that the *if* holds is the
  responsibility of the diagnostics layer.

## The library's encoding

Every element of the zeroth-order deformation equation has a direct
counterpart in [`ham.deformation.HamProblem`](../api/deformation.md):

| Math object | Library field | Provided by |
| --- | --- | --- |
| \(L\) | `HamProblem.L` | [`ham.operator.LinearOperator`](../api/operator.md) |
| \(N\) | `HamProblem.N` | [`ham.nonlinear.NonlinearOperator`](../api/nonlinear.md) |
| \(H(x)\) | `HamProblem.H` | sympy `Expr` |
| \(\hbar\) | `HamProblem.hbar` | sympy `Symbol` |
| \(u_0(x)\) | `HamProblem.u0` | sympy `Expr` |

The boundary conditions on \(L^{-1}\) (homogeneous, since \(u_0\)
already satisfies the originals) live on `LinearOperator.bcs` as a
tuple of `BoundaryCondition(point, derivative_order, value)`.

The deformation equation itself never appears in code as a literal
expression — the library extracts its consequences (the m-th order
coefficient recurrence) directly. See
[the deformation chain](deformation-chain.md) for that derivation
and its encoding.
