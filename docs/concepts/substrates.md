# Substrates: where the coefficients live

The [homotopy equation](homotopy-equation.md) and the
[deformation chain](deformation-chain.md) say nothing about *what*
the coefficients \(u_k(x)\) of the homotopy series

\[
\varphi(x; q) \;=\; \sum_{k=0}^{\infty} u_k(x)\, q^k
\]

actually *are*. The math holds in any setting where:

- the \(u_k\) live in a function space where \(L\) and \(N\) make
  sense,
- you can do arithmetic on them (add, subtract, multiply, divide by
  integers),
- you can solve \(L[u_m] = \text{RHS}\) under the deformation
  boundary conditions,
- you can apply the linear / nonlinear operators that appear in
  \(N\) (differentiation in \(x\), pointwise products, integrals
  against simple bounds).

A **substrate** is a concrete choice of *what coefficients are* plus
*how those arithmetic and calculus operations are implemented*. The
library is generic over the substrate: every module from
[`ham.series`](../api/series.md) up to
[`ham.diagnostics`](../api/diagnostics.md) takes a type parameter
\(C\) for the coefficient type and works uniformly across whichever
substrate the caller passes in.

The library ships two substrates. They share every line of the
solver loop, the deformation builder, the nonlinear-operator tree
walker, and (almost all of) the diagnostic surface — the only thing
that varies is the substrate itself.

## The Backend protocol

Concretely, a substrate is summarised by six operations on the
coefficient type \(C\):

| operation | type | meaning |
| --- | --- | --- |
| `zero()` | \(\to C\) | the additive identity in \(C\) |
| `one()` | \(\to C\) | the multiplicative identity in \(C\) |
| `lift_xonly(expr)` | \(\texttt{sympy.Expr} \to C\) | the user writes \(L\), \(N\), \(u_0\), \(H\) in the sympy *authoring language*; the lift translates an \(x\)-only sympy expression into the substrate's coefficient form |
| `diff_x(c, k)` | \((C, \texttt{int}) \to C\) | \(k\)-th derivative w.r.t. the independent variable |
| `integrate_x(c)` | \(C \to C\) | antiderivative from the left boundary of the substrate's domain |
| `normalize(c)` | \(C \to C\) | substrate-specific canonical form |

These six operations are exactly what the algebraic core needs that
*cannot* be expressed polymorphically through Python's `+`, `-`,
`*`, `/int` on \(C\). The latter — the ring + scalar-rational
operations — are assumed to dispatch natively on \(C\); both
`sympy.Expr` and `numpy.ndarray` satisfy this without help.

The [`Backend[C]`](../api/backend.md) Protocol packages these six
operations as a frozen dataclass. To add a new substrate (Fourier
on the circle, Legendre-Gauss for alternative quadrature, …) you
write one factory function returning a `Backend[YourCoefficient]`
and the rest of the library — solver, diagnostics, contracts —
comes for free. A second concrete substrate — rational-Chebyshev
on \([0, \infty)\) — ships as a minimal Grid implementation today
(differentiation matrix only; rational-Cheb quadrature and
asymptotic-BC handling are tracked follow-ups in the [Stage
history](../design/stages.md#tracked-follow-ups)).

Equality is *not* a Backend operation. The library's design
decision is that the comparator used to decide "are these two
coefficients equal" lives at the verification site, not on the
substrate. The sympy comparator is `sp.expand(a - b) == 0`; the
spectral float comparator is `np.allclose`; the spectral sympy-
scalar comparator is element-wise sympy expansion. Keeping these out
of the Backend lets [`ham.operator.LinearOperator`](../api/operator.md)
stay substrate-agnostic — it depends on `Callable[[C], C]`, not on
the substrate-specific notion of "close enough".

## The sympy substrate

The classical symbolic-HAM substrate. \(C = \texttt{sympy.Expr}\):

- Each \(u_k(x)\) is a sympy expression in the independent variable.
- \(L.\text{invert}\) calls `sp.dsolve` (or a hand-coded shortcut
  like the `antiderivative` for \(L = d/dx\)) under the declared
  boundary conditions.
- `N.apply_series(phi)` compiles the user's sympy \(N\) expression
  into a recursive tree-walker that emits `Series` arithmetic
  against \(\varphi\); products are Cauchy, derivatives are
  coefficient-wise `sp.diff`, integrals are coefficient-wise
  `sp.integrate`.
- The convergence-control parameter \(\hbar\) is kept symbolic
  through every stage of the solve. The partial sum is a polynomial
  in \(\hbar\); the user substitutes \(\hbar\) at the end, plots
  the [ℏ-curve](convergence.md), and grid-searches for an optimal
  value.
- Cost scales with the algebraic complexity of \(N\) and \(u_0\),
  not with a spatial grid resolution. The output is *exact* (in the
  sense of symbolic computation) at every order.

This substrate is what the [tutorial](../tutorial.md) walks through
on the logistic equation and what every [worked
example](../examples/index.md) uses.

## The spectral substrate (SHAM)

The Spectral HAM substrate of Motsa, Sibanda, and Shateyi
(*CNSNS* 15, 2010). \(C = \texttt{numpy.ndarray}\):

- Each \(u_k\) is an `ndarray` of length \(N + 1\) holding the
  function's nodal values on a Chebyshev-Gauss-Lobatto grid (or
  any other implementation of the [`Grid`](../api/grids.md)
  protocol).
- \(L.\text{invert}\) becomes a dense linear solve: parse the
  user's sympy \(L\) expression into the matrix
  \(L_{\text{mat}} = \sum_k \mathrm{diag}(c_k(x)) \cdot D^k\),
  impose each boundary condition by replacing one row of
  \(L_{\text{mat}}\) with the row that evaluates
  \(u^{(\text{bc.order})}\) at the boundary node, then solve.
- `N.apply_series(phi)` runs the *same* tree-walker as on the
  sympy substrate; only the leaves change — derivatives become
  matrix-vector products with cached powers of the
  differentiation matrix, integrals become solves against the
  same matrix with a left-boundary constraint.
- The spectral L² norm of the residual is Clenshaw-Curtis
  quadrature against the grid's weights — exact for polynomials
  up to degree \(N\), and the lever
  [`residual_l2_squared(grid=...)`](../api/diagnostics.md) takes
  to translate the sympy `sp.integrate` path.

### The dual-scalar split

Inside the spectral substrate the library supports two scalar
choices for the array elements, which affect the trade-off between
speed and symbolic structure:

- **`scalar="float"`** — array entries are `float64`. The linear
  solver in `L.invert` is `np.linalg.solve`. \(\hbar\) is a numeric
  parameter set at problem construction (`hbar=sp.Float(-1.0)`)
  and *every iterate* is a numeric grid vector. This is the
  classical SHAM mode: fast linear algebra, no symbols, the
  ℏ-curve at a point becomes a single numeric value and the
  user builds the curve by sweeping over \(\hbar\) externally
  (`hbar_curve_at_sweep` re-runs the solver per \(\hbar\)).

- **`scalar="sympy"`** — array entries are `sympy.Expr` (numpy
  object-dtype). The linear solver is `sp.Matrix.LUsolve`. \(\hbar\)
  stays symbolic *inside* every grid entry, so the partial sum at
  any grid node is a polynomial in \(\hbar\) — the same
  interpretation as the sympy substrate. The cost is real:
  `sp.Matrix.LUsolve` on object-dtype matrices is much slower than
  `np.linalg.solve` on floats, and the per-iterate sympy arithmetic
  adds further cost. Use this when the ℏ-curve interpretation is
  worth more than the speed.

Both scalars share every line of the spectral substrate's code;
only the linear solver and the array dtype branch. The
[design decision (D-1)](../design/stages.md#substrate-parametrisation-arc-s0-s9-2026-05-1317)
to support both is what lets the library serve readers who want
classical SHAM speed and readers who want the symbolic ℏ-curve
without forking the substrate code.

## Choosing a substrate

A rough decision table:

| If you want / care about… | Use… |
| --- | --- |
| Closed-form solutions (the answer *is* a sympy expression) | sympy substrate |
| Closed-form Padé acceleration | sympy substrate (spectral Padé deferred) |
| The bare ℏ-curve as a polynomial you can plot, optimise, and substitute analytically | sympy substrate, or spectral with `scalar="sympy"` |
| Fast convergence at moderate orders on a fixed numeric domain | spectral, `scalar="float"` |
| Spectral accuracy of the residual norm (Clenshaw-Curtis quadrature, not sympy `sp.integrate`) | spectral substrate |
| Problems with strongly varying \(L\) coefficients or hand-coded \(L^{-1}\) hard to write | spectral substrate (the matrix solve handles arbitrary linear \(L\) by construction) |
| Both — symbolic structure *and* numerical speed | run both substrates on the same `HamProblem`-shaped inputs; the library's whole architecture is built so they agree where comparable |

The two substrates aren't competitors; they're complementary lenses
on the same HAM math. The point of the parametrisation is that
choosing one doesn't lock you out of the other.

## Algebraic dividend: laws hold uniformly

Because the substrate-generic core verifies its laws as Hypothesis
property tests parameterised over a list of `Backend` fixtures, the
*same algebraic identities* are pinned across all three backend
choices the library ships:

- `Series` ring laws — additive associativity, commutativity,
  identity, inverse; Cauchy product distributivity, associativity,
  commutativity, multiplicative identity.
- The q-calculus mutual-inverse law:
  \(\partial_q \circ \int_q = \mathrm{id}\) and
  \(\int_q \circ \partial_q = \mathrm{id}\) modulo the constant of
  integration.
- The x-calculus mutual-inverse law:
  \(\partial_x \circ \int_x = \mathrm{id}\) on substrate-resolved
  polynomials.
- Linearity of \(L\) under the substrate-appropriate equality
  comparator.
- Liao's commuting diagram for \(N\):
  \([q^k]\, N.\text{apply\_series}(\varphi)
  = [q^k]\, N.\text{apply\_scalar}\!\left(\sum_j u_j(x)\, q^j\right)\)
  on the sympy substrate, with the spectral analogue using the
  order-0 `Series` trick.

Adding a new substrate means adding a Backend fixture; the law
checks run against it automatically. This is the Algebra-Driven-
Design dividend the [tenets](../design/tenets.md) trade in for:
the architecture *encourages* extension by making the cost of new
substrates linear in their own code and zero in the library's.

## Where to go next

- [Backend API reference](../api/backend.md) — the Protocol
  itself, plus the `SympyBackend` factory.
- [`ham.grids`](../api/grids.md) and
  [`ham.spectral`](../api/spectral.md) — the spectral substrate's
  building blocks: `Grid` Protocol, `ChebGLGrid`, `SpectralBackend`,
  `spectral_linear_operator`, `spectral_inverter`.
- [Stage history](../design/stages.md#substrate-parametrisation-arc-s0-s9-2026-05-1317)
  — the S0-S9 arc that landed the substrate parametrisation,
  including the four design decisions (D-1 through D-4) that
  shaped what the substrate API actually looks like.
- [Tutorial: Step 10](../tutorial.md#step-10-same-problem-spectral-substrate-sham)
  — the spectral substrate walked through end-to-end on the
  logistic equation.
