"""Convergence diagnostics for HAM partial sums (Stage 6).

Liao's Theorem 2.1 only guarantees correctness *given* convergence, so
the diagnostics here are what justify reporting `u^{(M)}(x)` as a
solution. Three independent observables flow from a HamSolution:

  - `residual`:        N applied to the partial sum.
  - `residual_*`:      L2 and discrete norms of the residual (6b).
  - `hbar_curve_at`:   the partial sum specialised to x = x_star, as
                       a sympy expression in hbar — the "ℏ-curve"
                       in Liao's terminology (6c).

This module is the functional core. No plotting, no I/O. Callers pull
data and render it externally.
"""

from collections.abc import Callable, Sequence

import sympy as sp

from ham.solver import HamSolution


def residual(solution: HamSolution, hbar_value: sp.Expr | None = None) -> sp.Expr:
    """Apply N to the partial sum: N[u^{(M)}(x)].

    Returns a sympy Expr. When `hbar_value` is None, ℏ stays symbolic so
    the caller can post-process (substitute, plot, minimise). When
    `hbar_value` is supplied, the problem's ℏ symbol is substituted in
    the partial sum before N is applied.

    A residual that vanishes identically means u^{(M)} satisfies the
    original problem exactly at the given ℏ; in practice the residual
    measures how far the truncated series is from a solution.
    """
    if hbar_value is None:
        partial = solution.partial_sum()
    else:
        partial = solution.evaluate_at_hbar(hbar_value)
    return sp.expand(solution.problem.N.apply_scalar(partial))


def residual_l2_squared(
    solution: HamSolution,
    hbar_value: sp.Expr | None,
    interval: tuple[sp.Expr, sp.Expr],
) -> sp.Expr:
    """L² norm squared of the residual: integral of N[u^{(M)}]^2 over [a, b].

    Returns `∫_a^b (N[u^{(M)}(x)])^2 dx` as a sympy expression. When
    `hbar_value` is None the result retains ℏ symbolically, which is
    what the optimal-ℏ grid search consumes.

    Cheap for polynomial residuals (sympy integrates exactly); may be
    slow or fail to close for transcendental residuals.
    """
    var = solution.problem.L.var
    a, b = interval
    r = residual(solution, hbar_value)
    return sp.integrate(r**2, (var, a, b))


def residual_discrete_sum_of_squares(
    solution: HamSolution,
    hbar_value: sp.Expr | None,
    samples: Sequence[sp.Expr],
) -> sp.Expr:
    """Discrete L² norm squared of the residual: `Σ_i N[u^{(M)}(x_i)]^2`.

    Returns the sum of squares evaluated at user-supplied sample points.
    Cheaper and more robust than the L² integral when the residual is
    transcendental or the domain is unbounded. With ℏ symbolic, the
    result is a polynomial in ℏ usable by `optimal_hbar`.
    """
    var = solution.problem.L.var
    r = residual(solution, hbar_value)
    total: sp.Expr = sp.Integer(0)
    for sample in samples:
        total = total + r.subs(var, sample) ** 2
    return sp.expand(total)


def hbar_curve_at(solution: HamSolution, x_star: sp.Expr) -> sp.Expr:
    """The ℏ-curve: partial sum evaluated at x = x_star, as a polynomial in ℏ.

    At fixed working order M and fixed x = x_star, the partial sum is a
    polynomial in ℏ of degree at most M. The graph of this polynomial
    is Liao's "ℏ-curve"; a plateau (where the curve is nearly horizontal)
    indicates a candidate convergence region in ℏ.

    No plotting here. Callers render the polynomial externally, or pass
    it to `optimal_hbar` via a closure that substitutes ℏ values.
    """
    var = solution.problem.L.var
    return sp.expand(solution.partial_sum().subs(var, x_star))


def optimal_hbar(
    solution: HamSolution,
    hbar_grid: Sequence[sp.Expr],
    norm_fn: Callable[[HamSolution, sp.Expr], sp.Expr],
) -> sp.Expr:
    """Grid search: return the value in `hbar_grid` minimising `norm_fn(solution, h)`.

    `norm_fn` is a caller-supplied function (HamSolution, hbar) → Expr,
    typically built by binding the interval/samples of a residual norm.
    Grid values should be concrete sympy numbers (Integer, Rational,
    Float) so the norm at each grid point evaluates to a real scalar.

    On ties, returns the first element of `hbar_grid` reaching the minimum
    (Python's `min` is stable in that sense). Raises `ValueError` for an
    empty grid — there is no well-defined minimum over no candidates.
    """
    if not hbar_grid:
        raise ValueError("optimal_hbar requires a non-empty hbar_grid.")
    return min(hbar_grid, key=lambda h: float(norm_fn(solution, h)))
