"""Convergence diagnostics for HAM partial sums (Stage 6).

Liao's Theorem 2.1 only guarantees correctness *given* convergence, so
the diagnostics here are what justify reporting `u^{(M)}(x)` as a
solution. Three independent observables flow from a HamSolution[sp.Expr]:

  - `residual`:        N applied to the partial sum.
  - `residual_*`:      L2 and discrete norms of the residual (6b).
  - `hbar_curve_at`:   the partial sum specialised to x = x_star, as
                       a sympy expression in hbar — the "ℏ-curve"
                       in Liao's terminology (6c).

This module is the functional core. No plotting, no I/O. Callers pull
data and render it externally.

Stage 12 added multi-parameter substitution support: every residual and
curve function accepts a `substitutions` dict-keyword in addition to
the legacy `hbar_value` shortcut, and a generalised
`optimal_parameters` grid search complements `optimal_hbar` for HAM
problems with more than one free parameter (e.g. Blasius via
exponential basis carries both ℏ and the basis decay rate alpha).
"""

from collections.abc import Callable, Mapping, Sequence

import sympy as sp

from ham.solver import HamSolution


def _resolve_substitutions(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr | None,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None,
) -> Mapping[sp.Symbol, sp.Expr] | None:
    """Normalise the two-shortcut API into a single substitution mapping.

    The legacy `hbar_value=v` is sugar for `substitutions={problem.hbar: v}`;
    accepting both at once is ambiguous and rejected.
    """
    if hbar_value is not None and substitutions is not None:
        raise ValueError(
            "Pass either `hbar_value` or `substitutions`, not both. "
            "`hbar_value=v` is a shortcut for `substitutions={problem.hbar: v}`."
        )
    if hbar_value is not None:
        return {solution.problem.hbar: hbar_value}
    return substitutions


def _apply_substitutions(
    expr: sp.Expr, substitutions: Mapping[sp.Symbol, sp.Expr] | None
) -> sp.Expr:
    """Apply a substitution mapping to a sympy expression, or pass through if None."""
    if substitutions is None:
        return expr
    return expr.subs(substitutions)


def residual(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr | None = None,
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> sp.Expr:
    """Apply N to the partial sum: N[u^{(M)}(x)].

    Returns a sympy Expr. With no substitution arguments, every symbol
    (ℏ and any other parameters introduced by the problem setup) stays
    symbolic so the caller can post-process (substitute, plot, minimise).

    `hbar_value=v` is the legacy single-parameter shortcut equivalent to
    `substitutions={problem.hbar: v}`; supply at most one of them.

    `substitutions` is a mapping from sympy symbols to concrete values,
    applied to the partial sum before N is invoked. Use this when the
    problem carries free parameters beyond ℏ (e.g. the basis decay rate
    alpha in the exponential-basis Blasius example).

    A residual that vanishes identically means u^{(M)} satisfies the
    original problem exactly at the given parameter values; in practice
    the residual measures how far the truncated series is from a
    solution.
    """
    subs = _resolve_substitutions(solution, hbar_value, substitutions)
    partial = _apply_substitutions(solution.partial_sum(), subs)
    return sp.expand(solution.problem.N.apply_scalar(partial))


def residual_l2_squared(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr | None,
    interval: tuple[sp.Expr, sp.Expr],
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> sp.Expr:
    """L² norm squared of the residual: integral of N[u^{(M)}]^2 over [a, b].

    Returns `∫_a^b (N[u^{(M)}(x)])^2 dx` as a sympy expression. With no
    substitution arguments the result retains every free symbol, which
    is what the optimal-parameter grid searches consume.

    See `residual` for the relationship between `hbar_value` and
    `substitutions`.
    """
    var = solution.problem.L.var
    a, b = interval
    r = residual(solution, hbar_value, substitutions=substitutions)
    return sp.integrate(r**2, (var, a, b))


def residual_discrete_sum_of_squares(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr | None,
    samples: Sequence[sp.Expr],
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> sp.Expr:
    """Discrete L² norm squared of the residual: `Σ_i N[u^{(M)}(x_i)]^2`.

    Returns the sum of squares evaluated at user-supplied sample points.
    Cheaper and more robust than the L² integral when the residual is
    transcendental or the domain is unbounded.

    See `residual` for the relationship between `hbar_value` and
    `substitutions`.
    """
    var = solution.problem.L.var
    r = residual(solution, hbar_value, substitutions=substitutions)
    total: sp.Expr = sp.Integer(0)
    for sample in samples:
        total = total + r.subs(var, sample) ** 2
    return sp.expand(total)


def hbar_curve_at(
    solution: HamSolution[sp.Expr],
    x_star: sp.Expr,
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> sp.Expr:
    """The ℏ-curve: partial sum evaluated at x = x_star, as a polynomial in ℏ.

    At fixed working order M and fixed x = x_star, the partial sum is a
    polynomial in ℏ of degree at most M. The graph of this polynomial
    is Liao's "ℏ-curve"; a plateau (where the curve is nearly horizontal)
    indicates a candidate convergence region in ℏ.

    Supply `substitutions` to pin any auxiliary parameters (e.g. alpha in
    the exponential-basis Blasius problem) while keeping ℏ symbolic.

    No plotting here. Callers render the polynomial externally, or pass
    it to `optimal_hbar` / `optimal_parameters` via a closure.
    """
    var = solution.problem.L.var
    curve = solution.partial_sum().subs(var, x_star)
    return sp.expand(_apply_substitutions(curve, substitutions))


def optimal_hbar(
    solution: HamSolution[sp.Expr],
    hbar_grid: Sequence[sp.Expr],
    norm_fn: Callable[[HamSolution[sp.Expr], sp.Expr], sp.Expr],
) -> sp.Expr:
    """Grid search: return the value in `hbar_grid` minimising `norm_fn(solution, h)`.

    `norm_fn` is a caller-supplied function (HamSolution[sp.Expr], hbar) → Expr,
    typically built by binding the interval/samples of a residual norm.
    Grid values should be concrete sympy numbers (Integer, Rational,
    Float) so the norm at each grid point evaluates to a real scalar.

    On ties, returns the first element of `hbar_grid` reaching the minimum
    (Python's `min` is stable in that sense). Raises `ValueError` for an
    empty grid — there is no well-defined minimum over no candidates.

    For multi-parameter problems, use `optimal_parameters` instead.
    """
    if not hbar_grid:
        raise ValueError("optimal_hbar requires a non-empty hbar_grid.")
    return min(hbar_grid, key=lambda h: float(norm_fn(solution, h)))


def optimal_parameters(
    solution: HamSolution[sp.Expr],
    parameter_grid: Sequence[Mapping[sp.Symbol, sp.Expr]],
    norm_fn: Callable[[HamSolution[sp.Expr], Mapping[sp.Symbol, sp.Expr]], sp.Expr],
) -> Mapping[sp.Symbol, sp.Expr]:
    """Multi-parameter grid search: return the substitution dict minimising the norm.

    Generalises `optimal_hbar` to HAM problems with more than one free
    parameter. Each grid point is a mapping from sympy symbols to
    concrete values (Integer/Rational/Float); `norm_fn` receives the
    solution and the per-point substitution mapping and returns a
    scalar (sympy Expr coercible to float).

    Typical use (Blasius exponential basis, two parameters):

        def norm(sol, subs):
            return residual_l2_squared(
                sol, None, (0, sp.oo), substitutions=subs,
            )

        grid = [
            {hbar: sp.Rational(h, 10), alpha: sp.Rational(a, 10)}
            for h in range(-15, 0)
            for a in range(5, 15)
        ]
        best = optimal_parameters(solution, grid, norm)

    On ties, returns the first grid entry reaching the minimum. Raises
    `ValueError` for an empty grid.
    """
    if not parameter_grid:
        raise ValueError("optimal_parameters requires a non-empty parameter_grid.")
    return min(parameter_grid, key=lambda subs: float(norm_fn(solution, subs)))
