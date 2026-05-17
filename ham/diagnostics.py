"""Convergence diagnostics for HAM partial sums (Stage 6 + S8).

Liao's Theorem 2.1 only guarantees correctness *given* convergence, so
the diagnostics here are what justify reporting `u^{(M)}` as a
solution. Substrate-aware after S8:

  - `residual`:        N applied to the partial sum. Returns
                       `sp.Expr` for sympy / spectral-sympy mixed
                       output, or `np.ndarray` for spectral float.
  - `residual_l2_squared`: L² norm squared of the residual. With
                       `interval` only — sympy `sp.integrate`. With
                       `grid` — Clenshaw-Curtis quadrature against
                       `grid.quadrature_weights`. The two share the
                       caller-facing signature so the same diagnostic
                       call site works on either substrate.
  - `hbar_curve_at`:   the partial sum specialised to x = x_star.
                       Sympy substrate: substitute `var = x_star`.
                       Spectral substrate (`grid` supplied): pick the
                       grid node nearest `x_star` and return the
                       entry of the partial sum there (a polynomial
                       in ℏ on the sympy-scalar path; a numeric float
                       on the float path).
  - `hbar_curve_at_sweep`: spectral-float-only ℏ-curve. ℏ on the
                       float path is a numeric problem-construction
                       parameter, so the curve is built by re-running
                       the solver per ℏ value (embarrassingly
                       parallel; the explicit sweep makes the cost
                       visible at the call site).

This module is the functional core. No plotting, no I/O. Callers pull
data and render it externally.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ham.backend import Backend
from ham.deformation import HamProblem
from ham.grids import Grid
from ham.series import Series, SupportsCoefficientArith
from ham.solver import HamSolution, solve


def _resolve_substitutions[C: SupportsCoefficientArith](
    solution: HamSolution[C],
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


def _is_spectral[C: SupportsCoefficientArith](solution: HamSolution[C]) -> bool:
    """True if the solution's coefficients are ndarrays (spectral substrate)."""
    return isinstance(solution.phi.backend.zero(), np.ndarray)


def residual[C: SupportsCoefficientArith](
    solution: HamSolution[C],
    hbar_value: sp.Expr | None = None,
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> C:
    """Apply N to the partial sum: N[u^{(M)}].

    Substrate-aware. For the sympy backend (`C = sp.Expr`) this is
    `N.apply_scalar(partial)` — the legacy path, unchanged from pre-S8.
    For the spectral backend (`C = np.ndarray`) this builds an order-0
    `Series` with the partial sum as its sole coefficient and reads
    `[q^0] N[phi]`; this routes `N`'s tree-walker through the
    spectral substrate's `diff_x` / `lift_xonly` so the result lives
    on the grid.

    `hbar_value=v` / `substitutions` apply before N is invoked on the
    sympy path. On the spectral sympy-scalar path they apply
    element-wise to the resulting object array. On the spectral float
    path they're rejected — ℏ on the float path is pre-substituted at
    HamProblem construction and no symbols survive to be substituted.

    A residual that vanishes identically means u^{(M)} satisfies the
    original problem exactly at the given parameter values; in practice
    the residual measures how far the truncated series is from a
    solution.
    """
    subs = _resolve_substitutions(solution, hbar_value, substitutions)
    if _is_spectral(solution):
        return _spectral_residual(solution, subs)
    return cast("C", _sympy_residual(solution, subs))


def _sympy_residual(
    solution: HamSolution[sp.Expr],
    substitutions: Mapping[sp.Symbol, sp.Expr] | None,
) -> sp.Expr:
    partial = _apply_substitutions(solution.partial_sum(), substitutions)
    return sp.expand(solution.problem.N.apply_scalar(partial))


def _spectral_residual[C: SupportsCoefficientArith](
    solution: HamSolution[C],
    substitutions: Mapping[sp.Symbol, sp.Expr] | None,
) -> C:
    backend = solution.phi.backend
    partial = solution.partial_sum()
    phi_at_partial: Series[C] = Series([partial], order=0, backend=backend)
    result = solution.problem.N.apply_series(phi_at_partial).coeff(0)
    result = backend.normalize(result)
    if substitutions is None:
        return result
    result_arr: NDArray[Any] = result  # type: ignore[assignment]
    if result_arr.dtype == object:
        return np.array(  # type: ignore[return-value]
            [entry.subs(substitutions) for entry in result_arr], dtype=object
        )
    raise ValueError(
        "Cannot apply `substitutions` / `hbar_value` to a float-dtype spectral "
        "residual — ℏ should be pre-substituted at HamProblem construction for "
        "the spectral float backend, leaving no symbols to substitute here."
    )


def residual_l2_squared[C: SupportsCoefficientArith](
    solution: HamSolution[C],
    hbar_value: sp.Expr | None,
    interval: tuple[sp.Expr, sp.Expr] | None = None,
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
    grid: Grid | None = None,
) -> sp.Expr:
    """L² norm squared of the residual.

    Sympy substrate (`interval` supplied, `grid=None`): returns
    `∫_a^b N[u^{(M)}(x)]^2 dx` as a sympy expression via
    `sp.integrate`. Existing pre-S8 behaviour.

    Spectral substrate (`grid` supplied, `interval` ignored): returns
    `Σ_j w_j · N[u^{(M)}](x_j)^2` using the grid's Clenshaw-Curtis
    weights. For the float backend this is a numeric scalar (boxed
    as `sp.Float` via `sp.Integer(0) + np.float64` coercion); for the
    sympy-scalar backend it is a polynomial in ℏ.

    Both substrates share the caller-facing signature; the diagnostic
    site dispatches on which one is set.

    See `residual` for the relationship between `hbar_value` and
    `substitutions`.
    """
    subs = _resolve_substitutions(solution, hbar_value, substitutions)
    r = residual(solution, substitutions=subs)
    if grid is not None:
        r_arr: NDArray[Any] = r  # type: ignore[assignment]
        weighted = grid.quadrature_weights * (r_arr * r_arr)
        return sp.sympify(np.sum(weighted))
    if interval is None:
        raise ValueError(
            "residual_l2_squared requires either `interval` (sympy substrate, "
            "for `sp.integrate`) or `grid` (spectral substrate, for "
            "Clenshaw-Curtis quadrature)."
        )
    var = solution.problem.L.var
    a, b = interval
    r_sym = cast("sp.Expr", r)
    return sp.integrate(r_sym**2, (var, a, b))


def residual_discrete_sum_of_squares(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr | None,
    samples: Sequence[sp.Expr],
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
) -> sp.Expr:
    """Discrete L² norm squared of the residual: `Σ_i N[u^{(M)}(x_i)]^2`.

    Sympy-only at this stage. Spectral analogue is the grid-based path
    in `residual_l2_squared(grid=...)` — the spectral substrate already
    samples N on the grid, so a separate sample-points API isn't needed.

    See `residual` for the relationship between `hbar_value` and
    `substitutions`.
    """
    var = solution.problem.L.var
    r = residual(solution, hbar_value, substitutions=substitutions)
    total: sp.Expr = sp.Integer(0)
    for sample in samples:
        total = total + r.subs(var, sample) ** 2
    return sp.expand(total)


def hbar_curve_at[C: SupportsCoefficientArith](
    solution: HamSolution[C],
    x_star: sp.Expr,
    *,
    substitutions: Mapping[sp.Symbol, sp.Expr] | None = None,
    grid: Grid | None = None,
) -> sp.Expr:
    """The ℏ-curve: partial sum evaluated at `x = x_star`.

    Sympy substrate: substitute `var = x_star` in the partial sum.
    The result is a polynomial in ℏ (at fixed working order M and
    fixed x = x_star, the partial sum is a polynomial in ℏ of degree
    at most M). The plateau region in this polynomial is Liao's
    "ℏ-curve" indicating a candidate convergence interval.

    Spectral substrate (`grid` supplied): find the grid node nearest
    `x_star` and return the partial-sum entry there. For the
    sympy-scalar spectral backend this is a polynomial in ℏ (same
    interpretation as the sympy substrate). For the float spectral
    backend it is a numeric value — ℏ was pre-substituted at problem
    construction, so the "curve" has degenerated to a single point;
    use `hbar_curve_at_sweep` to build a curve by re-running the
    solver across ℏ values.

    Supply `substitutions` to pin any auxiliary parameters (e.g.
    alpha in the exponential-basis Blasius problem) while keeping ℏ
    symbolic.
    """
    if grid is not None:
        idx = int(np.argmin(np.abs(grid.nodes - float(x_star))))
        partial_arr = cast("NDArray[Any]", solution.partial_sum())
        entry = partial_arr[idx]
        if substitutions is not None and hasattr(entry, "subs"):
            entry = entry.subs(substitutions)
        return sp.sympify(entry)
    partial_expr = cast("sp.Expr", solution.partial_sum())
    var = solution.problem.L.var
    curve = partial_expr.subs(var, x_star)
    return sp.expand(_apply_substitutions(curve, substitutions))


def hbar_curve_at_sweep(
    problem_factory: Callable[[sp.Expr], HamProblem[NDArray[Any]]],
    x_star: sp.Expr,
    hbar_grid: Sequence[sp.Expr],
    order: int,
    grid: Grid,
    backend: Backend[NDArray[Any]],
) -> list[tuple[sp.Expr, float]]:
    """Spectral-float ℏ-curve: re-run the solver per ℏ; return (ℏ, value) pairs.

    On the spectral float backend, ℏ is pre-substituted at
    `HamProblem(...)` construction and the solver sees a fully numeric
    problem. The ℏ-curve at a point therefore is not a polynomial in ℏ
    that we can extract from one solve; it's the sequence of numeric
    values produced by solving at each ℏ in `hbar_grid` and reading
    the partial sum at the grid node closest to `x_star`.

    `problem_factory(ℏ_value)` constructs the HamProblem with that ℏ
    pre-substituted. The function is the caller's hook for varying ℏ
    while keeping everything else fixed.

    The sweep is embarrassingly parallel and the cost (one full solve
    per ℏ) is significant; making the user pass `problem_factory`
    explicitly keeps that cost visible at the call site rather than
    hidden behind a `.subs`-shaped abstraction that wouldn't apply
    here.
    """
    idx = int(np.argmin(np.abs(grid.nodes - float(x_star))))
    pairs: list[tuple[sp.Expr, float]] = []
    for hbar_value in hbar_grid:
        problem = problem_factory(hbar_value)
        solution = solve(problem, order=order, backend=backend)
        partial = solution.partial_sum()
        pairs.append((hbar_value, float(partial[idx])))
    return pairs


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
    For the spectral float backend (where ℏ is set at problem
    construction), use `hbar_curve_at_sweep` followed by `min` on the
    returned pairs.
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
