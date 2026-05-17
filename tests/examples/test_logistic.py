"""Regression tests for the logistic-equation example.

Pins HAM output against the closed-form sigmoid Taylor expansion and
the example's per-problem validity gate. Exercises the non-zero
initial guess (`u_0 = 1/2`) code path that the earlier worked examples
do not cover.
"""

from itertools import pairwise

import sympy as sp
from examples.logistic import (
    HBAR,
    ORIGINAL_BCS,
    T,
    build_problem,
    is_convergent,
    solve_to,
    taylor_reference,
)
from ham.contracts import verify_initial_guess
from ham.diagnostics import optimal_hbar, residual_l2_squared
from ham.solver import HamSolution


def test_initial_guess_satisfies_original_bcs() -> None:
    """u_0 = 1/2 satisfies u(0) = 1/2; verify_initial_guess accepts it.

    Exercises the /non-homogeneous/ original-BC path: the deformation
    BC declared on L is u(0) = 0, but the original problem demands
    u(0) = 1/2, which u_0 satisfies by construction.
    """
    verify_initial_guess(build_problem(), ORIGINAL_BCS)


def test_u0_matches_initial_condition() -> None:
    """The initial guess sits at u(0) = 1/2 — the original BC."""
    sol = solve_to(0)
    assert sp.expand(sol.phi.coeff(0) - sp.Rational(1, 2)) == 0


def test_partial_sum_matches_sigmoid_taylor_at_order_5() -> None:
    """At hbar = -1, M = 5: HAM partial sum equals the sigmoid Taylor truncation."""
    sol = solve_to(5)
    actual = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = taylor_reference(5)
    assert sp.expand(actual - expected) == 0


def test_partial_sum_matches_sigmoid_taylor_at_order_7() -> None:
    """Same check at order 7."""
    sol = solve_to(7)
    actual = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = taylor_reference(7)
    assert sp.expand(actual - expected) == 0


def test_partial_sum_at_hbar_neg_one_has_no_even_corrections_above_t_zero() -> None:
    """The sigmoid Taylor has constant 1/2 then only odd-power corrections.

    Each even-power coefficient `t^{2k}` for `k >= 1` must therefore
    vanish in the HAM partial sum at hbar = -1.
    """
    sol = solve_to(7)
    partial = sol.evaluate_at_hbar(sp.Integer(-1))
    poly = sp.Poly(partial, T)
    for k in range(2, 8, 2):
        assert poly.nth(k) == 0


def test_residual_l2_decreases_in_m_at_hbar_neg_one() -> None:
    """As M grows the L2 residual norm on [0, 1] must not increase."""
    interval = (sp.Integer(0), sp.Integer(1))
    norms = [residual_l2_squared(solve_to(m), sp.Integer(-1), interval) for m in range(1, 7)]
    for prev, nxt in pairwise(norms):
        assert sp.simplify(nxt - prev) <= 0


def test_optimal_hbar_grid_search_selects_minus_one() -> None:
    """For the logistic problem on [0,1] at M=7, hbar=-1 dominates the grid.

    Unlike the quadratic-drag example, here the L² advantage of plain
    Taylor truncation is clean: at hbar=-1 the norm² is ~1.9e-9, an
    order of magnitude smaller than at hbar=-1/2 (~8.8e-6) and three
    orders below hbar=-3/2 (~3.1e-4).
    """
    sol = solve_to(7)
    interval = (sp.Integer(0), sp.Integer(1))

    def norm(s: HamSolution[sp.Expr], h: sp.Expr) -> sp.Expr:
        return residual_l2_squared(s, h, interval)

    grid = [sp.Rational(-3, 2), sp.Integer(-1), sp.Rational(-1, 2), sp.Integer(0)]
    assert optimal_hbar(sol, grid, norm) == sp.Integer(-1)


def test_validity_gate_passes_at_hbar_neg_one() -> None:
    """At hbar = -1 the L² residual on [0,1] is well below the threshold."""
    sol = solve_to(5)
    assert is_convergent(sol, sp.Integer(-1)) is True


def test_validity_gate_fails_at_hbar_zero() -> None:
    """At hbar = 0 the partial sum is u_0 = 1/2; N[1/2] = -1/4 (constant)."""
    sol = solve_to(5)
    assert is_convergent(sol, sp.Integer(0)) is False


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1() -> None:
    """The library convention that u_k for k>=1 carries hbar must hold here too."""
    sol = solve_to(3)
    for k in range(1, sol.order + 1):
        assert HBAR in sol.phi.coeff(k).free_symbols


def test_u0_carries_no_hbar() -> None:
    """The non-zero initial guess u_0 = 1/2 is hbar-free."""
    sol = solve_to(3)
    assert HBAR not in sol.phi.coeff(0).free_symbols


# --- Spectral substrate (SHAM) -------------------------------------------


def test_spectral_partial_sum_matches_sigmoid_on_grid_at_order_7() -> None:
    """SHAM at ℏ = -1, M = 7 on a 20-node ChebGL grid matches the sigmoid.

    The sigmoid's Taylor coefficients shrink fast (factors 1/4, 1/48,
    1/480, 17/80640, ...) so the truncation error on [0, 1] is small;
    the next nonzero Taylor term beyond t^7 is roughly 6.6e-3 at t=1.
    """
    import numpy as np
    from examples.logistic import solve_to_spectral
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    sol = solve_to_spectral(7, grid=grid)
    exact = 1.0 / (1.0 + np.exp(-grid.nodes))
    err = float(np.max(np.abs(sol.partial_sum() - exact)))
    assert err < 1e-2


def test_spectral_partial_sum_converges_with_order() -> None:
    """L∞ error against the sigmoid on the grid is monotone-decreasing in M."""
    import numpy as np
    from examples.logistic import solve_to_spectral
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    exact = 1.0 / (1.0 + np.exp(-grid.nodes))
    errors = [
        float(np.max(np.abs(solve_to_spectral(m, grid=grid).partial_sum() - exact)))
        for m in (1, 3, 5, 7)
    ]
    for prev, nxt in pairwise(errors):
        assert nxt < prev


def test_spectral_residual_l2_squared_below_validity_threshold() -> None:
    """Clenshaw-Curtis L² norm² at ℏ = -1, M = 7 is below the sympy gate's threshold."""
    from examples.logistic import _DEFAULT_THRESHOLD, solve_to_spectral
    from ham.diagnostics import residual_l2_squared
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    sol = solve_to_spectral(7, grid=grid)
    l2_sq = float(residual_l2_squared(sol, None, grid=grid))
    # Mirror the sympy validity gate's threshold = 1/100; the spectral
    # quadrature should give a value well below threshold² at M=7.
    assert l2_sq < float(_DEFAULT_THRESHOLD) ** 2


def test_spectral_hbar_curve_at_sympy_scalar_returns_polynomial_in_hbar() -> None:
    """sympy-scalar SHAM keeps ℏ symbolic; the curve at a grid node is a polynomial.

    Substituting ℏ → -1 in that polynomial should give a numeric value
    close to the sigmoid at the chosen node — within the HAM truncation
    error at the test's order.
    """
    import numpy as np
    from examples.logistic import solve_to_spectral
    from ham.diagnostics import hbar_curve_at
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=6, domain=(0.0, 1.0))
    sol = solve_to_spectral(3, grid=grid, scalar="sympy", hbar_value=HBAR)
    curve = hbar_curve_at(sol, sp.Float(0.5), grid=grid)
    assert HBAR in curve.free_symbols

    value = float(curve.subs(HBAR, sp.Float(-1.0)))
    expected_sigmoid_at_half = 1.0 / (1.0 + np.exp(-0.5))
    # Order-3 Taylor at t=0.5: error bounded by t^5/480 = 0.5^5/480 ≈ 6.5e-5.
    assert abs(value - expected_sigmoid_at_half) < 1e-2
