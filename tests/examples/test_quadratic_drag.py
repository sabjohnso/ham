"""Regression tests for the quadratic-drag projectile example.

Pins HAM output against the closed-form `tanh(t)` reference and the
example's per-problem validity gate.
"""

from itertools import pairwise

import sympy as sp
from examples.quadratic_drag import (
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
    """u_0 = 0 satisfies v(0) = 0; verify_initial_guess accepts it.

    Documentary check that ORIGINAL_BCS is consistent with build_problem().
    """
    verify_initial_guess(build_problem(), ORIGINAL_BCS)


def test_partial_sum_matches_tanh_taylor_at_order_5() -> None:
    """At hbar = -1, M = 5: HAM partial sum equals the tanh(t) Taylor truncation."""
    sol = solve_to(5)
    actual = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = taylor_reference(5)
    assert sp.expand(actual - expected) == 0


def test_partial_sum_matches_tanh_taylor_at_order_7() -> None:
    """Same check at order 7 to catch regressions in deeper terms."""
    sol = solve_to(7)
    actual = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = taylor_reference(7)
    assert sp.expand(actual - expected) == 0


def test_partial_sum_at_hbar_neg_one_has_only_odd_powers() -> None:
    """tanh is odd; the HAM partial sum at hbar=-1 must contain no even powers."""
    sol = solve_to(7)
    partial = sol.evaluate_at_hbar(sp.Integer(-1))
    poly = sp.Poly(partial, T)
    for k in range(0, 8, 2):
        assert poly.nth(k) == 0


def test_residual_l2_decreases_in_m_at_hbar_neg_one() -> None:
    """As M grows the L2 residual norm on [0, 1] must not increase."""
    interval = (sp.Integer(0), sp.Integer(1))
    norms = [residual_l2_squared(solve_to(m), sp.Integer(-1), interval) for m in range(1, 7)]
    for prev, nxt in pairwise(norms):
        assert sp.simplify(nxt - prev) <= 0


def test_optimal_hbar_returns_grid_minimum() -> None:
    """optimal_hbar must return the grid value with the smallest L² residual norm.

    Note: the winning ℏ shifts with M for this problem — at M=3 and M=7
    HAM at ℏ=-1/2 slightly beats plain Taylor truncation (ℏ=-1) in the
    L²-norm sense, while at M=5 they nearly tie. That non-uniformity is
    the HAM adaptive-ℏ advantage at work. The test stays behavioural:
    whichever grid value wins, it must actually minimise the norm, and
    the obviously-bad endpoints (ℏ = -3/2, ℏ = 0) must never be picked.
    """
    sol = solve_to(5)
    interval = (sp.Integer(0), sp.Integer(1))

    def norm(s: HamSolution, h: sp.Expr) -> sp.Expr:
        return residual_l2_squared(s, h, interval)

    grid = [sp.Rational(-3, 2), sp.Integer(-1), sp.Rational(-1, 2), sp.Integer(0)]
    best = optimal_hbar(sol, grid, norm)
    best_norm = norm(sol, best)
    for h in grid:
        assert sp.simplify(best_norm - norm(sol, h)) <= 0
    assert best not in (sp.Rational(-3, 2), sp.Integer(0))


def test_validity_gate_passes_at_hbar_neg_one_high_order() -> None:
    """At hbar=-1 and M=7 the L2 residual is well below the example's threshold."""
    sol = solve_to(7)
    assert is_convergent(sol, sp.Integer(-1)) is True


def test_validity_gate_fails_at_hbar_zero() -> None:
    """At hbar=0 the partial sum is u_0 = 0 and the residual is -1 (constant)."""
    sol = solve_to(5)
    assert is_convergent(sol, sp.Integer(0)) is False


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1() -> None:
    """The example respects the library convention that u_k for k>=1 carries hbar."""
    sol = solve_to(3)
    for k in range(1, sol.order + 1):
        assert HBAR in sol.phi.coeff(k).free_symbols
