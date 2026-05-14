"""Regression tests for the Volterra integro-differential example.

Pins HAM output against the Taylor expansion derived directly from
the integro-differential equation by the recurrence in
`examples.volterra.taylor_reference`. Exercises the Stage 9a Integral
branch of `NonlinearOperator._compile` end-to-end.
"""

import sympy as sp
from examples.volterra import (
    ALPHA,
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
    """u_0 = alpha = 1/10 satisfies u(0) = alpha; verify_initial_guess accepts it."""
    verify_initial_guess(build_problem(), ORIGINAL_BCS)


def test_u0_matches_initial_population() -> None:
    """The initial guess sits at u(0) = alpha — the original BC."""
    sol = solve_to(0)
    assert sp.expand(sol.phi.coeff(0) - ALPHA) == 0


def test_taylor_reference_low_coefficients_match_hand_derivation() -> None:
    """For alpha = 1/10, κ = 1: a_1 = 9/100, a_2 = 31/1000 by direct recurrence."""
    taylor = taylor_reference(3)
    poly = sp.Poly(taylor, T)
    assert poly.nth(0) == sp.Rational(1, 10)
    assert poly.nth(1) == sp.Rational(9, 100)
    assert poly.nth(2) == sp.Rational(31, 1000)
    assert poly.nth(3) == sp.Rational(2, 1875)


def test_partial_sum_coefficients_match_taylor_up_to_order_m() -> None:
    """At ℏ = -1, HAM at order M matches Taylor at order M for the first M+1 coefficients.

    The integro-differential character then introduces "extra" higher-
    degree terms (HAM polynomial degree = 2M for this problem) that do
    NOT match higher-order Taylor coefficients. Only the first M+1
    coefficients are pinned by this cross-check.
    """
    for m in range(1, 6):
        sol = solve_to(m)
        ham = sol.evaluate_at_hbar(sp.Integer(-1))
        taylor = taylor_reference(m)
        ham_poly = sp.Poly(ham, T)
        taylor_poly = sp.Poly(taylor, T)
        for k in range(m + 1):
            assert sp.simplify(ham_poly.nth(k) - taylor_poly.nth(k)) == 0, (m, k)


def test_ham_polynomial_degree_is_twice_working_order() -> None:
    """Volterra HAM grows polynomial degree by 2 per step (L^{-1} + integral).

    At order M, the partial sum is a polynomial of degree 2M in t.
    This is the integro-differential signature; purely-polynomial-in-u
    problems grow degree by 1 per step.
    """
    for m in range(1, 6):
        sol = solve_to(m)
        ham = sol.evaluate_at_hbar(sp.Integer(-1))
        poly = sp.Poly(ham, T)
        assert poly.degree() == 2 * m, (m, poly.degree())


def test_residual_l2_overall_decrease_with_m_at_hbar_neg_one() -> None:
    """The L² residual norm at M=6 is orders of magnitude below the norm at M=1.

    Unlike the polynomial-N worked examples, Volterra's residual is *not*
    strictly monotone in M — at M=3 there is a small uptick versus M=2 of
    about +1e-6, after which the norm continues decreasing to ~3e-9 at
    M=6. Liao's convergence theorem only guarantees convergence in the
    limit, not strict monotonicity step-by-step; the test pins the
    overall trend (M=6 is at least an order of magnitude smaller than
    M=1) rather than asserting strict monotone decrease.
    """
    interval = (sp.Integer(0), sp.Integer(1))
    norm_m1 = float(residual_l2_squared(solve_to(1), sp.Integer(-1), interval))
    norm_m6 = float(residual_l2_squared(solve_to(6), sp.Integer(-1), interval))
    assert norm_m6 < norm_m1 / 100


def test_optimal_hbar_grid_search_selects_minus_one() -> None:
    """For the Volterra problem on [0,1] at M=5, ℏ=-1 dominates the grid."""
    sol = solve_to(5)
    interval = (sp.Integer(0), sp.Integer(1))

    def norm(s: HamSolution, h: sp.Expr) -> sp.Expr:
        return residual_l2_squared(s, h, interval)

    grid = [sp.Rational(-3, 2), sp.Integer(-1), sp.Rational(-1, 2), sp.Integer(0)]
    assert optimal_hbar(sol, grid, norm) == sp.Integer(-1)


def test_validity_gate_passes_at_hbar_neg_one() -> None:
    """At ℏ = -1, M=5, the L² residual on [0,1] is well below the threshold."""
    sol = solve_to(5)
    assert is_convergent(sol, sp.Integer(-1)) is True


def test_validity_gate_fails_at_hbar_zero() -> None:
    """At ℏ = 0 the partial sum is u_0 = alpha = 1/10; residual is the constant N[1/10]."""
    sol = solve_to(5)
    assert is_convergent(sol, sp.Integer(0)) is False


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1() -> None:
    """u_k for k >= 1 carries the ℏ symbol; the substitute-late convention holds."""
    sol = solve_to(3)
    for k in range(1, sol.order + 1):
        assert HBAR in sol.phi.coeff(k).free_symbols


def test_u0_carries_no_hbar() -> None:
    """The constant initial guess u_0 = alpha is ℏ-free."""
    sol = solve_to(3)
    assert HBAR not in sol.phi.coeff(0).free_symbols
