"""Tests for the HAM convergence diagnostics (Stage 6).

Stage 6a slice: the `residual` primitive — N applied to the partial sum.
Hand-derived targets pin two reference problems:

  - exp problem (u' = u, u(0) = 1) at M = 4, hbar = -1:
        partial sum = 1 + x + x²/2 + x³/6 + x⁴/24
        N[partial] = (partial)' - partial = -x⁴/24
  - quadratic problem (u' = u², u(0) = 1) at M = 3, hbar = -1:
        partial sum = 1 + x + x² + x³
        (partial)² = 1 + 2x + 3x² + 4x³ + 3x⁴ + 2x⁵ + x⁶
        N[partial] = (partial)' - (partial)² = -4x³ - 3x⁴ - 2x⁵ - x⁶
"""

import pytest
import sympy as sp
from ham.deformation import HamProblem
from ham.diagnostics import (
    hbar_curve_at,
    optimal_hbar,
    residual,
    residual_discrete_sum_of_squares,
    residual_l2_squared,
)
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import solve

from tests.strategies import X

U = sp.Function("u")
HBAR = sp.Symbol("hbar")


def _ivp_operator() -> LinearOperator:
    """L = d/dx with homogeneous BC u(0) = 0."""
    return LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )


def _exp_problem() -> HamProblem:
    """u' = u, u(0) = 1; N[u] = u' - u."""
    return HamProblem(
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X), dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


def _quadratic_problem() -> HamProblem:
    """u' = u², u(0) = 1; N[u] = u' - u²."""
    return HamProblem(
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X) ** 2, dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


# --- residual primitive ---------------------------------------------------


def test_residual_at_order_zero_is_n_of_u0_for_exp() -> None:
    """For order=0 the partial sum is u_0; residual is N[u_0] = -1 for the exp problem."""
    problem = _exp_problem()
    sol = solve(problem, order=0)
    assert sp.expand(residual(sol, sp.Integer(-1)) - sp.Integer(-1)) == 0


def test_residual_at_order_zero_is_n_of_u0_for_quadratic() -> None:
    """For order=0 the residual is N[1] = 0 - 1 = -1 for the quadratic problem."""
    problem = _quadratic_problem()
    sol = solve(problem, order=0)
    assert sp.expand(residual(sol, sp.Integer(-1)) - sp.Integer(-1)) == 0


def test_residual_for_exp_problem_at_order_4_hbar_neg_one() -> None:
    """At hbar=-1 the exp partial sum is the Taylor truncation; residual is -x⁴/24."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    expected = -(X**4) / sp.Integer(24)
    assert sp.expand(residual(sol, sp.Integer(-1)) - expected) == 0


def test_residual_for_quadratic_problem_at_order_3_hbar_neg_one() -> None:
    """At hbar=-1 the quadratic partial sum is the geometric truncation.

    Residual matches hand-derivation: -4x^3 - 3x^4 - 2x^5 - x^6.
    """
    problem = _quadratic_problem()
    sol = solve(problem, order=3)
    expected = -4 * X**3 - 3 * X**4 - 2 * X**5 - X**6
    assert sp.expand(residual(sol, sp.Integer(-1)) - expected) == 0


def test_residual_is_symbolic_in_hbar_when_no_value_supplied() -> None:
    """residual(sol) with no hbar_value retains the hbar symbol in the result."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    r = residual(sol)
    assert HBAR in r.free_symbols


def test_residual_symbolic_then_substituted_matches_direct_call() -> None:
    """residual(sol).subs(hbar, v) equals residual(sol, v) (consistency)."""
    problem = _exp_problem()
    sol = solve(problem, order=3)
    symbolic = residual(sol)
    direct = residual(sol, sp.Integer(-1))
    assert sp.expand(symbolic.subs(HBAR, sp.Integer(-1)) - direct) == 0


def test_residual_for_exp_problem_at_hbar_neg_one_is_negative_taylor_tail() -> None:
    """For the exp problem at hbar=-1 and order M, residual is exactly -x^M / M!.

    Partial sum at hbar=-1 is the truncated Taylor series of exp(x):
    u^{(M)} = sum_{k=0..M} x^k / k!. Then
    (u^{(M)})' = sum_{k=1..M} x^{k-1}/(k-1)! = sum_{j=0..M-1} x^j/j!
              = u^{(M)} - x^M/M!
    so N[u^{(M)}] = (u^{(M)})' - u^{(M)} = -x^M / M!.
    """
    problem = _exp_problem()
    for m in range(1, 5):
        sol = solve(problem, order=m)
        expected = -(X**m) / sp.factorial(m)
        assert sp.expand(residual(sol, sp.Integer(-1)) - expected) == 0


# --- residual_l2_squared --------------------------------------------------


def test_residual_l2_squared_for_exp_problem_at_order_4_unit_interval() -> None:
    """exp M=4, hbar=-1: residual = -x^4/24, so int_0^1 (x^4/24)^2 dx = 1/5184."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    result = residual_l2_squared(sol, sp.Integer(-1), (sp.Integer(0), sp.Integer(1)))
    assert sp.simplify(result - sp.Rational(1, 5184)) == 0


def test_residual_l2_squared_at_order_zero_is_n_of_u0_squared_times_length() -> None:
    """For exp at order 0, residual = -1, so L2 squared over [0, 1] is 1."""
    problem = _exp_problem()
    sol = solve(problem, order=0)
    result = residual_l2_squared(sol, sp.Integer(-1), (sp.Integer(0), sp.Integer(1)))
    assert sp.simplify(result - sp.Integer(1)) == 0


def test_residual_l2_squared_retains_hbar_when_no_value_supplied() -> None:
    """With hbar_value=None the L2 norm is a polynomial in hbar."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    result = residual_l2_squared(sol, None, (sp.Integer(0), sp.Integer(1)))
    assert HBAR in result.free_symbols


def test_residual_l2_squared_for_quadratic_problem_at_order_3_unit_interval() -> None:
    """For quadratic M=3, hbar=-1: residual = -4x^3 - 3x^4 - 2x^5 - x^6.

    int_0^1 r^2 dx is a concrete rational; compute via sp.integrate and
    pin the value.
    """
    problem = _quadratic_problem()
    sol = solve(problem, order=3)
    result = residual_l2_squared(sol, sp.Integer(-1), (sp.Integer(0), sp.Integer(1)))
    r = -4 * X**3 - 3 * X**4 - 2 * X**5 - X**6
    expected = sp.integrate(r**2, (X, sp.Integer(0), sp.Integer(1)))
    assert sp.simplify(result - expected) == 0


# --- residual_discrete_sum_of_squares -------------------------------------


def test_discrete_sum_of_squares_at_empty_samples_is_zero() -> None:
    """No samples ⇒ sum is 0 (the empty-sum identity)."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    result = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [])
    assert result == 0


def test_discrete_sum_at_sample_zero_vanishes_for_exp_problem() -> None:
    """For the exp problem the residual at x=0 is zero for every M >= 1."""
    problem = _exp_problem()
    for m in range(1, 4):
        sol = solve(problem, order=m)
        result = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [sp.Integer(0)])
        assert sp.simplify(result) == 0


def test_discrete_sum_at_unit_sample_for_exp_problem_order_4() -> None:
    """At x=1 the exp-problem residual is -1/24, so sum of squares is 1/576."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    result = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [sp.Integer(1)])
    assert sp.simplify(result - sp.Rational(1, 576)) == 0


def test_discrete_sum_concatenates_additively() -> None:
    """sum_of_squares([a, b]) == sum_of_squares([a]) + sum_of_squares([b])."""
    problem = _exp_problem()
    sol = solve(problem, order=3)
    a, b = sp.Rational(1, 4), sp.Rational(3, 4)
    both = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [a, b])
    only_a = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [a])
    only_b = residual_discrete_sum_of_squares(sol, sp.Integer(-1), [b])
    assert sp.simplify(both - (only_a + only_b)) == 0


def test_discrete_sum_retains_hbar_when_no_value_supplied() -> None:
    """With hbar_value=None the discrete sum is a polynomial in hbar."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    result = residual_discrete_sum_of_squares(sol, None, [sp.Rational(1, 2), sp.Integer(1)])
    assert HBAR in result.free_symbols


# --- hbar_curve_at: partial sum as a function of hbar at fixed x ----------


def test_hbar_curve_at_zero_recovers_u0_for_exp_problem() -> None:
    """At x=0 every u_k vanishes (homogeneous BCs) except u_0=1; curve is constant 1."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    curve = hbar_curve_at(sol, sp.Integer(0))
    assert sp.expand(curve - sp.Integer(1)) == 0


def test_hbar_curve_at_one_for_exp_problem_order_2() -> None:
    """Hand-derived: partial sum at x=1 for the exp problem M=2 is 1 - 2*hbar - hbar^2/2.

    u_0(1) = 1, u_1(1) = -hbar, u_2(1) = hbar^2(1/2 - 1) - hbar = -hbar^2/2 - hbar.
    Sum: 1 + (-hbar) + (-hbar^2/2 - hbar) = 1 - 2*hbar - hbar^2/2.
    """
    problem = _exp_problem()
    sol = solve(problem, order=2)
    curve = hbar_curve_at(sol, sp.Integer(1))
    expected = sp.Integer(1) - 2 * HBAR - HBAR**2 / sp.Integer(2)
    assert sp.expand(curve - expected) == 0


def test_hbar_curve_substitution_matches_direct_evaluation() -> None:
    """hbar_curve_at(sol, x*).subs(hbar, v) == sol.evaluate_at_hbar(v).subs(x, x*)."""
    problem = _exp_problem()
    sol = solve(problem, order=3)
    x_star = sp.Rational(1, 2)
    hbar_value = sp.Integer(-1)
    via_curve = hbar_curve_at(sol, x_star).subs(HBAR, hbar_value)
    via_eval = sol.evaluate_at_hbar(hbar_value).subs(X, x_star)
    assert sp.simplify(via_curve - via_eval) == 0


def test_hbar_curve_is_polynomial_in_hbar() -> None:
    """At fixed x*, the partial sum is a polynomial in hbar of degree <= M."""
    problem = _exp_problem()
    sol = solve(problem, order=3)
    curve = hbar_curve_at(sol, sp.Integer(1))
    poly = sp.Poly(curve, HBAR)
    assert poly.degree() <= sol.order


# --- optimal_hbar: grid-search minimizer ----------------------------------


def test_optimal_hbar_rejects_empty_grid() -> None:
    """An empty grid has no minimum; optimal_hbar must surface that explicitly."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    norm = _l2_unit_interval_norm
    with pytest.raises(ValueError, match="non-empty"):
        optimal_hbar(sol, [], norm)


def test_optimal_hbar_with_singleton_grid_returns_that_element() -> None:
    """A grid with one value: optimal_hbar trivially returns it."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    norm = _l2_unit_interval_norm
    assert optimal_hbar(sol, [sp.Integer(-1)], norm) == sp.Integer(-1)


def test_optimal_hbar_picks_neg_one_over_zero_for_exp_problem() -> None:
    """For the exp problem at hbar=0 the partial sum is u_0=1 and the residual is -1.

    The L2 norm squared over [0,1] is therefore 1 at hbar=0 versus 1/5184
    at hbar=-1; optimal_hbar must return -1.
    """
    problem = _exp_problem()
    sol = solve(problem, order=4)
    norm = _l2_unit_interval_norm
    grid = [sp.Integer(0), sp.Integer(-1)]
    assert optimal_hbar(sol, grid, norm) == sp.Integer(-1)


def test_optimal_hbar_via_discrete_norm_picks_neg_one_for_exp_problem() -> None:
    """Same outcome via the discrete sum-of-squares at samples [1/2, 1]."""
    problem = _exp_problem()
    sol = solve(problem, order=4)

    def norm(s, h):  # type: ignore[no-untyped-def]
        return residual_discrete_sum_of_squares(s, h, [sp.Rational(1, 2), sp.Integer(1)])

    grid = [sp.Integer(0), sp.Integer(-1), sp.Rational(-3, 2)]
    assert optimal_hbar(sol, grid, norm) == sp.Integer(-1)


def _l2_unit_interval_norm(s, h):  # type: ignore[no-untyped-def]
    """Closure-free norm bound to interval [0, 1] for re-use across tests."""
    return residual_l2_squared(s, h, (sp.Integer(0), sp.Integer(1)))
