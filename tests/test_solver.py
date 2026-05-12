"""Tests for the HAM solver (Stage 5).

Stage 5a slice: the HamSolution return type and the per-step solve_step
function. Hand-derived u_1 / u_2 for two reference problems pin the
behaviour of one HAM step. The solve driver and end-to-end Taylor-match
tests live in 5b.
"""

import dataclasses

import pytest
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.series import QSeries
from ham.solver import HamSolution, solve, solve_step

from tests.strategies import X

U = sp.Function("u")
HBAR = sp.Symbol("hbar")


def _ivp_operator() -> LinearOperator:
    """L = d/dx with homogeneous BC u(0) = 0 (the deformation BC)."""
    return LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )


def _exp_problem() -> HamProblem:
    """u' = u, u(0) = 1: N[u] = u' - u, u_0 = 1, exact solution exp(x)."""
    return HamProblem(
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X), dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


def _quadratic_problem() -> HamProblem:
    """u' = u^2, u(0) = 1: N[u] = u' - u^2, u_0 = 1, exact solution 1/(1-x)."""
    return HamProblem(
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X) ** 2, dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


# --- HamSolution data type ------------------------------------------------


def test_ham_solution_exposes_problem_and_phi() -> None:
    """HamSolution stores the originating problem and the homotopy series."""
    problem = _exp_problem()
    phi = QSeries([sp.Integer(1)], order=0)
    sol = HamSolution(problem=problem, phi=phi)
    assert sol.problem is problem
    assert sol.phi is phi


def test_ham_solution_is_frozen() -> None:
    """HamSolution is a frozen dataclass."""
    problem = _exp_problem()
    phi = QSeries([sp.Integer(1)], order=0)
    sol = HamSolution(problem=problem, phi=phi)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sol.phi = QSeries.zero(order=0)  # type: ignore[misc]


def test_ham_solution_order_matches_phi_order() -> None:
    """sol.order is sol.phi.order (the working order M)."""
    problem = _exp_problem()
    phi = QSeries([sp.Integer(1), X, X**2], order=2)
    sol = HamSolution(problem=problem, phi=phi)
    assert sol.order == 2


def test_partial_sum_is_sum_of_phi_coefficients() -> None:
    """partial_sum() == Σ_k phi.coeff(k), ℏ kept symbolic in the result."""
    problem = _exp_problem()
    phi = QSeries([sp.Integer(1), -HBAR * X, HBAR**2 * (X**2 / 2 - X) - HBAR * X], order=2)
    sol = HamSolution(problem=problem, phi=phi)
    expected = sp.Integer(1) + (-HBAR * X) + (HBAR**2 * (X**2 / 2 - X) - HBAR * X)
    assert sp.expand(sol.partial_sum() - expected) == 0


def test_evaluate_at_hbar_substitutes_value() -> None:
    """evaluate_at_hbar(v) substitutes ℏ → v into partial_sum()."""
    problem = _exp_problem()
    phi = QSeries([sp.Integer(1), -HBAR * X], order=1)
    sol = HamSolution(problem=problem, phi=phi)
    assert sp.expand(sol.evaluate_at_hbar(sp.Integer(-1)) - (1 + X)) == 0


# --- solve_step: one HAM step ---------------------------------------------


def test_solve_step_for_exp_problem_at_m_equals_1() -> None:
    """First-order HAM step for u' = u, u(0)=1: u_1 = -ℏ x.

    R_1 = N[u_0] = (1)' - 1 = -1; rhs = ℏ · 1 · (-1) = -ℏ;
    L.invert(-ℏ) = ∫_0^x -ℏ ds = -ℏ x; chi_1 = 0 so u_1 = -ℏ x.
    """
    problem = _exp_problem()
    phi_0 = QSeries.constant(problem.u0, order=0)
    u_1 = solve_step(problem, phi_0, 1)
    assert sp.expand(u_1 - (-HBAR * X)) == 0


def test_solve_step_for_exp_problem_at_m_equals_2() -> None:
    """Second-order HAM step for u' = u: u_2 = ℏ²(x²/2 - x) - ℏ x.

    phi_1 = 1 + (-ℏ x) q; N[phi] = phi' - phi
    [q^1] phi' = -ℏ; [q^1] phi = -ℏ x; so R_2 = -ℏ - (-ℏ x) = -ℏ + ℏ x.
    rhs = ℏ R_2 = ℏ²(x - 1); ∫_0^x ℏ²(s - 1) ds = ℏ²(x²/2 - x);
    chi_2 = 1, so u_2 = ℏ²(x²/2 - x) + (-ℏ x) = ℏ²(x²/2 - x) - ℏ x.
    """
    problem = _exp_problem()
    phi_1 = QSeries([sp.Integer(1), -HBAR * X], order=1)
    u_2 = solve_step(problem, phi_1, 2)
    expected = HBAR**2 * (X**2 / sp.Integer(2) - X) - HBAR * X
    assert sp.expand(u_2 - expected) == 0


def test_solve_step_for_quadratic_problem_at_m_equals_1() -> None:
    """First-order step for u' = u², u(0)=1: u_1 = -ℏ x.

    R_1 = N[u_0] = 0 - 1 = -1, same as the exp problem at m=1.
    """
    problem = _quadratic_problem()
    phi_0 = QSeries.constant(problem.u0, order=0)
    u_1 = solve_step(problem, phi_0, 1)
    assert sp.expand(u_1 - (-HBAR * X)) == 0


def test_solve_step_for_quadratic_problem_at_m_equals_2() -> None:
    """Second-order step for u' = u²: u_2 = ℏ²(x² - x) - ℏ x.

    phi_1 = 1 + (-ℏ x) q; N[phi] = phi' - phi²
    [q^1] phi' = -ℏ; [q^1] phi² = 2·1·(-ℏx) = -2ℏx;
    so R_2 = -ℏ - (-2ℏx) = -ℏ + 2ℏ x = ℏ(2x - 1);
    rhs = ℏ² (2x - 1); ∫_0^x ℏ²(2s - 1) ds = ℏ²(x² - x);
    chi_2 = 1, so u_2 = ℏ²(x² - x) + (-ℏ x) = ℏ²(x² - x) - ℏ x.
    """
    problem = _quadratic_problem()
    phi_1 = QSeries([sp.Integer(1), -HBAR * X], order=1)
    u_2 = solve_step(problem, phi_1, 2)
    expected = HBAR**2 * (X**2 - X) - HBAR * X
    assert sp.expand(u_2 - expected) == 0


def test_solve_step_at_hbar_minus_one_collapses_to_taylor_term() -> None:
    """At ℏ = -1: u_1(exp) = x, u_2(exp) = x²/2 (matches exp(x) Taylor)."""
    problem = _exp_problem()
    phi_0 = QSeries.constant(problem.u0, order=0)
    u_1 = solve_step(problem, phi_0, 1).subs(HBAR, sp.Integer(-1))
    assert sp.expand(u_1 - X) == 0

    phi_1 = QSeries([sp.Integer(1), u_1], order=1)
    u_2 = solve_step(problem, phi_1, 2).subs(HBAR, sp.Integer(-1))
    assert sp.expand(u_2 - X**2 / sp.Integer(2)) == 0


# --- solve: end-to-end driver ---------------------------------------------


def test_solve_at_order_zero_returns_u0() -> None:
    """solve(problem, 0) returns a solution with phi = constant(u_0, order=0)."""
    problem = _exp_problem()
    sol = solve(problem, order=0)
    assert sol.order == 0
    assert sp.expand(sol.phi.coeff(0) - problem.u0) == 0
    assert sp.expand(sol.partial_sum() - problem.u0) == 0


def test_solve_rejects_negative_order() -> None:
    """solve(problem, -1) raises ValueError (working order is non-negative)."""
    problem = _exp_problem()
    with pytest.raises(ValueError, match="order"):
        solve(problem, order=-1)


def test_solve_phi_carries_hand_derived_coefficients_on_exp_problem() -> None:
    """For u' = u, the QSeries.coeff(k) values match the hand-derivations."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    assert sp.expand(sol.phi.coeff(0) - 1) == 0
    assert sp.expand(sol.phi.coeff(1) - (-HBAR * X)) == 0
    expected_u_2 = HBAR**2 * (X**2 / sp.Integer(2) - X) - HBAR * X
    assert sp.expand(sol.phi.coeff(2) - expected_u_2) == 0


def test_solve_exp_problem_matches_taylor_to_order_4_at_hbar_minus_one() -> None:
    """For u' = u, u(0) = 1 at ℏ = -1: HAM partial sum equals 1 + x + x²/2 + x³/6 + x⁴/24."""
    problem = _exp_problem()
    sol = solve(problem, order=4)
    expansion = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = sum((X**k / sp.factorial(k) for k in range(5)), sp.Integer(0))
    assert sp.expand(expansion - expected) == 0


def test_solve_quadratic_problem_matches_geometric_taylor_to_order_3_at_hbar_minus_one() -> None:
    """For u' = u², u(0) = 1 at ℏ = -1: HAM partial sum equals 1 + x + x² + x³."""
    problem = _quadratic_problem()
    sol = solve(problem, order=3)
    expansion = sol.evaluate_at_hbar(sp.Integer(-1))
    expected = sum((X**k for k in range(4)), sp.Integer(0))
    assert sp.expand(expansion - expected) == 0


def test_solve_at_order_m_returns_phi_of_order_m() -> None:
    """solve to order M produces phi with phi.order == M (M+1 coefficients)."""
    problem = _exp_problem()
    for m in range(5):
        sol = solve(problem, order=m)
        assert sol.order == m
        assert sol.phi.coeff(m) is not None  # coefficient at index M exists


def test_solve_keeps_hbar_symbolic() -> None:
    """Coefficients u_k for k >= 1 retain hbar as a free symbol."""
    problem = _exp_problem()
    sol = solve(problem, order=2)
    assert HBAR in sol.phi.coeff(1).free_symbols
    assert HBAR in sol.phi.coeff(2).free_symbols
    # u_0 is constant, no hbar dependence
    assert HBAR not in sol.phi.coeff(0).free_symbols
