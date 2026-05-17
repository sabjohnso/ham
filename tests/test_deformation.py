"""Tests for the HAM deformation-equation builder (Stage 4).

Stage 4a slice: the HamProblem[sp.Expr] dataclass bundling (L, N, H, hbar, u_0)
and the chi_m heaviside function. r_m / rhs_m land in 4b.
"""

import dataclasses

import pytest
import sympy as sp
from ham.deformation import HamProblem, chi_m
from ham.nonlinear import NonlinearOperator
from ham.operator import LinearOperator
from ham.series import QSeries

from tests.strategies import X

U = sp.Function("u")
HBAR = sp.Symbol("hbar")


# --- chi_m: the m-th-equation heaviside -----------------------------------


def test_chi_at_zero_is_zero() -> None:
    """chi_m(0) == 0 (m=0 is below the deformation index range)."""
    assert chi_m(0) == 0


def test_chi_at_one_is_zero() -> None:
    """chi_m(1) == 0 (drops L[u_0] from the m=1 equation since u_0 is given)."""
    assert chi_m(1) == 0


def test_chi_at_two_is_one() -> None:
    """chi_m(2) == 1 (recurrence kicks in)."""
    assert chi_m(2) == 1


def test_chi_above_two_is_one() -> None:
    """chi_m(m) == 1 for every m >= 2."""
    for m in range(3, 10):
        assert chi_m(m) == 1


# --- HamProblem[sp.Expr] data type -------------------------------------------------


def test_ham_problem_stores_all_fields() -> None:
    """HamProblem[sp.Expr] exposes L, N, H, hbar, u0 as attributes."""
    l_op = LinearOperator(var=X, action=lambda e: sp.diff(e, X))
    n_op = NonlinearOperator(expr=U(X) ** 2, dependent=U, indep=X)
    problem = HamProblem[sp.Expr](L=l_op, N=n_op, H=X**2 + sp.Integer(1), hbar=HBAR, u0=X)
    assert problem.L is l_op
    assert problem.N is n_op
    assert sp.expand(problem.H - (X**2 + 1)) == 0
    assert problem.hbar is HBAR
    assert sp.expand(problem.u0 - X) == 0


def test_ham_problem_is_frozen() -> None:
    """HamProblem[sp.Expr] is a frozen dataclass — fields cannot be reassigned."""
    l_op = LinearOperator(var=X, action=lambda e: sp.diff(e, X))
    n_op = NonlinearOperator(expr=U(X), dependent=U, indep=X)
    problem = HamProblem[sp.Expr](L=l_op, N=n_op, H=sp.Integer(1), hbar=HBAR, u0=sp.Integer(0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        problem.hbar = sp.Integer(2)  # type: ignore[misc]


# --- r_m and rhs_m: the m-th deformation equation RHS ---------------------


def _make_problem(n_expr: sp.Expr, u0: sp.Expr = X, h: sp.Expr = sp.S.One) -> HamProblem[sp.Expr]:
    """A reusable HAM problem with L = d/dx (no BCs needed for Stage 4)."""
    return HamProblem[sp.Expr](
        L=LinearOperator(var=X, action=lambda e: sp.diff(e, X)),
        N=NonlinearOperator(expr=n_expr, dependent=U, indep=X),
        H=h,
        hbar=HBAR,
        u0=u0,
    )


def test_r_m_rejects_m_below_one() -> None:
    """r_m raises ValueError for m < 1 (deformation equations are 1-indexed)."""
    problem = _make_problem(n_expr=U(X))
    phi = QSeries.constant(X, order=0)
    with pytest.raises(ValueError, match="m >= 1"):
        problem.r_m(phi, 0)
    with pytest.raises(ValueError, match="m >= 1"):
        problem.r_m(phi, -1)


def test_r_m_rejects_phi_order_below_m_minus_one() -> None:
    """r_m raises ValueError when phi does not carry coeffs up through u_{m-1}."""
    problem = _make_problem(n_expr=U(X))
    phi = QSeries.constant(X, order=0)  # carries only u_0
    with pytest.raises(ValueError, match=r"phi\.order"):
        problem.r_m(phi, 2)  # would need u_1 in phi


def test_r_m_at_one_is_n_of_u0() -> None:
    """R_1 = [q^0] N[phi_0] = N[u_0] (Liao's m=1 cross-check)."""
    n_expr = U(X) ** 2 + sp.sin(X)
    problem = _make_problem(n_expr=n_expr, u0=X)
    phi = QSeries.constant(problem.u0, order=0)
    expected = problem.N.apply_scalar(problem.u0)
    assert sp.expand(problem.r_m(phi, 1) - expected) == 0


def test_r_m_for_linear_n_reduces_to_phi_coefficient() -> None:
    """For N(u) = u (linear), R_m = u_{m-1} = phi.coeff(m-1)."""
    problem = _make_problem(n_expr=U(X))
    phi = QSeries([X, X**2, X**3], order=2)
    for m in range(1, phi.order + 2):
        assert sp.expand(problem.r_m(phi, m) - phi.coeff(m - 1)) == 0


def test_r_m_for_quadratic_n_matches_cauchy_sum() -> None:
    """For N(u) = u^2, R_m = sum_{i+j=m-1} u_i u_j (Cauchy)."""
    problem = _make_problem(n_expr=U(X) ** 2)
    phi = QSeries([X, X + 1, sp.Integer(3)], order=2)
    for m in range(1, phi.order + 2):
        expected: sp.Expr = sp.Integer(0)
        for i in range(m):
            expected = expected + phi.coeff(i) * phi.coeff(m - 1 - i)
        assert sp.expand(problem.r_m(phi, m) - expected) == 0


def test_r_m_for_u_times_u_prime_includes_derivatives() -> None:
    """For N(u) = u*u', R_m involves coefficient-wise x-derivatives via Cauchy."""
    problem = _make_problem(n_expr=U(X) * U(X).diff(X))
    phi = QSeries([X**2, X, sp.Integer(5)], order=2)
    for m in range(1, phi.order + 2):
        expected = sp.Integer(0)
        for i in range(m):
            expected = expected + phi.coeff(i) * sp.diff(phi.coeff(m - 1 - i), X)
        assert sp.expand(problem.r_m(phi, m) - expected) == 0


def test_rhs_m_is_hbar_times_h_times_r_m() -> None:
    """rhs_m = hbar * H * R_m by construction (composition law)."""
    problem = _make_problem(n_expr=U(X) ** 2, u0=X, h=X**2 + sp.Integer(1))
    phi = QSeries([X, X**2], order=1)
    for m in range(1, 3):
        r = problem.r_m(phi, m)
        assert sp.expand(problem.rhs_m(phi, m) - HBAR * problem.H * r) == 0


def test_r_m_is_causal_in_q() -> None:
    """Perturbing phi.coeff(M) for M > m-1 leaves R_m invariant.

    Stage 3's causality lifts cleanly: R_m only ever depends on the
    coefficients of phi at index 0..m-1.
    """
    problem = _make_problem(n_expr=U(X) ** 2)
    coeffs = [X, X**2, sp.Integer(5), X + sp.Integer(1)]
    phi = QSeries(coeffs, order=3)
    perturbed = list(coeffs)
    perturbed[3] = perturbed[3] + sp.Integer(999) * X
    phi_perturbed = QSeries(perturbed, order=3)

    for m in range(1, 4):  # m in 1..3 reads only u_0..u_2; perturbed u_3 must not enter
        original = problem.r_m(phi, m)
        shifted = problem.r_m(phi_perturbed, m)
        assert sp.expand(original - shifted) == 0


def test_r_m_at_one_is_independent_of_higher_phi_coeffs() -> None:
    """R_1 only sees phi.coeff(0); higher coeffs of phi must not enter."""
    problem = _make_problem(n_expr=U(X) ** 3)
    phi_short = QSeries.constant(X, order=0)
    phi_long = QSeries([X, X**5, X**7, sp.Integer(42)], order=3)
    assert sp.expand(problem.r_m(phi_short, 1) - problem.r_m(phi_long, 1)) == 0
