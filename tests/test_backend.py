"""Algebraic laws for Backend[C].

A Backend packages the operations Series[C] and NonlinearOperator
cannot perform polymorphically through Python `+ - *` arithmetic:
identities (`zero`, `one`), the lift of x-only sympy expressions into
C (`lift_xonly`), and the x-calculus (`diff_x`, `integrate_x`).

The laws below assert these operations satisfy the commutative-ring +
differential-ring structure on the coefficient type C. From S5b
onward the same law file will be parametrised over a list of Backend
fixtures (sympy, spectral-over-float, spectral-over-sympy); for S0
only the sympy backend exists and equality is sp.expand-based per
PLAN.org D-4.
"""

import sympy as sp
from ham.backend import Backend, SympyBackend
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import X, polynomial_in_x


def sympy_equal(a: sp.Expr, b: sp.Expr) -> bool:
    """Equality comparator for the sympy backend, supplied per D-4."""
    return bool(sp.expand(a - b) == 0)


backend: Backend[sp.Expr] = SympyBackend(X)


# --- Ring identities -------------------------------------------------------


@given(polynomial_in_x())
def test_zero_is_left_additive_identity(p: sp.Expr) -> None:
    """zero() + p == p."""
    assert sympy_equal(backend.zero() + p, p)


@given(polynomial_in_x())
def test_zero_is_right_additive_identity(p: sp.Expr) -> None:
    """p + zero() == p."""
    assert sympy_equal(p + backend.zero(), p)


@given(polynomial_in_x())
def test_one_is_left_multiplicative_identity(p: sp.Expr) -> None:
    """one() * p == p."""
    assert sympy_equal(backend.one() * p, p)


@given(polynomial_in_x())
def test_one_is_right_multiplicative_identity(p: sp.Expr) -> None:
    """p * one() == p."""
    assert sympy_equal(p * backend.one(), p)


# --- lift_xonly ------------------------------------------------------------


@given(polynomial_in_x())
def test_lift_xonly_is_identity_for_sympy_backend(p: sp.Expr) -> None:
    """For SympyBackend, lift_xonly is identity (input already lives in C)."""
    assert sympy_equal(backend.lift_xonly(p), p)


# --- Differential-ring laws ------------------------------------------------


@given(polynomial_in_x())
def test_diff_x_at_zero_is_identity(p: sp.Expr) -> None:
    """diff_x(p, 0) == p."""
    assert sympy_equal(backend.diff_x(p, 0), p)


@given(
    polynomial_in_x(),
    st.integers(min_value=0, max_value=2),
    st.integers(min_value=0, max_value=2),
)
def test_diff_x_composition(p: sp.Expr, j: int, k: int) -> None:
    """diff_x(diff_x(p, j), k) == diff_x(p, j + k)."""
    assert sympy_equal(
        backend.diff_x(backend.diff_x(p, j), k),
        backend.diff_x(p, j + k),
    )


@given(polynomial_in_x(), polynomial_in_x())
def test_diff_x_is_linear(p: sp.Expr, q: sp.Expr) -> None:
    """diff_x is linear: diff_x(a*p + b*q, 1) == a*diff_x(p, 1) + b*diff_x(q, 1)."""
    a, b = sp.Integer(3), sp.Integer(-5)
    lhs = backend.diff_x(a * p + b * q, 1)
    rhs = a * backend.diff_x(p, 1) + b * backend.diff_x(q, 1)
    assert sympy_equal(lhs, rhs)


@given(polynomial_in_x())
def test_diff_inverts_integrate_for_polynomials(p: sp.Expr) -> None:
    """diff_x(integrate_x(p), 1) == p — fundamental theorem of calculus."""
    primitive = backend.integrate_x(p)
    assert sympy_equal(backend.diff_x(primitive, 1), p)


@given(polynomial_in_x())
def test_integrate_x_vanishes_at_lower_bound(p: sp.Expr) -> None:
    """integrate_x(p) evaluated at the lower bound (0) is 0, by construction."""
    primitive = backend.integrate_x(p)
    assert sympy_equal(primitive.subs(X, sp.Integer(0)), sp.Integer(0))


@given(polynomial_in_x(), polynomial_in_x())
def test_integrate_x_is_linear(p: sp.Expr, q: sp.Expr) -> None:
    """integrate_x is linear: integrate_x(a*p + b*q) == a*integrate_x(p) + b*integrate_x(q)."""
    a, b = sp.Integer(3), sp.Integer(-5)
    lhs = backend.integrate_x(a * p + b * q)
    rhs = a * backend.integrate_x(p) + b * backend.integrate_x(q)
    assert sympy_equal(lhs, rhs)
