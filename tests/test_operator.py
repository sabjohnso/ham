"""Property tests for LinearOperator — the auxiliary linear operator L.

Stage 2a of the HAM build (see PLAN.org): forward action only. The wrapper
delegates `apply` to a Callable[[Expr], Expr] and lifts it coefficient-wise
to QSeries via `apply_series`. BC data and inversion strategy are deferred
to 2b/2c.

Linearity is not enforced by the wrapper — the user supplies a Callable
they claim is linear. These tests exercise the wrapper against a sample
of known-linear actions (here, d/dx) to verify the wrapper itself does
not break linearity en route.
"""

import sympy as sp
from ham.operator import LinearOperator
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import X, polynomial_in_x, qseries_polynomial_coeffs


def _diff_x(e: sp.Expr) -> sp.Expr:
    return sp.diff(e, X)


@given(u=polynomial_in_x())
def test_apply_delegates_to_action(u: sp.Expr) -> None:
    """LinearOperator(action).apply(u) == action(u)."""
    op = LinearOperator(action=_diff_x)
    assert sp.expand(op.apply(u) - _diff_x(u)) == 0


def test_apply_of_zero_is_zero() -> None:
    """L[0] == 0 for L = d/dx (smoke test of a linear action through the wrapper)."""
    op = LinearOperator(action=_diff_x)
    assert op.apply(sp.Integer(0)) == sp.Integer(0)


@given(
    u=polynomial_in_x(),
    v=polynomial_in_x(),
    alpha_int=st.integers(min_value=-10, max_value=10),
    beta_int=st.integers(min_value=-10, max_value=10),
)
def test_apply_preserves_linearity_of_diff_x(
    u: sp.Expr, v: sp.Expr, alpha_int: int, beta_int: int
) -> None:
    """L[alpha*u + beta*v] == alpha*L[u] + beta*L[v] for L = d/dx."""
    alpha = sp.Integer(alpha_int)
    beta = sp.Integer(beta_int)
    op = LinearOperator(action=_diff_x)
    lhs = op.apply(alpha * u + beta * v)
    rhs = alpha * op.apply(u) + beta * op.apply(v)
    assert sp.expand(lhs - rhs) == 0


@given(s=qseries_polynomial_coeffs())
def test_apply_series_is_coefficient_wise(s: QSeries) -> None:
    """[q^k] L.apply_series(s) == action(s.coeff(k)) for L = d/dx."""
    op = LinearOperator(action=_diff_x)
    out = op.apply_series(s)
    assert out.order == s.order
    for k in range(s.order + 1):
        assert sp.expand(out.coeff(k) - _diff_x(s.coeff(k))) == 0


@given(
    s=qseries_polynomial_coeffs(),
    t=qseries_polynomial_coeffs(),
    alpha_int=st.integers(min_value=-10, max_value=10),
    beta_int=st.integers(min_value=-10, max_value=10),
)
def test_apply_series_preserves_linearity(
    s: QSeries, t: QSeries, alpha_int: int, beta_int: int
) -> None:
    """L[alpha*s + beta*t] == alpha*L[s] + beta*L[t] coefficient-wise."""
    alpha = sp.Integer(alpha_int)
    beta = sp.Integer(beta_int)
    op = LinearOperator(action=_diff_x)
    lhs = op.apply_series(alpha * s + beta * t)
    rhs = alpha * op.apply_series(s) + beta * op.apply_series(t)
    order = min(lhs.order, rhs.order)
    for k in range(order + 1):
        assert sp.expand(lhs.coeff(k) - rhs.coeff(k)) == 0
