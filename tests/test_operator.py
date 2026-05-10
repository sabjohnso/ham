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

import pytest
import sympy as sp
from ham.operator import BoundaryCondition, LinearOperator, antiderivative
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import X, polynomial_in_x, qseries_polynomial_coeffs


def _diff_x(e: sp.Expr) -> sp.Expr:
    return sp.diff(e, X)


@given(u=polynomial_in_x())
def test_apply_delegates_to_action(u: sp.Expr) -> None:
    """LinearOperator(action).apply(u) == action(u)."""
    op = LinearOperator(var=X, action=_diff_x)
    assert sp.expand(op.apply(u) - _diff_x(u)) == 0


def test_apply_of_zero_is_zero() -> None:
    """L[0] == 0 for L = d/dx (smoke test of a linear action through the wrapper)."""
    op = LinearOperator(var=X, action=_diff_x)
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
    op = LinearOperator(var=X, action=_diff_x)
    lhs = op.apply(alpha * u + beta * v)
    rhs = alpha * op.apply(u) + beta * op.apply(v)
    assert sp.expand(lhs - rhs) == 0


@given(s=qseries_polynomial_coeffs())
def test_apply_series_is_coefficient_wise(s: QSeries) -> None:
    """[q^k] L.apply_series(s) == action(s.coeff(k)) for L = d/dx."""
    op = LinearOperator(var=X, action=_diff_x)
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
    op = LinearOperator(var=X, action=_diff_x)
    lhs = op.apply_series(alpha * s + beta * t)
    rhs = alpha * op.apply_series(s) + beta * op.apply_series(t)
    order = min(lhs.order, rhs.order)
    for k in range(order + 1):
        assert sp.expand(lhs.coeff(k) - rhs.coeff(k)) == 0


# --- Inversion (Stage 2b): BCs + canonical-case antiderivative -------------


def _canonical_ivp() -> LinearOperator:
    """L = d/dx with u(0) = 0, inverted by definite integration from 0."""
    return LinearOperator(
        var=X,
        action=_diff_x,
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
        inverter=antiderivative(X, sp.Integer(0)),
    )


def test_invert_without_inverter_raises() -> None:
    """L.invert with no inverter set is a placeholder until 2c lands the default."""
    op = LinearOperator(var=X, action=_diff_x)
    with pytest.raises(NotImplementedError):
        op.invert(sp.Integer(1))


@given(rhs=polynomial_in_x())
def test_apply_invert_is_identity_on_image(rhs: sp.Expr) -> None:
    """L o L^{-1} = id on the image: apply(invert(rhs)) == rhs."""
    op = _canonical_ivp()
    assert sp.expand(op.apply(op.invert(rhs)) - rhs) == 0


@given(u=polynomial_in_x())
def test_invert_apply_recovers_modulo_kernel(u: sp.Expr) -> None:
    """L^{-1} o L = id modulo kernel: for the canonical IVP, invert(apply(u)) == u - u(0)."""
    op = _canonical_ivp()
    recovered = op.invert(op.apply(u))
    expected = u - u.subs(X, 0)
    assert sp.expand(recovered - expected) == 0


@given(rhs=polynomial_in_x())
def test_invert_satisfies_declared_bcs(rhs: sp.Expr) -> None:
    """For each declared BC, the inverted u satisfies it pointwise."""
    op = _canonical_ivp()
    u = op.invert(rhs)
    for bc in op.bcs:
        derivative = sp.diff(u, X, bc.derivative_order)
        assert sp.expand(derivative.subs(X, bc.point) - bc.value) == 0


def test_boundary_condition_default_value_is_zero() -> None:
    """BoundaryCondition.value defaults to sympy.Integer(0) (homogeneous HAM case)."""
    bc = BoundaryCondition(point=sp.Integer(0), derivative_order=0)
    assert bc.value == sp.Integer(0)
