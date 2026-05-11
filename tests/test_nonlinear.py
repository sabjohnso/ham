"""Property tests for NonlinearOperator — the HAM nonlinear operator N.

Stage 3a (this file, initial slice): scalar evaluation only. N wraps a
sympy expression in u(x), u'(x), ... and substitutes a concrete function
for u, evaluating derivatives. apply_series (substituting a QSeries for u,
polynomial regime) is 3b/3c. Transcendental rejection is 3d.
"""

import pytest
import sympy as sp
from ham.nonlinear import NonlinearOperator
from ham.series import QSeries
from hypothesis import given

from tests.strategies import X, polynomial_in_x, qseries_polynomial_coeffs

U = sp.Function("u")
"""The dependent-function symbol used across these tests."""


# --- Stage 3a: apply_scalar -----------------------------------------------


def test_apply_scalar_constant_expression() -> None:
    """N built from an expr with no u-dependence returns the expr unchanged."""
    expr = X**2 + sp.Integer(3)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_scalar(sp.cos(X))
    assert sp.expand(result - expr) == 0


def test_apply_scalar_substitutes_u() -> None:
    """N(u) = u^2 evaluated at u = x is x^2."""
    expr = U(X) ** 2
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_scalar(X)
    assert sp.expand(result - X**2) == 0


def test_apply_scalar_evaluates_first_derivative() -> None:
    """N(u) = u' evaluated at u = sin(x) is cos(x)."""
    expr = U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_scalar(sp.sin(X))
    assert sp.simplify(result - sp.cos(X)) == 0


def test_apply_scalar_evaluates_higher_derivative() -> None:
    """N(u) = u'' evaluated at u = x^3 is 6*x."""
    expr = U(X).diff(X, 2)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_scalar(X**3)
    assert sp.expand(result - 6 * X) == 0


def test_apply_scalar_handles_mixed_expression() -> None:
    """N(u) = u*u' + u^2 - x evaluated at u = x gives x + x^2 - x = x^2."""
    expr = U(X) * U(X).diff(X) + U(X) ** 2 - X
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_scalar(X)
    assert sp.expand(result - X**2) == 0


@given(u=polynomial_in_x())
def test_apply_scalar_of_identity_is_identity(u: sp.Expr) -> None:
    """N(u) = u: apply_scalar(u_concrete) == u_concrete."""
    expr = U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    assert sp.expand(n_op.apply_scalar(u) - u) == 0


@given(u=polynomial_in_x())
def test_apply_scalar_constant_compatibility(u: sp.Expr) -> None:
    """For any u, an x-only expr is unaffected by apply_scalar."""
    expr = sp.Integer(7) * X**2 - X + sp.Integer(2)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    assert sp.expand(n_op.apply_scalar(u) - expr) == 0


@given(u=polynomial_in_x())
def test_apply_scalar_squares_polynomial(u: sp.Expr) -> None:
    """N(u) = u^2: apply_scalar(u) == u^2 for polynomial u."""
    expr = U(X) ** 2
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    assert sp.expand(n_op.apply_scalar(u) - u**2) == 0


# --- Stage 3b: apply_series for polynomial-in-u, no derivatives -----------


def test_apply_series_constant_expression() -> None:
    """N(u) = 5 evaluated against any phi is the constant series 5."""
    expr = sp.Integer(5)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2, sp.Integer(3)], order=2)
    result = n_op.apply_series(phi)
    assert result.order == phi.order
    assert sp.expand(result.coeff(0) - 5) == 0
    assert result.coeff(1) == 0
    assert result.coeff(2) == 0


def test_apply_series_identity_returns_phi() -> None:
    """N(u) = u: apply_series(phi) == phi coefficient-wise."""
    expr = U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2, sp.Integer(3)], order=2)
    result = n_op.apply_series(phi)
    assert result.order == phi.order
    for k in range(phi.order + 1):
        assert sp.expand(result.coeff(k) - phi.coeff(k)) == 0


def test_apply_series_addition_in_u() -> None:
    """N(u) = u + 5: apply_series adds 5 to coeff(0) and leaves the rest."""
    expr = U(X) + sp.Integer(5)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - (X + 5)) == 0
    assert sp.expand(result.coeff(1) - X**2) == 0


def test_apply_series_integer_scalar_in_u() -> None:
    """N(u) = 2*u: apply_series doubles every coefficient."""
    expr = 2 * U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - 2 * X) == 0
    assert sp.expand(result.coeff(1) - 2 * X**2) == 0


def test_apply_series_x_dependent_scalar_in_u() -> None:
    """N(u) = x*u: apply_series multiplies every coefficient by x (gluing law)."""
    expr = X * U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1), sp.Integer(2)], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - X) == 0
    assert sp.expand(result.coeff(1) - 2 * X) == 0


def test_apply_series_u_squared_explicit_cauchy() -> None:
    """N(u) = u^2: result.coeff(k) == sum_{i+j=k} phi.coeff(i) * phi.coeff(j)."""
    expr = U(X) ** 2
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X + 1, sp.Integer(3)], order=2)
    result = n_op.apply_series(phi)
    for k in range(phi.order + 1):
        expected: sp.Expr = sp.Integer(0)
        for i in range(k + 1):
            expected = expected + phi.coeff(i) * phi.coeff(k - i)
        assert sp.expand(result.coeff(k) - expected) == 0


def test_apply_series_eagerly_truncates_to_phi_order() -> None:
    """Cubic-in-u operator on a phi of order 1: result.order == 1 (not 3)."""
    expr = U(X) ** 3 + U(X) ** 2 + U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X + 1], order=1)
    result = n_op.apply_series(phi)
    assert result.order == phi.order


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_u_squared_matches_phi_times_phi(phi: QSeries) -> None:
    """N(u) = u^2: result agrees with (phi * phi).trunc(phi.order) coefficient-wise."""
    expr = U(X) ** 2
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_series(phi)
    expected = (phi * phi).trunc(phi.order)
    assert result.order == expected.order
    for k in range(phi.order + 1):
        assert sp.expand(result.coeff(k) - expected.coeff(k)) == 0


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_constant_compatibility(phi: QSeries) -> None:
    """For a u-free expr c(x), apply_series(phi) == constant_series(c, phi.order)."""
    expr = X**2 + sp.Integer(3) * X + sp.Integer(2)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_series(phi)
    assert result.order == phi.order
    assert sp.expand(result.coeff(0) - expr) == 0
    for k in range(1, phi.order + 1):
        assert result.coeff(k) == 0


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_causality_in_q(phi: QSeries) -> None:
    """N[phi].coeff(k) depends only on phi.coeff(0..k) for k < phi.order.

    Perturbing phi.coeff(phi.order) by an arbitrary expression must leave
    every result coefficient with index strictly less than phi.order
    unchanged (Cauchy structure: high-q tail cannot bleed downwards).
    """
    expr = U(X) ** 2 + U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result_orig = n_op.apply_series(phi)

    perturbed = [phi.coeff(k) for k in range(phi.order + 1)]
    perturbed[phi.order] = perturbed[phi.order] + sp.Integer(999) * X
    phi_perturbed = QSeries(perturbed, order=phi.order)
    result_perturbed = n_op.apply_series(phi_perturbed)

    for k in range(phi.order):
        assert sp.expand(result_orig.coeff(k) - result_perturbed.coeff(k)) == 0


def test_apply_series_sin_of_u_rejected() -> None:
    """sin(u) is not polynomial in u; compiler raises NotImplementedError."""
    expr = sp.sin(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2], order=1)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


# --- Stage 3c: apply_series with x-derivatives of u -----------------------


def test_apply_series_first_derivative_in_u() -> None:
    """N(u) = u': apply_series differentiates each q-coefficient in x."""
    expr = U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X**2, X, sp.Integer(5)], order=2)
    result = n_op.apply_series(phi)
    assert result.order == phi.order
    assert sp.expand(result.coeff(0) - 2 * X) == 0
    assert sp.expand(result.coeff(1) - 1) == 0
    assert sp.expand(result.coeff(2) - 0) == 0


def test_apply_series_second_derivative_in_u() -> None:
    """N(u) = u'': apply_series differentiates each q-coefficient in x twice."""
    expr = U(X).diff(X, 2)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X**3, X**2], order=1)
    result = n_op.apply_series(phi)
    assert result.order == phi.order
    assert sp.expand(result.coeff(0) - 6 * X) == 0
    assert sp.expand(result.coeff(1) - 2) == 0


def test_apply_series_u_times_u_prime_explicit_cauchy() -> None:
    """N(u) = u·u': result.coeff(k) == sum_{i+j=k} phi.coeff(i) * d/dx phi.coeff(j)."""
    expr = U(X) * U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X**2, X, sp.Integer(5)], order=2)
    result = n_op.apply_series(phi)
    for k in range(phi.order + 1):
        expected: sp.Expr = sp.Integer(0)
        for i in range(k + 1):
            expected = expected + phi.coeff(i) * sp.diff(phi.coeff(k - i), X)
        assert sp.expand(result.coeff(k) - expected) == 0


def test_apply_series_sum_of_u_and_u_prime() -> None:
    """N(u) = u + u': result == phi + phi.map_coeffs(d/dx) coefficient-wise."""
    expr = U(X) + U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X**2, X], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - (X**2 + 2 * X)) == 0
    assert sp.expand(result.coeff(1) - (X + 1)) == 0


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_first_derivative_matches_map_coeffs(phi: QSeries) -> None:
    """N(u) = u': apply_series(phi) agrees with phi.map_coeffs(d/dx)."""
    expr = U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_series(phi)
    expected = phi.map_coeffs(lambda c: sp.diff(c, X))
    assert result.order == expected.order
    for k in range(phi.order + 1):
        assert sp.expand(result.coeff(k) - expected.coeff(k)) == 0


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_u_times_u_prime_matches_cauchy(phi: QSeries) -> None:
    """N(u) = u·u': result == (phi * phi.map_coeffs(d/dx)).trunc(phi.order)."""
    expr = U(X) * U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    result = n_op.apply_series(phi)
    phi_prime = phi.map_coeffs(lambda c: sp.diff(c, X))
    expected = (phi * phi_prime).trunc(phi.order)
    for k in range(phi.order + 1):
        assert sp.expand(result.coeff(k) - expected.coeff(k)) == 0


# --- Stage 3d: transcendental + non-polynomial rejection -----------------


def test_apply_series_exp_of_u_rejected() -> None:
    """exp(u) requires formal-series composition; compiler raises."""
    expr = sp.exp(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, sp.Integer(1)], order=1)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_cos_of_u_rejected() -> None:
    """cos(u) requires formal-series composition; compiler raises."""
    expr = sp.cos(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, sp.Integer(1)], order=1)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_log_of_u_rejected() -> None:
    """log(u) requires formal-series composition; compiler raises."""
    expr = sp.log(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, sp.Integer(1)], order=1)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_sqrt_of_u_rejected() -> None:
    """sqrt(u) is Pow(u, 1/2); the compiler only handles non-negative integer powers."""
    expr = sp.sqrt(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_reciprocal_of_u_rejected() -> None:
    """1/u is Pow(u, -1); the compiler does not implement series inversion."""
    expr = sp.Integer(1) / U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_error_message_names_offending_subexpression() -> None:
    """The NotImplementedError carries the rejected subexpression in its message."""
    expr = sp.sin(U(X)) + U(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X], order=0)
    with pytest.raises(NotImplementedError) as exc_info:
        n_op.apply_series(phi)
    assert "sin(u(x))" in str(exc_info.value)


def test_apply_series_error_message_names_sympy_type() -> None:
    """The error message includes the sympy node type for debugging."""
    expr = sp.exp(U(X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError) as exc_info:
        n_op.apply_series(phi)
    assert "exp" in str(exc_info.value)


def test_apply_series_derivative_of_non_u_rejected() -> None:
    """Derivative(u(x)*x, x) is not Derivative(u(x), x); compiler rejects it.

    Sympy keeps this as an unevaluated Derivative whose `expr` is the
    product u(x)*x, not u(x) alone. The compiler is not a product-rule
    engine and surfaces a NotImplementedError; the user should write
    out the product rule explicitly (u'·x + u).
    """
    expr = sp.Derivative(U(X) * X, X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, sp.Integer(1)], order=1)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)
