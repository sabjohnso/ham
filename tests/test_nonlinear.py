"""Property tests for NonlinearOperator — the HAM nonlinear operator N.

Stage 3a (this file, initial slice): scalar evaluation only. N wraps a
sympy expression in u(x), u'(x), ... and substitutes a concrete function
for u, evaluating derivatives. apply_series (substituting a QSeries for u,
polynomial regime) is 3b/3c. Transcendental rejection is 3d.
"""

import pytest
import sympy as sp
from ham.backend import SympyBackend
from ham.nonlinear import NonlinearOperator
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import MAX_ORDER, X, polynomial_in_x, qseries_polynomial_coeffs

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


@given(
    phi=qseries_polynomial_coeffs(),
    perturbation=polynomial_in_x(),
    k_raw=st.integers(min_value=0, max_value=MAX_ORDER),
)
def test_apply_series_causality_in_q(phi: QSeries, perturbation: sp.Expr, k_raw: int) -> None:
    """For every k in [0, phi.order], perturbing phi.coeff(k) leaves
    N[phi].coeff(j) invariant for all j < k.

    The strict form of Cauchy causality: high-q content cannot bleed
    downwards into any lower coefficient of N[phi]. The compiler is
    built on Cauchy products, coefficient-wise differentiation, and
    coefficient-wise integration — all of which preserve causality —
    so this property should hold for every polynomial-in-u N the
    compiler handles. The chosen N exercises Cauchy product (u^2),
    product with derivative (u·u'), and pure derivative (u'), so all
    three causality paths are covered by one Hypothesis draw.
    """
    k = min(k_raw, phi.order)
    expr = U(X) ** 2 + U(X) * U(X).diff(X) + U(X).diff(X)
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)

    result_orig = n_op.apply_series(phi)

    new_coeffs = [phi.coeff(i) for i in range(phi.order + 1)]
    new_coeffs[k] = new_coeffs[k] + perturbation
    phi_perturbed = QSeries(new_coeffs, order=phi.order)
    result_perturbed = n_op.apply_series(phi_perturbed)

    for j in range(k):
        assert sp.expand(result_orig.coeff(j) - result_perturbed.coeff(j)) == 0


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


# --- Stage 9a: apply_series with definite integrals 0..indep --------------


def test_apply_series_integral_of_u_only() -> None:
    """N(u) = integral 0..t u(s) ds: each q-coefficient gets integrated in x.

    phi = QSeries([1, x, x^2], 2)  ⇒
    integral coeffs: [int_0^t 1 dt' = t, int_0^t t' dt' = t^2/2, int_0^t t'^2 dt' = t^3/3]
    """
    s = sp.Symbol("s")
    expr = sp.Integral(U(s), (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1), X, X**2], order=2)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - X) == 0
    assert sp.expand(result.coeff(1) - X**2 / sp.Integer(2)) == 0
    assert sp.expand(result.coeff(2) - X**3 / sp.Integer(3)) == 0


def test_apply_series_integral_of_u_squared() -> None:
    """N(u) = integral 0..t u(s)^2 ds: integrand compiled to Cauchy product first."""
    s = sp.Symbol("s")
    expr = sp.Integral(U(s) ** 2, (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1), X], order=1)
    result = n_op.apply_series(phi)
    # phi^2 (eager-truncated to order 1): [q^0] = 1; [q^1] = 2*1*x = 2x
    # Integrate: [q^0] = int_0^t 1 dt' = t; [q^1] = int_0^t 2 t' dt' = t^2
    assert sp.expand(result.coeff(0) - X) == 0
    assert sp.expand(result.coeff(1) - X**2) == 0


def test_apply_series_outer_u_times_integral() -> None:
    """N(u) = u(t) * integral 0..t u(s) ds: outer u multiplies the integral QSeries.

    phi = QSeries([1, 2], 1)
    int phi = QSeries([t, 2t], 1)
    phi * (int phi) (eager-trunc to order 1):
      [q^0]: 1*t = t
      [q^1]: 1*2t + 2*t = 4t
    """
    s = sp.Symbol("s")
    expr = U(X) * sp.Integral(U(s), (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1), sp.Integer(2)], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - X) == 0
    assert sp.expand(result.coeff(1) - 4 * X) == 0


def test_apply_series_integral_constant_compatibility() -> None:
    """For a u-free integrand the integral is just sp.integrate at coeff(0)."""
    s = sp.Symbol("s")
    expr = sp.Integral(s, (s, 0, X))  # integral 0..t s ds = t^2/2
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([X, X**2], order=1)
    result = n_op.apply_series(phi)
    assert sp.expand(result.coeff(0) - X**2 / sp.Integer(2)) == 0
    assert result.coeff(1) == 0


def test_apply_series_integral_rejects_non_zero_lower_bound() -> None:
    """Only integrals starting at 0 are supported."""
    s = sp.Symbol("s")
    expr = sp.Integral(U(s), (s, sp.Integer(1), X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError, match="0"):
        n_op.apply_series(phi)


def test_apply_series_integral_rejects_upper_bound_not_indep() -> None:
    """Only integrals ending at the indep variable are supported."""
    s = sp.Symbol("s")
    expr = sp.Integral(U(s), (s, 0, sp.Integer(1)))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_double_integral_rejected() -> None:
    """Nested integrals or multi-variable integrals raise NotImplementedError."""
    s1 = sp.Symbol("s1")
    s2 = sp.Symbol("s2")
    expr = sp.Integral(U(s1), (s1, 0, s2), (s2, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError):
        n_op.apply_series(phi)


def test_apply_series_integral_rejects_integrand_with_indep_dependence() -> None:
    """Integrand with explicit X-dependence outside u(s) raises.

    `Integral(X * u(s), (s, 0, X))` semantically means `X · ∫_0^X u(s) ds`,
    but the compiler's `subs(s, X)` step would silently turn it into
    `∫_0^X s · u(s) ds` — a different function. Guarding against this
    case is item 1 of the post-review plan.
    """
    s = sp.Symbol("s")
    expr = sp.Integral(X * U(s), (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError, match="independent variable"):
        n_op.apply_series(phi)


def test_apply_series_integral_rejects_integrand_with_indep_inside_transcendental() -> None:
    """`Integral(cos(X) · u(s), (s, 0, X))` also raises.

    Any free occurrence of the independent variable in the integrand
    is enough to trip the guard, regardless of how the variable appears.
    """
    s = sp.Symbol("s")
    expr = sp.Integral(sp.cos(X) * U(s), (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError, match="independent variable"):
        n_op.apply_series(phi)


def test_apply_series_integral_error_message_names_indep_variable() -> None:
    """The raised error message includes the independent variable's name."""
    s = sp.Symbol("s")
    expr = sp.Integral(X**2 * U(s), (s, 0, X))
    n_op = NonlinearOperator(expr=expr, dependent=U, indep=X)
    phi = QSeries([sp.Integer(1)], order=0)
    with pytest.raises(NotImplementedError) as exc_info:
        n_op.apply_series(phi)
    assert "x" in str(exc_info.value)


# --- Stage S2: apply_series dispatches through phi.backend ---------------


def test_apply_series_propagates_phi_backend() -> None:
    """apply_series returns a Series whose backend is phi.backend.

    Until S2, _compile hard-coded `QSeries.zero` / `QSeries.constant`
    (carrying the default `SympyBackend` instance) for every internal
    construction, so the result's backend was the default — not whatever
    backend the caller's `phi` carried. After S2 the construction
    threads `phi.backend` through every leaf, which is the precondition
    for SHAM in S7 where `phi` will live over a spectral backend.
    """
    alt_backend = SympyBackend(X)  # same field semantics, distinct identity
    phi = QSeries([X, X**2], order=1, backend=alt_backend)

    expr_in_u = U(X) ** 2 + U(X).diff(X)
    n_op = NonlinearOperator(expr=expr_in_u, dependent=U, indep=X)

    result = n_op.apply_series(phi)

    assert result.backend is alt_backend


@given(phi=qseries_polynomial_coeffs())
def test_apply_series_commutes_with_apply_scalar_polynomial_n(phi: QSeries) -> None:
    """[q^k] N.apply_series(phi) == [q^k] N.apply_scalar(Σ phi.coeff(j) q^j).

    Liao's commuting diagram for polynomial N: applying N coefficient-by-
    coefficient through `apply_series` equals applying N scalar-wise to
    phi-as-polynomial-in-q and projecting onto q^k. This is the property
    that makes `apply_series` a faithful evaluation of N on the homotopy
    series. Promoted to an explicit test so S5b's parametrised property
    suite can re-use the law against any backend whose `apply_scalar`
    path is unavailable (or against the sympy-scalar SpectralBackend
    that S5b will introduce).
    """
    q = sp.Symbol("q")
    expr_in_u = U(X) ** 2 + U(X).diff(X) + sp.Integer(3)
    n_op = NonlinearOperator(expr=expr_in_u, dependent=U, indep=X)

    u_in_x_and_q = sum(
        (phi.coeff(k) * q**k for k in range(phi.order + 1)),
        sp.Integer(0),
    )
    expected_in_q = sp.expand(n_op.apply_scalar(u_in_x_and_q))
    actual_series = n_op.apply_series(phi)

    for k in range(phi.order + 1):
        expected_q_k = expected_in_q.coeff(q, k)
        assert sp.expand(actual_series.coeff(k) - expected_q_k) == 0
