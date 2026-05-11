"""Property tests for NonlinearOperator — the HAM nonlinear operator N.

Stage 3a (this file, initial slice): scalar evaluation only. N wraps a
sympy expression in u(x), u'(x), ... and substitutes a concrete function
for u, evaluating derivatives. apply_series (substituting a QSeries for u,
polynomial regime) is 3b/3c. Transcendental rejection is 3d.
"""

import sympy as sp
from ham.nonlinear import NonlinearOperator
from hypothesis import given

from tests.strategies import X, polynomial_in_x

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
