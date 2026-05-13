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
    """L = d/dx with u(0) = 0, no inverter — exercises the dsolve default."""
    return LinearOperator(
        var=X,
        action=_diff_x,
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )


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


def test_explicit_inverter_overrides_dsolve_default() -> None:
    """A user-supplied inverter takes precedence over the dsolve default."""
    calls: list[sp.Expr] = []

    def fake_inverter(rhs: sp.Expr) -> sp.Expr:
        calls.append(rhs)
        return sp.Integer(42)

    op = LinearOperator(var=X, action=_diff_x, inverter=fake_inverter)
    result = op.invert(sp.Integer(7))
    assert result == sp.Integer(42)
    assert calls == [sp.Integer(7)]


def test_dsolve_default_matches_antiderivative_on_canonical_ivp() -> None:
    """The dsolve default and the hand-coded antiderivative agree on d/dx, u(0)=0."""
    rhs = X**3 + sp.Integer(2) * X + sp.Integer(5)
    op_default = _canonical_ivp()
    explicit = antiderivative(X, sp.Integer(0))(rhs)
    assert sp.expand(op_default.invert(rhs) - explicit) == 0


def test_dsolve_default_handles_second_order_ivp() -> None:
    """L = d^2/dx^2 with u(0)=u'(0)=0; invert(x) should give x^3/6."""
    op = LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X, 2),
        bcs=(
            BoundaryCondition(point=sp.Integer(0), derivative_order=0),
            BoundaryCondition(point=sp.Integer(0), derivative_order=1),
        ),
    )
    assert sp.expand(op.invert(X) - X**3 / sp.Integer(6)) == 0


def test_dsolve_default_handles_first_order_linear_ode() -> None:
    """L = d/dx + I with u(0)=0; invert(exp(x)) should give (exp(x) - exp(-x))/2."""
    op = LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X) + e,
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )
    rhs = sp.exp(X)
    expected = (sp.exp(X) - sp.exp(-X)) / sp.Integer(2)
    assert sp.simplify(op.invert(rhs) - expected) == 0


def test_dsolve_default_honors_nonzero_bc_value() -> None:
    """L = d/dx with u(0)=5, rhs=0; invert should give the constant 5."""
    op = LinearOperator(
        var=X,
        action=_diff_x,
        bcs=(
            BoundaryCondition(
                point=sp.Integer(0),
                derivative_order=0,
                value=sp.Integer(5),
            ),
        ),
    )
    assert op.invert(sp.Integer(0)) == sp.Integer(5)


def test_dsolve_default_accepts_asymptotic_bc_at_infinity() -> None:
    """BoundaryCondition.point may be sp.oo; the dsolve inverter handles it.

    For the auxiliary operator L = d^3/dx^3 - d/dx with the three BCs
    u(0) = 0, u'(0) = 0, u'(∞) = 0, the homogeneous solution space is
    spanned by {1, exp(x), exp(-x)} with the asymptotic BC ruling out
    exp(x). Inverting against rhs = exp(-2x) gives the closed-form
    -1/6 + exp(-x)/3 - exp(-2x)/6, verifiable by direct substitution
    (and by Liao-style exponential-basis HAM, the use case this
    capability enables).
    """
    op = LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X, 3) - sp.diff(e, X),
        bcs=(
            BoundaryCondition(point=sp.Integer(0), derivative_order=0),
            BoundaryCondition(point=sp.Integer(0), derivative_order=1),
            BoundaryCondition(point=sp.oo, derivative_order=1),
        ),
    )
    rhs = sp.exp(-2 * X)
    expected = -sp.Rational(1, 6) + sp.exp(-X) / sp.Integer(3) - sp.exp(-2 * X) / sp.Integer(6)
    assert sp.simplify(op.invert(rhs) - expected) == 0
