"""Hypothesis strategies for QSeries property tests.

Kept separate from the test modules so that strategies can be reused across
properties without circular imports or test-class hierarchies.

Each QSeries-producing strategy accepts a `backend` kwarg defaulting to
`DEFAULT_BACKEND = SympyBackend(X)`. The default keeps existing
`tests/test_qseries.py` tests unchanged; the kwarg lets S5b's
parametrised property tests re-use the same draws under a
`SpectralBackend` (float or sympy scalar) without duplicating the
strategy logic.
"""

import sympy as sp
from ham.backend import Backend, SympyBackend
from ham.series import QSeries
from hypothesis import strategies as st

MAX_ORDER = 6
MAX_COEFF_DEGREE = 3
COEFF_MIN = -100
COEFF_MAX = 100

X = sp.Symbol("x")
"""The independent variable used by all coefficient strategies."""

DEFAULT_BACKEND: Backend[sp.Expr] = SympyBackend(X)
"""Default backend threaded into QSeries strategies — sympy keyed on X."""


@st.composite
def polynomial_in_x(draw: st.DrawFn, max_degree: int = MAX_COEFF_DEGREE) -> sp.Expr:
    """Draw a polynomial in `X` with sympy.Integer coefficients."""
    raw = draw(
        st.lists(
            st.integers(min_value=COEFF_MIN, max_value=COEFF_MAX),
            min_size=1,
            max_size=max_degree + 1,
        )
    )
    expr: sp.Expr = sp.Integer(0)
    for k, c in enumerate(raw):
        expr = expr + sp.Integer(c) * X**k
    return expr


@st.composite
def qseries_at_order(
    draw: st.DrawFn,
    order: int,
    *,
    backend: Backend[sp.Expr] = DEFAULT_BACKEND,
) -> QSeries:
    """Draw a QSeries with sympy.Integer coefficients at the given order."""
    raw = draw(
        st.lists(
            st.integers(min_value=COEFF_MIN, max_value=COEFF_MAX),
            min_size=0,
            max_size=order + 1,
        )
    )
    return QSeries([sp.Integer(c) for c in raw], order=order, backend=backend)


@st.composite
def one_qseries(
    draw: st.DrawFn,
    *,
    backend: Backend[sp.Expr] = DEFAULT_BACKEND,
) -> QSeries:
    """Draw a single QSeries at a randomly chosen order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    return draw(qseries_at_order(order=order, backend=backend))


@st.composite
def two_qseries_same_order(
    draw: st.DrawFn,
    *,
    backend: Backend[sp.Expr] = DEFAULT_BACKEND,
) -> tuple[QSeries, QSeries]:
    """Draw two QSeries sharing the same truncation order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    a = draw(qseries_at_order(order=order, backend=backend))
    b = draw(qseries_at_order(order=order, backend=backend))
    return a, b


@st.composite
def three_qseries_same_order(
    draw: st.DrawFn,
    *,
    backend: Backend[sp.Expr] = DEFAULT_BACKEND,
) -> tuple[QSeries, QSeries, QSeries]:
    """Draw three QSeries sharing the same truncation order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    a = draw(qseries_at_order(order=order, backend=backend))
    b = draw(qseries_at_order(order=order, backend=backend))
    c = draw(qseries_at_order(order=order, backend=backend))
    return a, b, c


@st.composite
def qseries_polynomial_coeffs(
    draw: st.DrawFn,
    *,
    backend: Backend[sp.Expr] = DEFAULT_BACKEND,
) -> QSeries:
    """Draw a QSeries whose coefficients are polynomials in X."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    coeffs = draw(
        st.lists(
            polynomial_in_x(),
            min_size=0,
            max_size=order + 1,
        )
    )
    return QSeries(coeffs, order=order, backend=backend)
