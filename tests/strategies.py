"""Hypothesis strategies for QSeries property tests.

Kept separate from the test modules so that strategies can be reused across
properties without circular imports or test-class hierarchies.
"""

import sympy as sp
from ham.series import QSeries
from hypothesis import strategies as st

MAX_ORDER = 6
COEFF_MIN = -100
COEFF_MAX = 100


@st.composite
def qseries_at_order(draw: st.DrawFn, order: int) -> QSeries:
    """Draw a QSeries with sympy.Integer coefficients at the given order."""
    raw = draw(
        st.lists(
            st.integers(min_value=COEFF_MIN, max_value=COEFF_MAX),
            min_size=0,
            max_size=order + 1,
        )
    )
    return QSeries([sp.Integer(c) for c in raw], order=order)


@st.composite
def one_qseries(draw: st.DrawFn) -> QSeries:
    """Draw a single QSeries at a randomly chosen order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    return draw(qseries_at_order(order=order))


@st.composite
def two_qseries_same_order(draw: st.DrawFn) -> tuple[QSeries, QSeries]:
    """Draw two QSeries sharing the same truncation order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    a = draw(qseries_at_order(order=order))
    b = draw(qseries_at_order(order=order))
    return a, b


@st.composite
def three_qseries_same_order(
    draw: st.DrawFn,
) -> tuple[QSeries, QSeries, QSeries]:
    """Draw three QSeries sharing the same truncation order."""
    order = draw(st.integers(min_value=0, max_value=MAX_ORDER))
    a = draw(qseries_at_order(order=order))
    b = draw(qseries_at_order(order=order))
    c = draw(qseries_at_order(order=order))
    return a, b, c
