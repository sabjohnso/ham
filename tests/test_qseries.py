"""Property tests for QSeries — the truncated formal power series in q.

Stage 1 of the HAM build (see PLAN.org). Each property here corresponds to
an algebraic law that QSeries must satisfy.
"""

import sympy as sp
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import (
    one_qseries,
    three_qseries_same_order,
    two_qseries_same_order,
)


@given(raw_coeffs=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=8))
def test_construction_round_trip(raw_coeffs: list[int]) -> None:
    """coeff(k) returns the k-th coefficient supplied at construction."""
    coeffs = [sp.Integer(c) for c in raw_coeffs]
    series = QSeries(coeffs, order=len(coeffs) - 1)
    for k, c in enumerate(coeffs):
        assert series.coeff(k) == c


# --- Additive abelian group laws -------------------------------------------


@given(pair=two_qseries_same_order())
def test_addition_is_coefficientwise(pair: tuple[QSeries, QSeries]) -> None:
    """[q^k](a + b) == [q^k]a + [q^k]b for all k <= order."""
    a, b = pair
    s = a + b
    for k in range(a.order + 1):
        assert s.coeff(k) == a.coeff(k) + b.coeff(k)


@given(pair=two_qseries_same_order())
def test_addition_is_commutative(pair: tuple[QSeries, QSeries]) -> None:
    """a + b == b + a, verified coefficient-wise."""
    a, b = pair
    for k in range(a.order + 1):
        assert (a + b).coeff(k) == (b + a).coeff(k)


@given(triple=three_qseries_same_order())
def test_addition_is_associative(triple: tuple[QSeries, QSeries, QSeries]) -> None:
    """(a + b) + c == a + (b + c), verified coefficient-wise."""
    a, b, c = triple
    for k in range(a.order + 1):
        assert ((a + b) + c).coeff(k) == (a + (b + c)).coeff(k)


@given(a=one_qseries())
def test_zero_is_additive_identity(a: QSeries) -> None:
    """a + 0 == a == 0 + a, verified coefficient-wise."""
    zero = QSeries.zero(a.order)
    for k in range(a.order + 1):
        assert (a + zero).coeff(k) == a.coeff(k)
        assert (zero + a).coeff(k) == a.coeff(k)


@given(a=one_qseries())
def test_negation_is_additive_inverse(a: QSeries) -> None:
    """a + (-a) == 0, verified coefficient-wise."""
    s = a + (-a)
    for k in range(a.order + 1):
        assert s.coeff(k) == sp.Integer(0)
