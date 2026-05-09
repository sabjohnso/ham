"""Property tests for QSeries — the truncated formal power series in q.

Stage 1 of the HAM build (see PLAN.org). Each property here corresponds to
an algebraic law that QSeries must satisfy.
"""

import sympy as sp
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st


@given(raw_coeffs=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=8))
def test_construction_round_trip(raw_coeffs: list[int]) -> None:
    """coeff(k) returns the k-th coefficient supplied at construction."""
    coeffs = [sp.Integer(c) for c in raw_coeffs]
    series = QSeries(coeffs, order=len(coeffs) - 1)
    for k, c in enumerate(coeffs):
        assert series.coeff(k) == c
