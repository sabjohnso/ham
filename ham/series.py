"""Truncated formal power series with sympy coefficients."""

from collections.abc import Sequence

import sympy as sp


class QSeries:
    """A truncated formal power series in a single indeterminate.

    Coefficients are sympy expressions (typically in the independent
    variable x). The series is held to a fixed truncation order; queries
    for coefficients beyond the order return zero.
    """

    def __init__(self, coeffs: Sequence[sp.Expr], order: int) -> None:
        self._coeffs: list[sp.Expr] = list(coeffs)
        self._order: int = order

    @classmethod
    def zero(cls, order: int) -> "QSeries":
        """Return the additive-identity series at the given truncation order."""
        return cls([], order=order)

    @property
    def order(self) -> int:
        return self._order

    def coeff(self, k: int) -> sp.Expr:
        """Return the coefficient of q^k (zero outside the stored range)."""
        if k < 0 or k > self._order or k >= len(self._coeffs):
            return sp.Integer(0)
        return self._coeffs[k]

    def __add__(self, other: "QSeries") -> "QSeries":
        order = max(self._order, other._order)
        coeffs = [self.coeff(k) + other.coeff(k) for k in range(order + 1)]
        return QSeries(coeffs, order=order)

    def __neg__(self) -> "QSeries":
        return QSeries([-c for c in self._coeffs], order=self._order)
