"""Truncated formal power series with sympy coefficients."""

from collections.abc import Callable, Sequence

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

    @classmethod
    def constant(cls, expr: sp.Expr, order: int) -> "QSeries":
        """Lift an x-only expression to a constant series: coeff(0)=expr, rest=0."""
        return cls([expr], order=order)

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

    def __mul__(self, other: "sp.Expr | int | QSeries") -> "QSeries":
        if isinstance(other, QSeries):
            return self._cauchy(other)
        scalar = sp.sympify(other)
        return QSeries([scalar * c for c in self._coeffs], order=self._order)

    def __rmul__(self, scalar: sp.Expr | int) -> "QSeries":
        return self.__mul__(scalar)

    def _cauchy(self, other: "QSeries") -> "QSeries":
        order = self._order + other._order
        coeffs: list[sp.Expr] = []
        for k in range(order + 1):
            term: sp.Expr = sp.Integer(0)
            for j in range(k + 1):
                term = term + self.coeff(j) * other.coeff(k - j)
            coeffs.append(term)
        return QSeries(coeffs, order=order)

    def trunc(self, n: int) -> "QSeries":
        """Truncate to order min(n, self.order). Higher-order coefficients are dropped."""
        new_order = min(n, self._order)
        return QSeries(self._coeffs[: new_order + 1], order=new_order)

    def diff_q(self) -> "QSeries":
        """Differentiate with respect to q. Order lowers by 1 (clamped at 0)."""
        if self._order == 0:
            return QSeries.zero(0)
        new_coeffs = [sp.Integer(k + 1) * self.coeff(k + 1) for k in range(self._order)]
        return QSeries(new_coeffs, order=self._order - 1)

    def integrate_q(self) -> "QSeries":
        """Integrate with respect to q (constant of integration = 0). Order raises by 1."""
        new_coeffs: list[sp.Expr] = [sp.Integer(0)]
        for k in range(self._order + 1):
            new_coeffs.append(self.coeff(k) / sp.Integer(k + 1))
        return QSeries(new_coeffs, order=self._order + 1)

    def map_coeffs(self, f: Callable[[sp.Expr], sp.Expr]) -> "QSeries":
        """Apply f to every coefficient. Functor map; q-order is preserved."""
        new_coeffs = [f(self.coeff(k)) for k in range(self._order + 1)]
        return QSeries(new_coeffs, order=self._order)
