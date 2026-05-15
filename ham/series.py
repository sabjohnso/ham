"""Truncated formal power series with substrate-parametrised coefficients.

`Series[C]` is generic over the coefficient type C; the per-substrate
operations (zero, one, lift_xonly, diff_x, integrate_x) are supplied
via a `Backend[C]` instance. The series itself is held to a fixed
truncation order; queries for coefficients beyond the order return
`backend.zero()`.

`QSeries` is the sympy-baked back-compat shim that pre-binds
`Backend[sp.Expr] = SympyBackend(sp.Symbol("x"))` so callers in
NonlinearOperator, the deformation builder, the solver loop, and the
existing test suites continue to work without threading a backend
through every constructor. New callers should construct `Series[C]`
directly with an explicit backend; the QSeries shim retires once
those callers migrate (S2 onwards).
"""

from collections.abc import Callable, Sequence
from typing import Any, Protocol, Self

import sympy as sp

from ham.backend import Backend, SympyBackend


class _SupportsCoefficientArith(Protocol):
    """Per-element arithmetic the coefficient substrate must support.

    Both `sympy.Expr` and `numpy.ndarray` satisfy this structurally;
    declaring it explicitly lets `Series[C]` typecheck under mypy
    strict by giving the TypeVar a bound that names the operations
    the methods actually use. Return types are `Any` because the
    concrete result type (sympy.Expr, numpy.ndarray) collapses back
    to `C` via Any-bivariance at the call sites.
    """

    def __add__(self, other: object, /) -> Any: ...
    def __neg__(self) -> Any: ...
    def __mul__(self, other: object, /) -> Any: ...
    def __rmul__(self, other: object, /) -> Any: ...
    def __truediv__(self, other: object, /) -> Any: ...


class Series[C: _SupportsCoefficientArith]:
    """A truncated formal power series in q with coefficients in C.

    The substrate `C` (typically `sympy.Expr` for symbolic HAM or
    `numpy.ndarray` of grid values for SHAM) is bridged by a
    `Backend[C]`: identities (`zero`, `one`), the lift from x-only
    sympy expressions, and the x-calculus. Out-of-range coefficient
    queries return `backend.zero()` rather than a hard-coded
    `sp.Integer(0)`, which is what makes the type substrate-agnostic.
    """

    def __init__(
        self,
        coeffs: Sequence[C],
        order: int,
        *,
        backend: Backend[C],
    ) -> None:
        self._coeffs: list[C] = list(coeffs)
        self._order: int = order
        self._backend: Backend[C] = backend

    @property
    def backend(self) -> Backend[C]:
        return self._backend

    @classmethod
    def zero(cls, order: int, *, backend: Backend[C]) -> Self:
        """The additive-identity series at the given truncation order."""
        return cls([], order=order, backend=backend)

    @classmethod
    def constant(cls, value: C, order: int, *, backend: Backend[C]) -> Self:
        """Lift a single coefficient to a constant series: coeff(0)=value, rest=zero."""
        return cls([value], order=order, backend=backend)

    @property
    def order(self) -> int:
        return self._order

    def coeff(self, k: int) -> C:
        """Return the coefficient of q^k (`backend.zero()` outside the stored range)."""
        if k < 0 or k > self._order or k >= len(self._coeffs):
            return self._backend.zero()
        return self._coeffs[k]

    def __add__(self, other: "Series[C]") -> Self:
        order = max(self._order, other._order)
        coeffs = [self.coeff(k) + other.coeff(k) for k in range(order + 1)]
        return type(self)(coeffs, order=order, backend=self._backend)

    def __neg__(self) -> Self:
        return type(self)([-c for c in self._coeffs], order=self._order, backend=self._backend)

    def __mul__(self, other: "C | int | Series[C]") -> Self:
        if isinstance(other, Series):
            return self._cauchy(other)
        return type(self)(
            [other * c for c in self._coeffs],
            order=self._order,
            backend=self._backend,
        )

    def __rmul__(self, scalar: "C | int") -> Self:
        return self.__mul__(scalar)

    def _cauchy(self, other: "Series[C]") -> Self:
        order = self._order + other._order
        coeffs: list[C] = []
        for k in range(order + 1):
            term: C = self._backend.zero()
            for j in range(k + 1):
                term = term + self.coeff(j) * other.coeff(k - j)
            coeffs.append(term)
        return type(self)(coeffs, order=order, backend=self._backend)

    def trunc(self, n: int) -> Self:
        """Truncate to order min(n, self.order). Higher-order coefficients are dropped."""
        new_order = min(n, self._order)
        return type(self)(self._coeffs[: new_order + 1], order=new_order, backend=self._backend)

    def diff_q(self) -> Self:
        """Differentiate with respect to q. Order lowers by 1 (clamped at 0)."""
        if self._order == 0:
            return type(self)([], order=0, backend=self._backend)
        new_coeffs = [(k + 1) * self.coeff(k + 1) for k in range(self._order)]
        return type(self)(new_coeffs, order=self._order - 1, backend=self._backend)

    def integrate_q(self) -> Self:
        """Integrate with respect to q (constant of integration = 0). Order raises by 1."""
        new_coeffs: list[C] = [self._backend.zero()]
        for k in range(self._order + 1):
            new_coeffs.append(self.coeff(k) / (k + 1))
        return type(self)(new_coeffs, order=self._order + 1, backend=self._backend)

    def map_coeffs(self, f: Callable[[C], C]) -> Self:
        """Apply f to every coefficient. Functor map; q-order is preserved."""
        new_coeffs = [f(self.coeff(k)) for k in range(self._order + 1)]
        return type(self)(new_coeffs, order=self._order, backend=self._backend)


_DEFAULT_QSERIES_INDEP = sp.Symbol("x")
_DEFAULT_QSERIES_BACKEND: Backend[sp.Expr] = SympyBackend(_DEFAULT_QSERIES_INDEP)


class QSeries(Series[sp.Expr]):
    """Sympy-backed Series with the original two-argument call surface.

    Defaults `backend` to `SympyBackend(sp.Symbol("x"))` so legacy
    callers (NonlinearOperator, the deformation builder, the solver
    loop, the existing test suites) continue to call `QSeries(coeffs,
    order)` and `QSeries.zero(order)` unchanged. The `backend` kwarg
    is *accepted* with a default — this is an LSP-relaxation of the
    parent's required-kwarg signature, which lets `type(self)`-based
    construction inside parent methods continue to work when self is
    a QSeries.
    """

    def __init__(
        self,
        coeffs: Sequence[sp.Expr],
        order: int,
        *,
        backend: Backend[sp.Expr] = _DEFAULT_QSERIES_BACKEND,
    ) -> None:
        super().__init__(coeffs, order, backend=backend)

    @classmethod
    def zero(
        cls,
        order: int,
        *,
        backend: Backend[sp.Expr] = _DEFAULT_QSERIES_BACKEND,
    ) -> Self:
        return cls([], order=order, backend=backend)

    @classmethod
    def constant(
        cls,
        value: sp.Expr,
        order: int,
        *,
        backend: Backend[sp.Expr] = _DEFAULT_QSERIES_BACKEND,
    ) -> Self:
        return cls([value], order=order, backend=backend)
