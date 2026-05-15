"""Tests for Series[C] — the substrate-generic version of QSeries.

QSeries continues to exist as the sympy-baked subclass and is tested
by tests/test_qseries.py; this file tests the generic API and the new
backend-injection points (coeff(out-of-range) returns backend.zero(),
factories accept an explicit backend kwarg, arithmetic propagates the
backend, etc.).

For S1 only the sympy backend exists; S5b will parametrise these
tests across SympyBackend / SpectralBackend(float) /
SpectralBackend(sp.Expr).
"""

from dataclasses import replace

import sympy as sp
from ham.backend import Backend, SympyBackend
from ham.series import QSeries, Series

from tests.strategies import X

SYMPY: Backend[sp.Expr] = SympyBackend(X)


def _sentinel_zero_backend(sentinel: sp.Expr) -> Backend[sp.Expr]:
    """A sympy-flavoured Backend whose zero() returns a detectable marker."""
    return replace(SYMPY, zero=lambda: sentinel)


def test_series_constructed_with_backend() -> None:
    """Series accepts a backend kwarg and stores it on the instance."""
    s = Series([sp.Integer(2), sp.Integer(3)], order=1, backend=SYMPY)
    assert s.order == 1
    assert s.coeff(0) == sp.Integer(2)
    assert s.coeff(1) == sp.Integer(3)
    assert s.backend is SYMPY


def test_coeff_out_of_range_returns_backend_zero() -> None:
    """coeff(k) outside the stored range delegates to backend.zero()."""
    sentinel = sp.Symbol("ZERO_SENTINEL")
    stub = _sentinel_zero_backend(sentinel)
    s = Series([sp.Integer(5)], order=3, backend=stub)
    assert s.coeff(0) == sp.Integer(5)
    assert s.coeff(1) == sentinel
    assert s.coeff(2) == sentinel
    assert s.coeff(3) == sentinel
    assert s.coeff(99) == sentinel
    assert s.coeff(-1) == sentinel


def test_zero_factory_uses_supplied_backend() -> None:
    """Series.zero(order, backend=...) carries the supplied backend through."""
    sentinel = sp.Symbol("ZERO_SENTINEL")
    stub = _sentinel_zero_backend(sentinel)
    s = Series.zero(order=2, backend=stub)
    assert s.backend is stub
    assert s.coeff(0) == sentinel
    assert s.coeff(1) == sentinel
    assert s.coeff(2) == sentinel


def test_constant_factory_uses_supplied_backend() -> None:
    """Series.constant(c, order, backend=...) stores c at slot 0, backend.zero() elsewhere."""
    sentinel = sp.Symbol("ZERO_SENTINEL")
    stub = _sentinel_zero_backend(sentinel)
    s = Series.constant(sp.Integer(7), order=2, backend=stub)
    assert s.coeff(0) == sp.Integer(7)
    assert s.coeff(1) == sentinel
    assert s.coeff(2) == sentinel


def test_addition_propagates_backend() -> None:
    """Adding two Series preserves the (shared) backend on the result."""
    a = Series([sp.Integer(1), sp.Integer(2)], order=1, backend=SYMPY)
    b = Series([sp.Integer(3), sp.Integer(4)], order=1, backend=SYMPY)
    s = a + b
    assert s.backend is SYMPY


def test_cauchy_product_propagates_backend() -> None:
    """Cauchy product preserves the (shared) backend on the result."""
    a = Series([sp.Integer(1), sp.Integer(2)], order=1, backend=SYMPY)
    b = Series([sp.Integer(3), sp.Integer(4)], order=1, backend=SYMPY)
    s = a * b
    assert s.backend is SYMPY


def test_diff_q_propagates_backend() -> None:
    """diff_q preserves the backend, including the order-0 fast path."""
    a = Series([sp.Integer(1), sp.Integer(2), sp.Integer(3)], order=2, backend=SYMPY)
    assert a.diff_q().backend is SYMPY
    # order-0 case takes a different code path (returns Series.zero(0, ...))
    z = Series([sp.Integer(5)], order=0, backend=SYMPY)
    assert z.diff_q().backend is SYMPY


def test_integrate_q_propagates_backend() -> None:
    """integrate_q preserves the backend."""
    a = Series([sp.Integer(1), sp.Integer(2)], order=1, backend=SYMPY)
    assert a.integrate_q().backend is SYMPY


def test_integrate_q_seeds_constant_with_backend_zero() -> None:
    """integrate_q sets coeff(0) to backend.zero(), not sp.Integer(0)."""
    sentinel = sp.Symbol("ZERO_SENTINEL")
    stub = _sentinel_zero_backend(sentinel)
    s = Series([sp.Integer(4)], order=0, backend=stub)
    integ = s.integrate_q()
    assert integ.coeff(0) == sentinel


def test_qseries_is_a_series_subclass() -> None:
    """QSeries remains a Series subclass — back-compat invariant for legacy callers."""
    q = QSeries([sp.Integer(1)], order=0)
    assert isinstance(q, Series)
