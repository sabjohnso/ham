"""Tests for SpectralBackend — Backend[np.ndarray] over a spectral grid.

S5b deliverable: `SpectralBackend(grid, indep, scalar)` is generic over
the per-element scalar ring. `scalar = "float"` gives classical SHAM
(numpy linear algebra, ℏ-sweeps external). `scalar = "sympy"` keeps ℏ
symbolic inside every grid entry (`ndarray[object]`), at the cost of a
`sp.Matrix.LUsolve` in `integrate_x`. Both share all of the Backend
contract; the linear solver and the array dtype are the only points of
variation. /Test-only/ stage — the spectral backend is not yet wired
into Series consumers (that happens in S7).
"""

import numpy as np
import pytest
import sympy as sp
from ham.grids import ChebGLGrid
from ham.spectral import SpectralBackend

X = sp.Symbol("x")


# --- Shape / dtype --------------------------------------------------------


def test_spectral_backend_float_zero_returns_float_ndarray() -> None:
    """zero() returns a float64 array of zeros with the grid's node count."""
    grid = ChebGLGrid(N=8)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    z = backend.zero()
    assert z.shape == (9,)
    assert z.dtype == np.float64
    assert np.all(z == 0.0)


def test_spectral_backend_float_one_returns_float_ndarray_of_ones() -> None:
    """one() returns a float64 array of ones with the grid's node count."""
    grid = ChebGLGrid(N=8)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    o = backend.one()
    assert o.shape == (9,)
    assert o.dtype == np.float64
    assert np.all(o == 1.0)


def test_spectral_backend_sympy_zero_is_object_array_of_sympy_zeros() -> None:
    """For scalar='sympy', zero() is an object array of sympy zeros."""
    grid = ChebGLGrid(N=4)
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    z = backend.zero()
    assert z.shape == (5,)
    assert z.dtype == object
    assert all(entry == sp.Integer(0) for entry in z)


def test_spectral_backend_rejects_unknown_scalar() -> None:
    """An unknown scalar tag is rejected at construction."""
    grid = ChebGLGrid(N=4)
    with pytest.raises(ValueError, match="scalar"):
        SpectralBackend(grid, indep=X, scalar="cursed")  # type: ignore[arg-type]


# --- lift_xonly -----------------------------------------------------------


def test_spectral_backend_float_lift_xonly_evaluates_at_grid_nodes() -> None:
    """lift_xonly(expr) evaluates expr at the grid nodes (float path)."""
    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    expr = X**2 + sp.Integer(3) * X
    grid_vals = backend.lift_xonly(expr)
    expected = grid.nodes**2 + 3 * grid.nodes
    np.testing.assert_allclose(grid_vals, expected, atol=1e-12)


def test_spectral_backend_float_lift_xonly_broadcasts_constant() -> None:
    """A constant (x-free) expression lifts to a full array of the constant value."""
    grid = ChebGLGrid(N=6)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    out = backend.lift_xonly(sp.Integer(7))
    assert out.shape == (7,)
    np.testing.assert_allclose(out, 7.0)


def test_spectral_backend_sympy_lift_xonly_preserves_non_x_symbols() -> None:
    """For scalar='sympy', lift_xonly substitutes x but keeps other symbols (e.g. hbar)."""
    hbar = sp.Symbol("hbar")
    grid = ChebGLGrid(N=4, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    expr = hbar * X**2
    out = backend.lift_xonly(expr)
    # Each entry should be hbar * (node_value)^2
    for idx, node in enumerate(grid.nodes):
        assert sp.expand(out[idx] - hbar * sp.Float(node) ** 2) == 0


# --- diff_x ---------------------------------------------------------------


def test_spectral_backend_float_diff_x_recovers_polynomial_derivative() -> None:
    """diff_x(p, 1) recovers p' to roundoff for polynomials the grid resolves."""
    grid = ChebGLGrid(N=16)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    p_vals = grid.nodes**3 - 2 * grid.nodes
    expected_deriv = 3 * grid.nodes**2 - 2
    np.testing.assert_allclose(backend.diff_x(p_vals, 1), expected_deriv, atol=1e-11)


def test_spectral_backend_diff_x_at_zero_is_identity() -> None:
    """diff_x(c, 0) returns c unchanged (zero derivative is identity)."""
    grid = ChebGLGrid(N=8)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    c = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    out = backend.diff_x(c, 0)
    np.testing.assert_array_equal(out, c)


def test_spectral_backend_float_diff_x_composition() -> None:
    """diff_x(diff_x(p, 1), 1) equals diff_x(p, 2) on resolved polynomials."""
    grid = ChebGLGrid(N=16)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    p = grid.nodes**4 - 2 * grid.nodes**2 + 1
    np.testing.assert_allclose(
        backend.diff_x(backend.diff_x(p, 1), 1), backend.diff_x(p, 2), atol=1e-10
    )


# --- integrate_x ---------------------------------------------------------


def test_spectral_backend_float_integrate_inverts_diff_on_polynomials() -> None:
    """diff_x ∘ integrate_x = id on polynomials the grid resolves to roundoff."""
    grid = ChebGLGrid(N=16, domain=(-1.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    p_vals = grid.nodes**3 - 2 * grid.nodes
    primitive = backend.integrate_x(p_vals)
    np.testing.assert_allclose(backend.diff_x(primitive, 1), p_vals, atol=1e-10)


def test_spectral_backend_float_integrate_vanishes_at_left_boundary() -> None:
    """integrate_x(c) evaluated at the left-boundary node is 0 by construction."""
    grid = ChebGLGrid(N=12, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    p_vals = grid.nodes**2 + grid.nodes + 1
    primitive = backend.integrate_x(p_vals)
    # Trefethen ordering: nodes[N] is the left boundary (a = 0 here).
    assert np.isclose(primitive[-1], 0.0, atol=1e-12)


def test_spectral_backend_sympy_integrate_inverts_diff_on_polynomials() -> None:
    """diff∘integrate = id for the sympy scalar, modulo sp.expand-based equality.

    A small grid (N=6) is enough — sympy's LUsolve is slow on object
    matrices, and the law is the same regardless of size.
    """
    grid = ChebGLGrid(N=6, domain=(-1.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    # Lift x to grid as an object array, then square it (also object).
    p_vals = backend.lift_xonly(X**2 + X)
    primitive = backend.integrate_x(p_vals)
    recovered = backend.diff_x(primitive, 1)
    for j in range(grid.nodes.shape[0]):
        # Each recovered[j] should equal p_vals[j] modulo simplification.
        residual = sp.simplify(sp.nsimplify(recovered[j] - p_vals[j], tolerance=1e-10))
        assert residual == 0 or abs(complex(residual)) < 1e-10, (j, recovered[j], p_vals[j])


# --- normalize -----------------------------------------------------------


def test_spectral_backend_float_normalize_is_identity() -> None:
    """For scalar='float', normalize is the identity (numpy has no expand notion)."""
    grid = ChebGLGrid(N=6)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    c = np.array([1.5, -2.3, 0.0, 4.0, 7.7, -1.1, 2.0])
    out = backend.normalize(c)
    np.testing.assert_array_equal(out, c)


def test_spectral_backend_sympy_normalize_expands_each_entry() -> None:
    """For scalar='sympy', normalize applies sp.expand element-wise."""
    grid = ChebGLGrid(N=2)
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    hbar = sp.Symbol("hbar")
    raw = np.array(
        [(hbar + 1) * (hbar - 1), (X + 1) * (X - 1), sp.Integer(5)],
        dtype=object,
    )
    out = backend.normalize(raw)
    assert out[0] == hbar**2 - 1
    assert out[1] == X**2 - 1
    assert out[2] == sp.Integer(5)


# --- Backend laws (per-substrate sanity) ---------------------------------


def test_spectral_backend_float_zero_is_additive_identity() -> None:
    """zero() + c == c for the float backend."""
    grid = ChebGLGrid(N=6)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    c = np.linspace(1.0, 10.0, 7)
    np.testing.assert_allclose(backend.zero() + c, c)
    np.testing.assert_allclose(c + backend.zero(), c)


def test_spectral_backend_float_one_is_multiplicative_identity() -> None:
    """one() * c == c element-wise for the float backend."""
    grid = ChebGLGrid(N=6)
    backend = SpectralBackend(grid, indep=X, scalar="float")
    c = np.linspace(1.0, 10.0, 7)
    np.testing.assert_allclose(backend.one() * c, c)
    np.testing.assert_allclose(c * backend.one(), c)


def test_spectral_backend_sympy_zero_is_additive_identity() -> None:
    """zero() + c == c for the sympy scalar backend (modulo sp.expand)."""
    grid = ChebGLGrid(N=4)
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    hbar = sp.Symbol("hbar")
    c = np.array([hbar * X, hbar**2, sp.Integer(7), X + hbar, sp.Integer(0)], dtype=object)
    result = backend.zero() + c
    for j in range(5):
        assert sp.expand(result[j] - c[j]) == 0


def test_spectral_backend_sympy_one_is_multiplicative_identity() -> None:
    """one() * c == c element-wise for the sympy scalar backend."""
    grid = ChebGLGrid(N=4)
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    hbar = sp.Symbol("hbar")
    c = np.array([hbar * X, hbar**2, sp.Integer(7), X + hbar, sp.Integer(0)], dtype=object)
    result = backend.one() * c
    for j in range(5):
        assert sp.expand(result[j] - c[j]) == 0
