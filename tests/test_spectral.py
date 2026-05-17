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
from ham.contracts import verify_linearity
from ham.grids import ChebGLGrid
from ham.operator import BoundaryCondition
from ham.spectral import SpectralBackend, spectral_linear_operator

X = sp.Symbol("x")
U = sp.Function("u")


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


# --- spectral_linear_operator (S6) ---------------------------------------


def test_spectral_linear_operator_action_recovers_polynomial_derivative() -> None:
    """For L = u'', action(p) returns p'' on a resolved polynomial."""
    expr = U(X).diff(X, 2)
    grid = ChebGLGrid(N=16, domain=(-1.0, 1.0))
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")  # noqa: N806 -- Liao's notation
    p = grid.nodes**4 - grid.nodes**2 + 1
    expected = 12 * grid.nodes**2 - 2
    np.testing.assert_allclose(L.apply(p), expected, atol=1e-10)


def test_spectral_linear_operator_action_with_constant_coefficient() -> None:
    """For L = 2*u, action(c) returns 2c element-wise."""
    expr = 2 * U(X)
    grid = ChebGLGrid(N=8)
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")  # noqa: N806 -- Liao's notation
    c = np.linspace(1.0, 10.0, 9)
    np.testing.assert_allclose(L.apply(c), 2 * c, atol=1e-12)


def test_spectral_linear_operator_action_with_x_coefficient() -> None:
    """For L = x*u, action(c) returns nodes * c (x-dependent diagonal scaling)."""
    expr = X * U(X)
    grid = ChebGLGrid(N=8, domain=(0.0, 2.0))
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")  # noqa: N806 -- Liao's notation
    c = np.ones(9)
    np.testing.assert_allclose(L.apply(c), grid.nodes, atol=1e-12)


def test_spectral_linear_operator_rejects_nonlinear_expression() -> None:
    """L = u**2 is not linear and is rejected at construction."""
    expr = U(X) ** 2
    grid = ChebGLGrid(N=8)
    with pytest.raises(ValueError, match="linear"):
        spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")


def test_spectral_linear_operator_rejects_constant_term() -> None:
    """L = u + 1 has a u-free constant term and is rejected."""
    expr = U(X) + sp.Integer(1)
    grid = ChebGLGrid(N=8)
    with pytest.raises(ValueError, match="linear"):
        spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")


# --- spectral_inverter ---------------------------------------------------


def test_spectral_inverter_solves_first_order_ivp_for_constant_rhs() -> None:
    """For L = u' with u(0) = 0 on [0, 1], invert(1) gives u(x) = x at the nodes."""
    expr = U(X).diff(X)
    grid = ChebGLGrid(N=12, domain=(0.0, 1.0))
    bcs = (BoundaryCondition(point=sp.Integer(0), derivative_order=0),)
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float", bcs=bcs)  # noqa: N806 -- Liao's notation
    rhs = np.ones(13)
    u = L.invert(rhs)
    np.testing.assert_allclose(u, grid.nodes, atol=1e-10)


def test_spectral_inverter_solves_second_order_bvp_with_zero_dirichlet() -> None:
    """For L = u'' with u(-1) = u(1) = 0, invert(-2) gives u(x) = 1 - x^2."""
    expr = U(X).diff(X, 2)
    grid = ChebGLGrid(N=16, domain=(-1.0, 1.0))
    bcs = (
        BoundaryCondition(point=sp.Integer(-1), derivative_order=0),
        BoundaryCondition(point=sp.Integer(1), derivative_order=0),
    )
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float", bcs=bcs)  # noqa: N806 -- Liao's notation
    rhs = -2 * np.ones(17)
    u = L.invert(rhs)
    expected = 1 - grid.nodes**2
    np.testing.assert_allclose(u, expected, atol=1e-10)


def test_spectral_inverter_honors_nonzero_bc_value() -> None:
    """For L = u' with u(0) = 5 and rhs = 0, invert gives the constant 5."""
    expr = U(X).diff(X)
    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    bcs = (BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(5)),)
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float", bcs=bcs)  # noqa: N806 -- Liao's notation
    rhs = np.zeros(9)
    u = L.invert(rhs)
    np.testing.assert_allclose(u, 5.0 * np.ones(9), atol=1e-10)


def test_spectral_inverter_rejects_infinite_bc_point() -> None:
    """Asymptotic BC at infinity cannot be honoured on a finite grid."""
    expr = U(X).diff(X)
    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    bcs = (BoundaryCondition(point=sp.oo, derivative_order=0),)
    with pytest.raises(ValueError, match="finite"):
        spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float", bcs=bcs)


# --- verify_linearity with injected equal comparator (D-4 demo) -----------


def test_spectral_linear_operator_passes_linearity_with_np_allclose_equal() -> None:
    """verify_linearity passes for L = u'' + 2 u' + u on the float spectral backend.

    The D-4 demonstration: the same verify_linearity machinery the
    sympy backend uses serves the spectral float backend just by
    swapping the comparator from `sp.expand(a-b) == 0` to `np.allclose`.
    """
    expr = U(X).diff(X, 2) + 2 * U(X).diff(X) + U(X)
    grid = ChebGLGrid(N=12, domain=(-1.0, 1.0))
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="float")  # noqa: N806 -- Liao's notation

    rng = np.random.default_rng(seed=42)
    samples = []
    for _ in range(3):
        u_sample = rng.standard_normal(13)
        v_sample = rng.standard_normal(13)
        alpha = np.array(float(rng.standard_normal()))
        beta = np.array(float(rng.standard_normal()))
        samples.append((u_sample, v_sample, alpha, beta))

    def np_allclose(a: np.ndarray, b: np.ndarray) -> bool:
        return bool(np.allclose(a, b, atol=1e-10))

    verify_linearity(L, samples, equal=np_allclose)


def test_spectral_linear_operator_passes_linearity_with_sympy_elementwise_equal() -> None:
    """verify_linearity passes for L = u'' + u on the sympy-scalar spectral backend.

    Same machinery; comparator is element-wise tolerance-on-the-residual
    for the object-array path. The grid nodes are floats (cos(πj/N) is
    not exact for general j, N), which carries through into the
    polynomial-in-ℏ residual as float coefficients near machine epsilon —
    `sp.expand(a-b) == 0` is too strict unless the nodes are rational.
    Substituting every free symbol to 1 and float-casting the residual
    gives a single numeric magnitude to threshold.
    """
    expr = U(X).diff(X, 2) + U(X)
    grid = ChebGLGrid(N=3, domain=(-1.0, 1.0))
    L = spectral_linear_operator(expr, dependent=U, indep=X, grid=grid, scalar="sympy")  # noqa: N806 -- Liao's notation

    hbar = sp.Symbol("hbar")
    u_sample = np.array([hbar, sp.Integer(2), hbar + 1, sp.Integer(0)], dtype=object)
    v_sample = np.array([sp.Integer(1), hbar**2, sp.Integer(3), hbar], dtype=object)
    alpha = np.array(sp.Integer(2), dtype=object)
    beta = np.array(sp.Rational(-1, 3), dtype=object)

    def sympy_elementwise_close(a: np.ndarray, b: np.ndarray, tol: float = 1e-9) -> bool:
        for x, y in zip(a, b, strict=True):
            diff = x - y
            if diff == 0:
                continue
            free = diff.free_symbols
            value = float(diff.subs({sym: sp.Integer(1) for sym in free}) if free else diff)
            if abs(value) > tol:
                return False
        return True

    verify_linearity(L, [(u_sample, v_sample, alpha, beta)], equal=sympy_elementwise_close)
