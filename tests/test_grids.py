"""Tests for the spectral grid protocol and ChebGLGrid.

S5a deliverable: a `Grid` packages the data a spectral backend needs —
nodes, dense differentiation matrix, quadrature weights, domain — plus
a memoised `differentiation_matrix_power(k)` for the calculus
operations. `ChebGLGrid` is the first concrete implementation
(Chebyshev-Gauss-Lobatto on a finite interval); future grid families
(rational-Chebyshev for semi-infinite, Fourier for periodic) will
implement the same Protocol without touching the Backend or any
consumer (PLAN.org D-3).
"""

import numpy as np
import pytest
from ham.grids import ChebGLGrid


def test_chebgl_grid_node_count() -> None:
    """A degree-N grid has N+1 nodes."""
    grid = ChebGLGrid(N=8)
    assert grid.nodes.shape == (9,)


def test_chebgl_grid_endpoints_are_domain_bounds() -> None:
    """The Trefethen-ordered nodes start at the right boundary (b) and end at the left (a)."""
    grid = ChebGLGrid(N=8, domain=(2.0, 5.0))
    assert np.isclose(grid.nodes[0], 5.0)
    assert np.isclose(grid.nodes[-1], 2.0)


def test_chebgl_grid_default_domain_is_minus_one_to_one() -> None:
    """Default ChebGLGrid lives on [-1, 1]."""
    grid = ChebGLGrid(N=6)
    assert grid.domain == (-1.0, 1.0)
    assert np.isclose(grid.nodes[0], 1.0)
    assert np.isclose(grid.nodes[-1], -1.0)


def test_chebgl_grid_rejects_invalid_n() -> None:
    """N < 1 is rejected (need at least 2 nodes for a meaningful grid)."""
    with pytest.raises(ValueError, match="N >= 1"):
        ChebGLGrid(N=0)


def test_chebgl_grid_rejects_inverted_domain() -> None:
    """a < b is required."""
    with pytest.raises(ValueError, match="a < b"):
        ChebGLGrid(N=8, domain=(5.0, 2.0))


def test_chebgl_differentiation_exact_on_resolved_polynomial() -> None:
    """`D · f` recovers `f'` to machine precision when `f` is a polynomial the grid resolves.

    ChebGL with N+1 nodes resolves polynomials up to degree N exactly,
    so D applied to their nodal values gives the nodal values of f'.
    """
    grid = ChebGLGrid(N=16)
    x = grid.nodes
    f = x**3 - 2 * x
    fprime_expected = 3 * x**2 - 2
    fprime_computed = grid.differentiation_matrix @ f
    np.testing.assert_allclose(fprime_computed, fprime_expected, atol=1e-11)


def test_chebgl_differentiation_matches_trefethen_program_11() -> None:
    """Trefethen Program 11 sanity check: D[cos(kx)] ≈ -k sin(kx) to roundoff.

    The classical Chebyshev-spectral benchmark; at N=20 and k=1 the
    error is near machine epsilon.
    """
    grid = ChebGLGrid(N=20)
    x = grid.nodes
    k = 1.0
    f = np.cos(k * x)
    fprime_expected = -k * np.sin(k * x)
    fprime_computed = grid.differentiation_matrix @ f
    np.testing.assert_allclose(fprime_computed, fprime_expected, atol=1e-10)


def test_chebgl_differentiation_on_scaled_domain() -> None:
    """Domain scaling: on [a, b], `D · x` evaluates `d/dx (x) = 1` everywhere."""
    grid = ChebGLGrid(N=12, domain=(2.0, 7.0))
    x = grid.nodes
    fprime = grid.differentiation_matrix @ x
    np.testing.assert_allclose(fprime, np.ones_like(x), atol=1e-11)


def test_chebgl_quadrature_integrates_low_degree_polynomials_exactly() -> None:
    """Clenshaw-Curtis weights integrate ∫_{-1}^{1} x^k dx exactly for k ≤ N."""
    grid = ChebGLGrid(N=10)
    x = grid.nodes
    w = grid.quadrature_weights
    for k in range(0, 8):
        integral_exact = 0.0 if k % 2 == 1 else 2.0 / (k + 1)
        integral_computed = float(np.sum(w * x**k))
        assert np.isclose(integral_computed, integral_exact, atol=1e-12), (
            f"k={k}: computed {integral_computed} vs exact {integral_exact}"
        )


def test_chebgl_quadrature_sum_of_weights_equals_domain_length() -> None:
    """∫_{a}^{b} 1 dx = b - a, computed via the weights."""
    grid = ChebGLGrid(N=10, domain=(2.0, 5.0))
    assert np.isclose(float(np.sum(grid.quadrature_weights)), 3.0, atol=1e-12)


def test_chebgl_diff_matrix_power_one_is_d() -> None:
    """`differentiation_matrix_power(1)` is the same object as `differentiation_matrix`."""
    grid = ChebGLGrid(N=8)
    assert grid.differentiation_matrix_power(1) is grid.differentiation_matrix


def test_chebgl_diff_matrix_power_zero_is_identity() -> None:
    """`differentiation_matrix_power(0)` is the identity matrix."""
    grid = ChebGLGrid(N=8)
    np.testing.assert_array_equal(grid.differentiation_matrix_power(0), np.eye(9))


def test_chebgl_diff_matrix_power_is_memoised() -> None:
    """Repeated access returns the same ndarray *object*, not just equal values."""
    grid = ChebGLGrid(N=8)
    a = grid.differentiation_matrix_power(2)
    b = grid.differentiation_matrix_power(2)
    assert a is b


def test_chebgl_diff_matrix_power_two_equals_double_application() -> None:
    """D^2 · f equals D · (D · f) for resolved polynomials."""
    grid = ChebGLGrid(N=16)
    x = grid.nodes
    f = x**4 - 2 * x**2 + 1
    via_power = grid.differentiation_matrix_power(2) @ f
    via_double = grid.differentiation_matrix @ (grid.differentiation_matrix @ f)
    np.testing.assert_allclose(via_power, via_double, atol=1e-11)


def test_chebgl_diff_matrix_power_rejects_negative_k() -> None:
    """Negative k is not supported (integration is the inverter's job, not D^k)."""
    grid = ChebGLGrid(N=8)
    with pytest.raises(ValueError, match="k >= 0"):
        grid.differentiation_matrix_power(-1)


# --- RationalChebGrid ----------------------------------------------------


def test_rational_cheb_grid_node_count() -> None:
    """RationalChebGrid(N) has N+1 nodes (matching ChebGLGrid)."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8)
    assert grid.nodes.shape == (9,)


def test_rational_cheb_grid_endpoint_at_infinity() -> None:
    """Index 0 maps to infinity; index N maps to 0."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8, L=2.0)
    assert np.isinf(grid.nodes[0])
    assert np.isclose(grid.nodes[-1], 0.0)


def test_rational_cheb_grid_node_density_clusters_near_l() -> None:
    """With L=1 the median interior node sits near x=1 — characteristic length."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=20, L=1.0)
    finite_nodes = grid.nodes[1:]
    median_node = np.median(finite_nodes)
    assert np.isclose(median_node, 1.0, atol=0.2)


def test_rational_cheb_grid_domain() -> None:
    """domain reports (0, inf)."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8)
    a, b = grid.domain
    assert a == 0.0
    assert np.isinf(b)


def test_rational_cheb_grid_rejects_invalid_n() -> None:
    from ham.grids import RationalChebGrid

    with pytest.raises(ValueError, match="N >= 1"):
        RationalChebGrid(N=0)


def test_rational_cheb_grid_rejects_invalid_l() -> None:
    from ham.grids import RationalChebGrid

    with pytest.raises(ValueError, match="L > 0"):
        RationalChebGrid(N=8, L=0)


def test_rational_cheb_grid_diff_matrix_zero_row_at_infinity() -> None:
    """The row of D_x at the infinity node is identically zero.

    At xi=1 (the image of x=infinity), dxi/dx = 0, so the chain rule
    D_x = diag(rho) @ D_xi produces a zero row at index 0. The
    implementation exposes this rather than papering over it.
    """
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8)
    np.testing.assert_array_equal(grid.differentiation_matrix[0, :], np.zeros(9))


def test_rational_cheb_grid_diff_matrix_recovers_exp_derivative() -> None:
    """D_x[exp(-x)] equals -exp(-x) at the finite nodes to spectral accuracy."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=24, L=4.0)
    # exp(-inf) is 0 in numpy, so this works as-is including at index 0.
    f = np.exp(-grid.nodes)
    fprime_computed = grid.differentiation_matrix @ f
    expected = -np.exp(-grid.nodes)
    # Compare on finite nodes (skip j=0, the zero-row at infinity)
    np.testing.assert_allclose(fprime_computed[1:], expected[1:], atol=1e-6)


def test_rational_cheb_grid_quadrature_integrates_decaying_exponential() -> None:
    """sum(weights * exp(-x)) ≈ ∫_0^∞ exp(-x) dx = 1.

    The canonical test integrand for rational-Cheb quadrature on
    [0, infinity): exp(-x) decays fast enough that the j=0
    contribution (where the weight is 0 by convention and the
    integrand is also 0) is correctly handled, and the remaining
    nodes sum to the exact integral within spectral accuracy.
    """
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=30, L=2.0)
    f = np.exp(-grid.nodes)  # exp(-inf) = 0 at index 0
    integral = float(np.sum(grid.quadrature_weights * f))
    assert abs(integral - 1.0) < 1e-6


def test_rational_cheb_grid_quadrature_integrates_decaying_rational() -> None:
    """sum(weights * 1/(1+x)²) ≈ ∫_0^∞ 1/(1+x)² dx = 1.

    A second decay shape, this time rational rather than exponential.
    Tests that the quadrature handles different asymptotic rates
    without the j=0 weight convention biasing the result.
    """
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=40, L=1.0)
    f = 1.0 / (1.0 + grid.nodes) ** 2
    f[0] = 0.0  # 1/(1+inf)² = 0; numpy gives 0, this is documentary
    integral = float(np.sum(grid.quadrature_weights * f))
    # Rational decay converges more slowly than exponential; ~3e-4 at N=40 is typical.
    assert abs(integral - 1.0) < 1e-3


def test_rational_cheb_grid_quadrature_weight_at_infinity_is_zero() -> None:
    """weights[0] = 0 by convention (the j=0 node maps to infinity)."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=10)
    assert grid.quadrature_weights[0] == 0.0


def test_rational_cheb_grid_diff_matrix_power_memoised() -> None:
    """D_x^k returns the same ndarray object on repeated access."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8)
    a = grid.differentiation_matrix_power(2)
    b = grid.differentiation_matrix_power(2)
    assert a is b


def test_rational_cheb_grid_diff_matrix_power_zero_is_identity() -> None:
    """D_x^0 is the identity matrix (same as ChebGLGrid's convention)."""
    from ham.grids import RationalChebGrid

    grid = RationalChebGrid(N=8)
    np.testing.assert_array_equal(grid.differentiation_matrix_power(0), np.eye(9))
