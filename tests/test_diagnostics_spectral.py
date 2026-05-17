"""Spectral-side diagnostic tests (S8).

Exercises:

  - `residual` on the spectral backend (via the order-0 apply_series
    trick, no `apply_scalar` required).
  - `residual_l2_squared` with `grid=` (Clenshaw-Curtis quadrature).
  - `hbar_curve_at` on the sympy-scalar spectral backend (the partial
    sum entry at the nearest grid node is already a polynomial in ℏ).
  - `hbar_curve_at_sweep` on the float spectral backend (re-runs the
    solver per ℏ value).

All on the canonical exp problem `u' = u, u(0) = 1`.
"""

import numpy as np
import sympy as sp
from ham.deformation import HamProblem
from ham.diagnostics import (
    hbar_curve_at,
    hbar_curve_at_sweep,
    residual,
    residual_l2_squared,
)
from ham.grids import ChebGLGrid
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition
from ham.solver import solve
from ham.spectral import Scalar, SpectralBackend, spectral_linear_operator

X = sp.Symbol("x")
U = sp.Function("u")
HBAR = sp.Symbol("hbar")

N_EXPR = U(X).diff(X) - U(X)
L_EXPR = U(X).diff(X)
DEFORMATION_BCS = (BoundaryCondition(point=sp.Integer(0), derivative_order=0),)


def _build_exp_problem(
    grid: ChebGLGrid, scalar: Scalar, hbar_value: sp.Expr
) -> HamProblem[np.ndarray]:
    L = spectral_linear_operator(  # noqa: N806
        L_EXPR, dependent=U, indep=X, grid=grid, scalar=scalar, bcs=DEFORMATION_BCS
    )
    N = NonlinearOperator(expr=N_EXPR, dependent=U, indep=X)  # noqa: N806
    return HamProblem(L=L, N=N, H=sp.Integer(1), hbar=hbar_value, u0=sp.Integer(1))


# --- residual ------------------------------------------------------------


def test_spectral_residual_shrinks_with_ham_order() -> None:
    """N[u^(M)] on the grid gets smaller in L∞ as the HAM order grows."""
    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_exp_problem(grid, "float", sp.Float(-1.0))

    sol_3 = solve(problem, order=3, backend=backend)
    sol_8 = solve(problem, order=8, backend=backend)

    r_3 = residual(sol_3)
    r_8 = residual(sol_8)

    assert isinstance(r_3, np.ndarray)
    assert isinstance(r_8, np.ndarray)
    assert np.max(np.abs(r_3)) > np.max(np.abs(r_8))
    # Order-8 Taylor of e^x on [0,1] gives residual bounded by x^8 / 8! ≈ 2.48e-5.
    assert np.max(np.abs(r_8)) < 1e-4


def test_spectral_residual_rejects_substitutions_on_float_backend() -> None:
    """On the float backend there are no symbols to substitute — the API rejects it."""
    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_exp_problem(grid, "float", sp.Float(-1.0))
    sol = solve(problem, order=4, backend=backend)

    import pytest

    with pytest.raises(ValueError, match="pre-substituted"):
        residual(sol, hbar_value=sp.Float(-0.5))


# --- residual_l2_squared -------------------------------------------------


def test_spectral_residual_l2_squared_float_is_small_at_converged_order() -> None:
    """At HAM order 10, the Clenshaw-Curtis L² norm² of the residual is tiny."""
    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_exp_problem(grid, "float", sp.Float(-1.0))
    sol = solve(problem, order=10, backend=backend)

    l2_sq = float(residual_l2_squared(sol, None, grid=grid))
    assert l2_sq < 1e-12


def test_spectral_residual_l2_squared_shrinks_with_order() -> None:
    """L² norm² of the residual is monotone-decreasing in the HAM order."""
    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_exp_problem(grid, "float", sp.Float(-1.0))
    norms = [
        float(residual_l2_squared(solve(problem, order=m, backend=backend), None, grid=grid))
        for m in (2, 4, 6, 8)
    ]
    from itertools import pairwise

    for previous, current in pairwise(norms):
        assert current < previous


def test_spectral_residual_l2_squared_requires_grid_or_interval() -> None:
    """Without either `grid` or `interval`, the function rejects the call."""
    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_exp_problem(grid, "float", sp.Float(-1.0))
    sol = solve(problem, order=3, backend=backend)

    import pytest

    with pytest.raises(ValueError, match=r"`interval`.*`grid`"):
        residual_l2_squared(sol, None)


# --- hbar_curve_at -------------------------------------------------------


def test_spectral_hbar_curve_at_sympy_scalar_returns_polynomial_in_hbar() -> None:
    """The partial-sum entry at the nearest grid node is a polynomial in ℏ.

    Substituting ℏ → -1 in that polynomial gives a numeric value
    close to e^(x_star) — within the HAM truncation error at the
    chosen order.
    """
    grid = ChebGLGrid(N=6, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="sympy")
    problem = _build_exp_problem(grid, "sympy", HBAR)
    sol = solve(problem, order=3, backend=backend)

    curve = hbar_curve_at(sol, sp.Float(0.5), grid=grid)
    assert HBAR in curve.free_symbols

    # The order-3 Taylor of e^(0.5) is 1 + 0.5 + 0.125 + 0.5^3/6 ≈ 1.6458.
    value = float(curve.subs(HBAR, sp.Float(-1.0)))
    truncated_taylor = 1.0 + 0.5 + 0.125 + (0.5**3) / 6.0
    assert abs(value - truncated_taylor) < 1e-3


# --- hbar_curve_at_sweep -------------------------------------------------


def test_spectral_hbar_curve_at_sweep_returns_one_value_per_hbar() -> None:
    """Re-running the solver across an ℏ-grid gives the (ℏ, value at x_star) curve."""
    grid = ChebGLGrid(N=16, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")

    def factory(hbar: sp.Expr) -> HamProblem[np.ndarray]:
        return _build_exp_problem(grid, "float", hbar)

    hbar_grid = [sp.Float(h) for h in (-1.5, -1.0, -0.5, 0.0)]
    pairs = hbar_curve_at_sweep(
        factory,
        x_star=sp.Float(0.5),
        hbar_grid=hbar_grid,
        order=4,
        grid=grid,
        backend=backend,
    )

    assert len(pairs) == 4
    # At ℏ = 0 the partial sum collapses to u_0 = 1 (no iteration).
    pair_at_zero = next(p for p in pairs if float(p[0]) == 0.0)
    assert abs(pair_at_zero[1] - 1.0) < 1e-12
    # At ℏ = -1 the order-4 Taylor reproduces e^(0.5) to ~x^5/120 ≈ 2.6e-4.
    pair_at_minus_one = next(p for p in pairs if abs(float(p[0]) + 1.0) < 1e-12)
    assert abs(pair_at_minus_one[1] - np.exp(0.5)) < 1e-3
