"""End-to-end spectral solves (S7).

Smoke tests for `solve(problem, order, backend=SpectralBackend(...))` on
the canonical `u' = u, u(0) = 1` problem (exact solution `e^x`):

  - `scalar="float"`: ℏ is pre-substituted to a numeric value; the
    partial sum on the grid converges to e^x at HAM order M.
  - `scalar="sympy"`: ℏ stays symbolic inside every grid entry; the
    partial sum at each grid node is a polynomial in ℏ, and
    substituting `ℏ = -1` reproduces the float result to machine
    precision.

Both validate that the solver loop is genuinely substrate-agnostic
(the parametrisation work S1-S6 paid off) and that the dual-scalar
SpectralBackend (PLAN.org D-1) round-trips through the inner loop
without behaviour drift.
"""

import numpy as np
import sympy as sp
from ham.deformation import HamProblem
from ham.grids import ChebGLGrid
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition
from ham.solver import solve
from ham.spectral import Scalar, SpectralBackend, spectral_linear_operator

X = sp.Symbol("x")
U = sp.Function("u")
HBAR = sp.Symbol("hbar")

# `u' - u = 0` on [0, 1], u(0) = 1. Exact: u(x) = e^x.
# HAM ingredients:
#   L[u]  = u'              (auxiliary linear operator)
#   N[u]  = u' - u          (original problem; N[u] = 0)
#   u_0   = 1               (satisfies u(0) = 1)
#   H     = 1               (auxiliary function)
# Deformation BC: u_m(0) = 0 for m >= 1 (homogeneous).
N_EXPR = U(X).diff(X) - U(X)
L_EXPR = U(X).diff(X)
DEFORMATION_BCS = (BoundaryCondition(point=sp.Integer(0), derivative_order=0),)


def _build_spectral_problem(
    grid: ChebGLGrid,
    scalar: Scalar,
    hbar_value: sp.Expr,
) -> HamProblem[np.ndarray]:
    """Build a HamProblem with a spectral L on `grid` at the given scalar."""
    L = spectral_linear_operator(  # noqa: N806 -- Liao's notation
        L_EXPR,
        dependent=U,
        indep=X,
        grid=grid,
        scalar=scalar,
        bcs=DEFORMATION_BCS,
    )
    N = NonlinearOperator(expr=N_EXPR, dependent=U, indep=X)  # noqa: N806
    return HamProblem(L=L, N=N, H=sp.Integer(1), hbar=hbar_value, u0=sp.Integer(1))


def test_spectral_solve_float_scalar_matches_exp_at_order_4() -> None:
    """For u' = u, u(0) = 1, HAM at hbar = -1, order 4 on a Cheb grid matches e^x.

    The HAM partial sum at hbar = -1 reproduces the truncated Taylor
    expansion of e^x, so the grid-node values should match
    `1 + x + x²/2 + x³/6 + x⁴/24` at the Chebyshev-GL nodes to roundoff,
    and match `exp(x)` to the Taylor-truncation error (~x⁵/120 ~ 8e-3
    on [0, 1]).
    """
    grid = ChebGLGrid(N=16, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_spectral_problem(grid, "float", hbar_value=sp.Float(-1.0))
    solution = solve(problem, order=4, backend=backend)

    grid_values = solution.partial_sum()
    truncated_exp = (
        1.0 + grid.nodes + grid.nodes**2 / 2.0 + grid.nodes**3 / 6.0 + grid.nodes**4 / 24.0
    )
    np.testing.assert_allclose(grid_values, truncated_exp, atol=1e-10)


def test_spectral_solve_float_converges_to_exp_with_higher_order() -> None:
    """The L∞ error in `partial_sum - exp(x)` shrinks with HAM order."""
    grid = ChebGLGrid(N=16, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_spectral_problem(grid, "float", hbar_value=sp.Float(-1.0))

    err_4 = np.max(
        np.abs(solve(problem, order=4, backend=backend).partial_sum() - np.exp(grid.nodes))
    )
    err_8 = np.max(
        np.abs(solve(problem, order=8, backend=backend).partial_sum() - np.exp(grid.nodes))
    )

    assert err_8 < err_4
    # Order 8 Taylor of e^x on [0,1] has error bounded by 1/9! ≈ 2.76e-6.
    assert err_8 < 1e-5


def test_spectral_solve_sympy_scalar_substituted_matches_float() -> None:
    """Sympy-scalar spectral solve at ℏ = -1 matches the float-scalar solve.

    Validates PLAN.org D-1's dual-scalar machinery: the sympy path
    carries ℏ symbolically through every grid entry, and substituting
    ℏ → -1 at the end must reproduce the float path's answer (which
    pre-substituted ℏ at the start). Same problem, same answer; only
    the substitution timing differs.
    """
    # Small grid — sympy LUsolve is slow on object matrices.
    grid = ChebGLGrid(N=6, domain=(0.0, 1.0))
    backend_sym = SpectralBackend(grid, indep=X, scalar="sympy")
    backend_flt = SpectralBackend(grid, indep=X, scalar="float")

    problem_sym = _build_spectral_problem(grid, "sympy", hbar_value=HBAR)
    problem_flt = _build_spectral_problem(grid, "float", hbar_value=sp.Float(-1.0))

    sol_sym = solve(problem_sym, order=3, backend=backend_sym)
    sol_flt = solve(problem_flt, order=3, backend=backend_flt)

    # Substitute ℏ = -1 in the sympy result's grid values.
    sym_at_neg_one = np.array(
        [float(entry.subs(HBAR, sp.Float(-1.0))) for entry in sol_sym.partial_sum()],
        dtype=np.float64,
    )
    flt_at_neg_one = sol_flt.partial_sum()

    np.testing.assert_allclose(sym_at_neg_one, flt_at_neg_one, atol=1e-9)


def test_spectral_solve_partial_sum_matches_taylor_coefficients_term_by_term() -> None:
    """The k-th HAM coefficient `phi.coeff(k)` at ℏ = -1 equals x^k / k!.

    For u' = u with u_0 = 1, HAM at ℏ = -1 reproduces the Taylor
    expansion of e^x term by term: u_k(x) = x^k / k!. Checking the
    coefficients directly (rather than only the partial sum) catches
    drift in any single solve_step.
    """
    import math

    grid = ChebGLGrid(N=12, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    problem = _build_spectral_problem(grid, "float", hbar_value=sp.Float(-1.0))
    solution = solve(problem, order=5, backend=backend)

    for k in range(0, 6):
        expected = grid.nodes**k / math.factorial(k)
        np.testing.assert_allclose(
            solution.phi.coeff(k),
            expected,
            atol=1e-10,
            err_msg=f"u_{k}(x) does not match x^{k}/{k}!",
        )
