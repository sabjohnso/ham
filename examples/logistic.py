"""HAM example: the logistic / sigmoid population model.

The unit-rate logistic equation

    u'(t) = u(t) * (1 - u(t)),    u(0) = 1/2,

has the closed-form solution `u(t) = 1 / (1 + exp(-t))` (the standard
sigmoid). Its Taylor expansion around `t = 0` is

    u(t) = 1/2 + t/4 - t^3/48 + t^5/480 - 17 t^7/80640 + O(t^9).

How this example illustrates Liao's three fundamental rules
-----------------------------------------------------------

1. **Solution expression.** The sigmoid is analytic at `t = 0` with a
   pure-polynomial Taylor expansion (only odd-power corrections around
   the centre value 1/2), so the polynomial basis is the natural one.
   `L = d/dt` and the deformation BC `u_k(0) = 0` for `k >= 1` keep
   every iterate in that base.

2. **Coefficient ergodicity** (base coefficients in
   `u = Σ_n c_n · t^n`, /not/ the q-coefficients `u_k`; see Liao Rule 2
   in `docs/concepts/convergence.md`). Each HAM step raises the
   polynomial degree by one (the inverse of `L = d/dt`), so every
   power `t^k` eventually appears in some `u_k(t)`.

3. **Solution existence.** This example exercises a code path the
   prior worked examples did not: a **non-zero initial guess**
   `u_0 = 1/2`, chosen to satisfy the original BC `u(0) = 1/2`
   exactly. The deformation BCs are then `u_k(0) = 0` for `k >= 1`
   without conflict (compatible with `L`'s homogeneous BCs).

HAM problem statement
---------------------

- Independent variable: `t = sp.Symbol("t")`.
- Dependent function:   `u = sp.Function("u")`.
- Nonlinear operator:   `N[u] = u' - u + u^2`  (so `N[sigmoid] = 0`).
- Linear operator:      `L = d/dt` with `u(0) = 0` (homogeneous BC).
- Auxiliary function:   `H(t) = 1`.
- Initial guess:        `u_0 = 1/2`  (matches the original BC).
- Convergence param:    `hbar` (symbolic).

At `hbar = -1` the HAM partial sum reproduces the truncated Taylor
expansion of the sigmoid, the deepest cross-check available.
"""

import numpy as np
import sympy as sp
from ham.deformation import HamProblem
from ham.diagnostics import (
    hbar_curve_at,
    optimal_hbar,
    residual,
    residual_l2_squared,
)
from ham.grids import ChebGLGrid
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import HamSolution, solve
from ham.spectral import Scalar, SpectralBackend, spectral_linear_operator

T = sp.Symbol("t")
U = sp.Function("u")
HBAR = sp.Symbol("hbar")
_DEFAULT_INTERVAL = (sp.Integer(0), sp.Integer(1))
_DEFAULT_THRESHOLD = sp.Rational(1, 100)

ORIGINAL_BCS: tuple[BoundaryCondition, ...] = (
    BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Rational(1, 2)),
)
"""The original problem's boundary conditions: u(0) = 1/2.

Exposed as a module-level constant so callers can hand it to
`ham.contracts.verify_initial_guess(build_problem(), ORIGINAL_BCS)`
to assert that the non-zero `u_0 = 1/2` satisfies the original BC.
The deformation BC declared on `build_problem().L` is the homogeneous
version `u(0) = 0`.
"""


def build_problem() -> HamProblem[sp.Expr]:
    """Assemble the logistic HAM problem (see module docstring)."""
    return HamProblem[sp.Expr](
        L=LinearOperator(
            var=T,
            action=lambda e: sp.diff(e, T),
            bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
        ),
        N=NonlinearOperator(
            expr=U(T).diff(T) - U(T) + U(T) ** 2,
            dependent=U,
            indep=T,
        ),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Rational(1, 2),
    )


def exact_solution() -> sp.Expr:
    """The closed-form solution `u(t) = 1 / (1 + exp(-t))`."""
    return sp.Integer(1) / (sp.Integer(1) + sp.exp(-T))


def taylor_reference(order: int) -> sp.Expr:
    """Truncated Taylor expansion of the sigmoid to (and including) `t^order`."""
    return sp.expand(sp.series(exact_solution(), T, 0, order + 1).removeO())


def solve_to(order: int) -> HamSolution[sp.Expr]:
    """Run the HAM solver on the logistic problem to working order `order`."""
    return solve(build_problem(), order=order)


def is_convergent(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr,
    interval: tuple[sp.Expr, sp.Expr] = _DEFAULT_INTERVAL,
    threshold: sp.Expr = _DEFAULT_THRESHOLD,
) -> bool:
    """Validity gate: L² residual on `interval` must be below `threshold`.

    Composes `ham.diagnostics.residual_l2_squared` into a per-example
    predicate. The default threshold (1/100) is calibrated for this
    problem on the unit interval: the sigmoid's Taylor coefficients
    shrink fast (factors of 1/4, 1/48, 1/480, ...), so the residual
    norm at hbar=-1 is already comfortably small at moderate M, while
    the underconverged hbar=0 case (partial sum = u_0 = 1/2, residual
    = -1/4 constant) sits well above the threshold.
    """
    norm_squared = residual_l2_squared(solution, hbar_value, interval)
    return bool(sp.simplify(norm_squared - threshold**2) < 0)


def _grid_at_neg_one_neighbourhood() -> list[sp.Expr]:
    """A sensible ℏ-grid covering -1 for grid-search optimisation."""
    return [
        sp.Rational(-3, 2),
        sp.Integer(-1),
        sp.Rational(-1, 2),
        sp.Integer(0),
    ]


def analyze(solution: HamSolution[sp.Expr]) -> dict[str, sp.Expr | bool]:
    """Bundle of diagnostics for a logistic solution.

    Returns a dict with the same shape as the quadratic-drag example:
    residual at ℏ=-1, ℏ-curve at t=1, optimal ℏ from the grid search,
    L² norm² at the optimal ℏ, and the validity gate at ℏ=-1.
    """
    interval = _DEFAULT_INTERVAL

    def norm(s: HamSolution[sp.Expr], h: sp.Expr) -> sp.Expr:
        return residual_l2_squared(s, h, interval)

    grid = _grid_at_neg_one_neighbourhood()
    best = optimal_hbar(solution, grid, norm)
    return {
        "residual_at_neg_one": residual(solution, sp.Integer(-1)),
        "hbar_curve_at_t_eq_1": hbar_curve_at(solution, sp.Integer(1)),
        "optimal_hbar": best,
        "l2_norm_at_optimal": norm(solution, best),
        "convergent_at_neg_one": is_convergent(solution, sp.Integer(-1)),
    }


# --- Spectral substrate (SHAM) -------------------------------------------


def build_spectral_problem(
    grid: ChebGLGrid,
    scalar: Scalar = "float",
    *,
    hbar_value: sp.Expr | None = None,
) -> HamProblem[np.ndarray]:
    """Same logistic HAM problem on the spectral substrate over `grid`.

    `scalar="float"` pre-substitutes ℏ to a numeric value (defaults to
    -1, the canonical Taylor-collapsing evaluation). `scalar="sympy"`
    keeps ℏ symbolic inside every grid entry so the partial sum at any
    node is a polynomial in ℏ — the same interpretation as
    `build_problem()`'s sympy path, sampled at the grid nodes.

    Initial guess `u_0 = 1/2` is the *non-zero* constant required by
    the original BC `u(0) = 1/2`; lifted to the grid this is a constant
    vector. The deformation BC `u_k(0) = 0` for k >= 1 is imposed by
    `spectral_inverter` per step.
    """
    if hbar_value is None:
        hbar_value = sp.Float(-1.0) if scalar == "float" else HBAR
    return HamProblem(
        L=spectral_linear_operator(
            U(T).diff(T),
            dependent=U,
            indep=T,
            grid=grid,
            scalar=scalar,
            bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
        ),
        N=NonlinearOperator(
            expr=U(T).diff(T) - U(T) + U(T) ** 2,
            dependent=U,
            indep=T,
        ),
        H=sp.Integer(1),
        hbar=hbar_value,
        u0=sp.Rational(1, 2),
    )


def solve_to_spectral(
    order: int,
    *,
    grid: ChebGLGrid | None = None,
    scalar: Scalar = "float",
    hbar_value: sp.Expr | None = None,
) -> HamSolution[np.ndarray]:
    """Run the spectral HAM solver on the logistic problem.

    Default grid is `ChebGLGrid(N=20, domain=(0.0, 1.0))` — the matching
    interval for the symbolic path's `[0, 1]` residual norm.
    """
    if grid is None:
        grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=T, scalar=scalar)
    problem = build_spectral_problem(grid, scalar=scalar, hbar_value=hbar_value)
    return solve(problem, order=order, backend=backend)


if __name__ == "__main__":  # pragma: no cover -- runnable script entry point
    M = 7
    sol = solve_to(M)
    analysis = analyze(sol)
    print(f"Logistic equation, working order M = {M}")
    print(f"  partial sum at hbar = -1:    {sol.evaluate_at_hbar(sp.Integer(-1))}")
    print(f"  sigmoid Taylor reference:    {taylor_reference(M)}")
    print(f"  residual at hbar = -1:       {analysis['residual_at_neg_one']}")
    print(f"  ℏ-curve at t = 1:            {analysis['hbar_curve_at_t_eq_1']}")
    print(f"  optimal ℏ over grid:         {analysis['optimal_hbar']}")
    print(f"  L² norm² at optimal ℏ:       {analysis['l2_norm_at_optimal']}")
    print(f"  convergent at ℏ = -1:        {analysis['convergent_at_neg_one']}")

    print()
    print(f"Spectral redo, ChebGLGrid(N=20, [0, 1]), float scalar, ℏ = -1, M = {M}")
    grid = ChebGLGrid(N=20, domain=(0.0, 1.0))
    sol_spec = solve_to_spectral(M, grid=grid)
    partial = sol_spec.partial_sum()
    exact = 1.0 / (1.0 + np.exp(-grid.nodes))
    err = float(np.max(np.abs(partial - exact)))
    print(f"  L∞ error vs sigmoid on grid: {err:.3e}")
