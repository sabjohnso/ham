"""HAM example: vertical fall under quadratic air drag.

Non-dimensionalised so that g/k = 1, the velocity v(t) of a falling
object subject to quadratic drag satisfies

    v'(t) = 1 - v(t)^2,    v(0) = 0,

whose closed-form solution is `v(t) = tanh(t)` (terminal velocity 1).
The Taylor expansion of tanh is

    tanh(t) = t - t^3/3 + 2 t^5/15 - 17 t^7/315 + O(t^9).

How this example illustrates Liao's three fundamental rules
-----------------------------------------------------------

1. **Solution expression.** `tanh(t)` is analytic at `t = 0` with a
   pure-polynomial Taylor expansion, so the polynomial basis `{t^k}` is
   appropriate. `L = d/dt` and the homogeneous BC `u_k(0) = 0` are
   chosen so that `L^{-1}` (definite integration from 0 to t) maps
   polynomials into polynomials, keeping every iterate in the chosen
   base.

2. **Coefficient ergodicity** (base coefficients in
   `u = Σ_n c_n · t^n`, /not/ the q-coefficients `u_k`; see Liao Rule 2
   in `docs/concepts/convergence.md`). Each HAM step raises the
   polynomial degree by one (via the inverse of `L = d/dt`), so every
   power `t^k` eventually appears in some `u_k(t)`. There is no
   "missing" base element that the deformation chain cannot reach.

3. **Solution existence.** `u_0 = 0` satisfies the original BC
   `v(0) = 0` exactly, so the deformation BCs can be the homogeneous
   `u_k(0) = 0` for `k >= 1` without conflict. `L` is invertible on
   its image (every polynomial in `t` is the derivative of another
   polynomial), and the explicit `antiderivative` inverter from
   `ham.operator` is available to bypass `sympy.dsolve` if higher-order
   runs need it.

HAM problem statement
---------------------

- Independent variable: `t = sp.Symbol("t")`.
- Dependent function:   `v = sp.Function("v")`.
- Nonlinear operator:   `N[v] = v' - 1 + v^2`  (so `N[tanh] = 0`).
- Linear operator:      `L = d/dt` with `v(0) = 0`.
- Auxiliary function:   `H(t) = 1`.
- Initial guess:        `u_0 = 0`  (matches the original BC).
- Convergence param:    `hbar` (symbolic).

At `hbar = -1` the HAM partial sum equals the truncated Taylor
expansion of `tanh(t)`, which is the deepest cross-check available.

A subtlety worth noting: `ℏ = -1` is not necessarily the L²-optimal
ℏ on `[0, 1]`. At order M = 7, the residual norm at `ℏ = -1/2` is
roughly a quarter of the norm at `ℏ = -1` (3.4e-4 vs 1.5e-3). That is
the HAM adaptive-ℏ advantage at work: by tuning ℏ the method can
distribute the truncation error across the interval better than plain
Taylor truncation does. Hence the `optimal_hbar` grid search in
`analyze()` may legitimately return a value other than -1.
"""

import sympy as sp
from ham.deformation import HamProblem
from ham.diagnostics import (
    hbar_curve_at,
    optimal_hbar,
    residual,
    residual_l2_squared,
)
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import HamSolution, solve

T = sp.Symbol("t")
V = sp.Function("v")
HBAR = sp.Symbol("hbar")
_DEFAULT_THRESHOLD = sp.Rational(1, 10)
_DEFAULT_INTERVAL = (sp.Integer(0), sp.Integer(1))

ORIGINAL_BCS: tuple[BoundaryCondition, ...] = (
    BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0)),
)
"""The original problem's boundary conditions: v(0) = 0.

Exposed as a module-level constant so callers can hand it to
`ham.contracts.verify_initial_guess(build_problem(), ORIGINAL_BCS)`
to assert that `u_0 = 0` satisfies the original BC. The deformation
BCs declared on `build_problem().L` are the homogeneous versions
(here identical, because the original BC is already homogeneous).
"""


def build_problem() -> HamProblem:
    """Assemble the quadratic-drag HAM problem (see module docstring)."""
    return HamProblem(
        L=LinearOperator(
            var=T,
            action=lambda e: sp.diff(e, T),
            bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
        ),
        N=NonlinearOperator(
            expr=V(T).diff(T) - sp.Integer(1) + V(T) ** 2,
            dependent=V,
            indep=T,
        ),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(0),
    )


def exact_solution() -> sp.Expr:
    """The closed-form solution `v(t) = tanh(t)`."""
    return sp.tanh(T)


def taylor_reference(order: int) -> sp.Expr:
    """Truncated Taylor expansion of `tanh(t)` to (and including) `t^order`."""
    return sp.expand(sp.series(sp.tanh(T), T, 0, order + 1).removeO())


def solve_to(order: int) -> HamSolution:
    """Run the HAM solver on the quadratic-drag problem to working order `order`."""
    return solve(build_problem(), order=order)


def is_convergent(
    solution: HamSolution,
    hbar_value: sp.Expr,
    interval: tuple[sp.Expr, sp.Expr] = _DEFAULT_INTERVAL,
    threshold: sp.Expr = _DEFAULT_THRESHOLD,
) -> bool:
    """Validity gate: L² residual on `interval` must be below `threshold`.

    Composes `ham.diagnostics.residual_l2_squared` into a per-example
    predicate. The default threshold (1/10) is calibrated for this
    problem on the unit interval. Empirically, the L² norm squared
    drops from 0.2 at M=1 to ~0.0015 at M=7 (so the norm itself is
    ~0.04 at M=7); the threshold sits in the middle so the
    underconverged hbar=0 case (norm² = 1) fails clearly and the
    converged hbar=-1 case at M≥3 passes.
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


def analyze(solution: HamSolution) -> dict[str, sp.Expr | bool]:
    """Bundle of diagnostics for a quadratic-drag solution.

    Returns a dict keyed by:
      - 'residual_at_neg_one': N applied to the partial sum at hbar = -1.
      - 'hbar_curve_at_t_eq_1': partial sum at t = 1 as a polynomial in ℏ.
      - 'optimal_hbar': grid value minimising L² residual norm on [0,1].
      - 'l2_norm_at_optimal': L² residual norm squared at that ℏ.
      - 'convergent_at_neg_one': validity-gate result at hbar = -1.
    """
    interval = (sp.Integer(0), sp.Integer(1))

    def norm(s: HamSolution, h: sp.Expr) -> sp.Expr:
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


if __name__ == "__main__":  # pragma: no cover -- runnable script entry point
    M = 7
    sol = solve_to(M)
    analysis = analyze(sol)
    print(f"Quadratic-drag projectile, working order M = {M}")
    print(f"  partial sum at hbar = -1:    {sol.evaluate_at_hbar(sp.Integer(-1))}")
    print(f"  tanh Taylor reference:       {taylor_reference(M)}")
    print(f"  residual at hbar = -1:       {analysis['residual_at_neg_one']}")
    print(f"  ℏ-curve at t = 1:            {analysis['hbar_curve_at_t_eq_1']}")
    print(f"  optimal ℏ over grid:         {analysis['optimal_hbar']}")
    print(f"  L² norm² at optimal ℏ:       {analysis['l2_norm_at_optimal']}")
    print(f"  convergent at ℏ = -1:        {analysis['convergent_at_neg_one']}")
