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

2. **Coefficient ergodicity.** Each HAM step raises the polynomial
   degree by one (the inverse of `L = d/dt`), so every power `t^k`
   eventually appears in some `u_k(t)`.

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
U = sp.Function("u")
HBAR = sp.Symbol("hbar")
_DEFAULT_INTERVAL = (sp.Integer(0), sp.Integer(1))
_DEFAULT_THRESHOLD = sp.Rational(1, 100)


def build_problem() -> HamProblem:
    """Assemble the logistic HAM problem (see module docstring)."""
    return HamProblem(
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


def solve_to(order: int) -> HamSolution:
    """Run the HAM solver on the logistic problem to working order `order`."""
    return solve(build_problem(), order=order)


def is_convergent(
    solution: HamSolution,
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


def analyze(solution: HamSolution) -> dict[str, sp.Expr | bool]:
    """Bundle of diagnostics for a logistic solution.

    Returns a dict with the same shape as the quadratic-drag example:
    residual at ℏ=-1, ℏ-curve at t=1, optimal ℏ from the grid search,
    L² norm² at the optimal ℏ, and the validity gate at ℏ=-1.
    """
    interval = _DEFAULT_INTERVAL

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
    print(f"Logistic equation, working order M = {M}")
    print(f"  partial sum at hbar = -1:    {sol.evaluate_at_hbar(sp.Integer(-1))}")
    print(f"  sigmoid Taylor reference:    {taylor_reference(M)}")
    print(f"  residual at hbar = -1:       {analysis['residual_at_neg_one']}")
    print(f"  ℏ-curve at t = 1:            {analysis['hbar_curve_at_t_eq_1']}")
    print(f"  optimal ℏ over grid:         {analysis['optimal_hbar']}")
    print(f"  L² norm² at optimal ℏ:       {analysis['l2_norm_at_optimal']}")
    print(f"  convergent at ℏ = -1:        {analysis['convergent_at_neg_one']}")
