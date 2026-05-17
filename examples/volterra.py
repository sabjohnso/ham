"""HAM example: Volterra integro-differential population model.

The single-species Volterra population model with cumulative resource
depletion (Liao, *Beyond Perturbation*, Ch. 10):

    u'(t) = κ · u(t) · [1 - u(t) - ∫_0^t u(τ) dτ],    u(0) = alpha.

The integral term represents accumulated resource consumption; as the
population grows, it depletes the available resources, eventually
forcing decay. With κ > 0 and 0 < alpha < 1, u(t) initially grows, peaks,
and decays asymptotically to 0.

This is the first worked example in the library with an
**integro-differential** N. The compiler's Integral branch (Stage 9a)
turns `sp.Integral(u(s), (s, 0, t))` into coefficient-wise integration
of the homotopy series — the integro-differential structure flows
through the same `apply_series` machinery as polynomial-in-u N.

How this example illustrates Liao's three fundamental rules
-----------------------------------------------------------

1. **Solution expression.** u(t) is analytic at t = 0 with a
   polynomial Taylor expansion (derivable directly from the
   integro-differential equation by the recurrence in
   `taylor_reference` below). Polynomial base via L = d/dt with
   homogeneous deformation BCs reflects that structure.

2. **Coefficient ergodicity** (base coefficients in
   `u = Σ_n c_n · t^n`, /not/ the q-coefficients `u_k`; see Liao Rule 2
   in `docs/concepts/convergence.md`). Each HAM step raises the
   polynomial degree of u_k(t) by /two/ — one from L^{-1}, one from
   the integral inside N — so every power t^k eventually appears in
   some u_m.

3. **Solution existence.** u_0 = alpha is a constant function satisfying
   the original BC u(0) = alpha exactly. The deformation BCs are the
   homogeneous u_m(0) = 0 for m ≥ 1.

HAM problem statement
---------------------

- Variable:        t = sp.Symbol("t").
- Function:        u = sp.Function("u").
- Rate parameter:  κ (a constant; we pick κ = 1).
- Initial pop.:    alpha (a constant; we pick alpha = 1/10).
- Nonlinear N[u]:  u' - κ·u·(1 - u - ∫_0^t u(s) ds)
                 = u' - κ·u + κ·u² + κ·u·∫_0^t u(s) ds.
- Linear L:        d/dt with u(0) = 0.
- Auxiliary H(t):  1.
- Initial guess:   u_0 = alpha (constant).

The exact solution has no closed analytic form. The reference for
this example is the Taylor expansion of u(t) derived by feeding a
power-series ansatz `u(t) = Σ a_k t^k` back through the integro-
differential equation — `taylor_reference(order)` below.

A property worth noting: unlike the four prior worked examples,
**HAM at ℏ = -1 does NOT collapse to the Taylor truncation here.**
The integral inside N gives a different relationship between
HAM order and Taylor coefficient than purely-polynomial N does:
HAM at ℏ = -1 produces the correct t^0 and t^1 coefficients at
order 1, but the higher-t coefficients differ from the Taylor
expansion until higher orders fill them in. The cross-checks below
respect this — they sample the HAM partial sum and the Taylor
reference at concrete `t` values rather than comparing
coefficient-by-coefficient.
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
S = sp.Symbol("s")
U = sp.Function("u")
HBAR = sp.Symbol("hbar")
KAPPA = sp.Integer(1)
ALPHA = sp.Rational(1, 10)
_DEFAULT_INTERVAL = (sp.Integer(0), sp.Integer(1))
_DEFAULT_THRESHOLD = sp.Rational(1, 1000)

ORIGINAL_BCS: tuple[BoundaryCondition, ...] = (
    BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=ALPHA),
)
"""The original problem's boundary conditions: u(0) = alpha = 1/10.

Exposed as a module-level constant for use with
`ham.contracts.verify_initial_guess(build_problem(), ORIGINAL_BCS)`.
The deformation BC on `build_problem().L` is the homogeneous
`u(0) = 0`.
"""


def build_problem() -> HamProblem[sp.Expr]:
    """Assemble the Volterra HAM problem (see module docstring)."""
    integral_term = sp.Integral(U(S), (S, 0, T))
    n_expr = U(T).diff(T) - KAPPA * U(T) * (sp.Integer(1) - U(T) - integral_term)
    return HamProblem[sp.Expr](
        L=LinearOperator(
            var=T,
            action=lambda e: sp.diff(e, T),
            bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
        ),
        N=NonlinearOperator(
            expr=n_expr,
            dependent=U,
            indep=T,
        ),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=ALPHA,
    )


def taylor_reference(order: int) -> sp.Expr:
    """Taylor expansion of u(t) about t = 0, to (and including) t^order.

    Derived from the integro-differential equation by power-series
    ansatz `u(t) = Σ a_k t^k`. Matching t^n on both sides gives

        (n+1) a_{n+1} = κ a_n
                      - κ Σ_{i+j=n} a_i a_j
                      - κ Σ_{ell=0..n-1} a_{n-1-ell} a_ell / (ell + 1),

    with a_0 = alpha.
    """
    coeffs: list[sp.Expr] = [ALPHA]
    for n in range(order):
        u_squared = sum((coeffs[i] * coeffs[n - i] for i in range(n + 1)), sp.Integer(0))
        u_times_integral = sum(
            (coeffs[n - 1 - ell] * coeffs[ell] / sp.Integer(ell + 1) for ell in range(n)),
            sp.Integer(0),
        )
        next_coeff = KAPPA * (coeffs[n] - u_squared - u_times_integral) / sp.Integer(n + 1)
        coeffs.append(sp.simplify(next_coeff))
    return sum((c * T**k for k, c in enumerate(coeffs)), sp.Integer(0))


def solve_to(order: int) -> HamSolution[sp.Expr]:
    """Run the HAM solver on the Volterra problem to working order `order`."""
    return solve(build_problem(), order=order)


def is_convergent(
    solution: HamSolution[sp.Expr],
    hbar_value: sp.Expr,
    interval: tuple[sp.Expr, sp.Expr] = _DEFAULT_INTERVAL,
    threshold: sp.Expr = _DEFAULT_THRESHOLD,
) -> bool:
    """Validity gate: L² residual on `interval` must be below `threshold`.

    The default threshold (1/1000) is calibrated for alpha = 1/10. The
    initial population is small, so the residual magnitudes are also
    small; a looser threshold would let an obviously-bad ℏ slip through.
    """
    norm_squared = residual_l2_squared(solution, hbar_value, interval)
    return bool(sp.simplify(norm_squared - threshold**2) < 0)


def _grid_at_neg_one_neighbourhood() -> list[sp.Expr]:
    return [
        sp.Rational(-3, 2),
        sp.Integer(-1),
        sp.Rational(-1, 2),
        sp.Integer(0),
    ]


def analyze(solution: HamSolution[sp.Expr]) -> dict[str, sp.Expr | bool]:
    """Diagnostics for a Volterra solution."""
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


if __name__ == "__main__":  # pragma: no cover -- runnable script entry point
    M = 5
    sol = solve_to(M)
    analysis = analyze(sol)
    print(f"Volterra population model, working order M = {M}")
    print(f"  alpha (initial population) = {ALPHA}, κ (rate) = {KAPPA}")
    print(f"  partial sum at hbar = -1:    {sol.evaluate_at_hbar(sp.Integer(-1))}")
    print(f"  Taylor reference (order {M}):    {taylor_reference(M)}")
    print(f"  residual at hbar = -1:       {analysis['residual_at_neg_one']}")
    print(f"  ℏ-curve at t = 1:            {analysis['hbar_curve_at_t_eq_1']}")
    print(f"  optimal ℏ over grid:         {analysis['optimal_hbar']}")
    print(f"  L² norm² at optimal ℏ:       {analysis['l2_norm_at_optimal']}")
    print(f"  convergent at ℏ = -1:        {analysis['convergent_at_neg_one']}")
