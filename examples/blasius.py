"""HAM example: Blasius boundary-layer equation.

The Blasius flat-plate boundary-layer ODE:

    f'''(η) + (1/2) f(η) f''(η) = 0,
    f(0) = 0,    f'(0) = 0,    f'(∞) = 1.

The third condition is asymptotic. Liao's canonical HAM treatment
(*Beyond Perturbation*, Ch. 14) uses an exponential basis that
encodes the asymptotic decay directly. The library's
`LinearOperator` only supports point boundary conditions at this
time, so this example uses the **truncated-domain** form: replace
`f'(∞) = 1` with `f'(η_max) = 1` for a large finite `η_max`. We
pick `η_max = 10`, which is well past the boundary layer (which
ends around η ≈ 5).

Howarth's high-precision numerical value f''(0) ≈ 0.469600 is the
standard cross-check for any HAM Blasius solver.

What makes this example different
---------------------------------

Unlike every prior worked example, **ℏ = -1 fails** for Blasius via
polynomial basis. The series at ℏ = -1 oscillates and diverges as M
grows; the right ℏ is somewhere around -0.4, found by sampling the
ℏ-curve.

The L² residual norm minimization (the diagnostic that works for
the other examples) also misleads here: there is a "false plateau"
at small positive ℏ where the residual norm is tiny but the
solution is wrong (the partial sum collapses near u_0). The
validity gate in this example is therefore based on **closeness to
Howarth's reference value** for f''(0), not on residual norm.

How this example illustrates Liao's three fundamental rules
-----------------------------------------------------------

1. **Solution expression.** A polynomial base is *not* the natural
   choice for Blasius — Liao's standard treatment uses an
   exponential base that captures the asymptotic decay structure.
   We use polynomial here for library compatibility, accepting the
   slow convergence as a feature of the example. The slow
   convergence /is/ Liao's point about Rule 1: a wrong base costs
   you orders.
2. **Coefficient ergodicity** (base coefficients in
   `u = Σ_n c_n · η^n`, /not/ the q-coefficients `u_k`; see Liao Rule 2
   in `docs/concepts/convergence.md`). Each HAM step raises the
   polynomial degree by 3 (from L^{-1} of d^3/dη^3), so every power
   η^k eventually appears in some u_k(η).
3. **Solution existence.** u_0 = η^2 / (2 η_max) satisfies all three
   truncated BCs exactly: u_0(0) = 0, u_0'(0) = 0, u_0'(η_max) = 1.
   The deformation BCs are the homogeneous versions of these three
   (all values zero), composed without conflict.

HAM problem statement
---------------------

- Variable:        η = sp.Symbol("eta").
- Function:        f = sp.Function("f").
- Domain cap:      η_max = 10.
- Nonlinear N[f]:  f''' + (1/2) f f''.
- Linear L:        d^3/d_eta^3 with f(0) = f'(0) = f'(η_max) = 0.
- Auxiliary H(η):  1.
- Initial guess:   u_0 = η^2 / (2 η_max).
- Reference:       Howarth's tabulated f''(0) ≈ 0.469600.
"""

import sympy as sp
from ham.deformation import HamProblem
from ham.diagnostics import hbar_curve_at, residual
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.solver import HamSolution, solve

ETA = sp.Symbol("eta")
F = sp.Function("f")
HBAR = sp.Symbol("hbar")
ETA_MAX = sp.Integer(10)
HOWARTH_F_DOUBLE_PRIME_AT_ZERO = sp.Rational(4696, 10000)
_DEFAULT_TOLERANCE = sp.Rational(1, 10)

ORIGINAL_BCS: tuple[BoundaryCondition, ...] = (
    BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0)),
    BoundaryCondition(point=sp.Integer(0), derivative_order=1, value=sp.Integer(0)),
    BoundaryCondition(point=ETA_MAX, derivative_order=1, value=sp.Integer(1)),
)
"""The original (truncated-domain) problem's boundary conditions.

  f(0) = 0,  f'(0) = 0,  f'(eta_max) = 1.

The third condition is the truncated version of Blasius's
asymptotic `f'(infty) = 1`; the polynomial-basis example replaces
the infinite endpoint with `ETA_MAX = 10` for library compatibility.
The exponential-basis example handles the true `f'(infty) = 1`
directly.

Exposed for use with
`ham.contracts.verify_initial_guess(build_problem(), ORIGINAL_BCS)`.
The deformation BCs declared on `build_problem().L` are the
homogeneous versions (all values zero).
"""


def build_problem() -> HamProblem:
    """Assemble the Blasius HAM problem in truncated-domain form."""
    u0 = ETA**2 / (sp.Integer(2) * ETA_MAX)
    n_expr = F(ETA).diff(ETA, 3) + sp.Rational(1, 2) * F(ETA) * F(ETA).diff(ETA, 2)
    return HamProblem(
        L=LinearOperator(
            var=ETA,
            action=lambda e: sp.diff(e, ETA, 3),
            bcs=(
                BoundaryCondition(point=sp.Integer(0), derivative_order=0),
                BoundaryCondition(point=sp.Integer(0), derivative_order=1),
                BoundaryCondition(point=ETA_MAX, derivative_order=1),
            ),
        ),
        N=NonlinearOperator(
            expr=n_expr,
            dependent=F,
            indep=ETA,
        ),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=u0,
    )


def solve_to(order: int) -> HamSolution:
    """Run the HAM solver on the Blasius problem to working order `order`."""
    return solve(build_problem(), order=order)


def f_double_prime_at_zero(solution: HamSolution, hbar_value: sp.Expr) -> sp.Expr:
    """Compute f''(0) from the HAM partial sum at the given ℏ.

    Howarth's reference value is HOWARTH_F_DOUBLE_PRIME_AT_ZERO ≈ 0.4696;
    closeness of this number to the reference is the per-problem
    convergence test for Blasius.
    """
    partial = solution.evaluate_at_hbar(hbar_value)
    return sp.diff(partial, ETA, 2).subs(ETA, 0)


def is_convergent(
    solution: HamSolution,
    hbar_value: sp.Expr,
    tolerance: sp.Expr = _DEFAULT_TOLERANCE,
) -> bool:
    """Validity gate: |f''(0) - Howarth_reference| < tolerance.

    Unlike the other worked examples, the gate here is NOT a residual-
    norm threshold. The L² residual minimization has a false plateau
    at small positive ℏ where the residual is tiny but the partial sum
    is far from the Blasius solution. Comparing to Howarth's reference
    value for f''(0) is a more honest convergence test for this
    polynomial-basis Blasius setup.
    """
    fdd = f_double_prime_at_zero(solution, hbar_value)
    diff = fdd - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
    return bool(sp.Abs(diff) < tolerance)


def _hbar_sweep_grid() -> list[sp.Expr]:
    """A grid spanning the negative-ℏ range where convergence is plausible."""
    return [sp.Rational(k, 10) for k in range(-10, 1)]


def analyze(solution: HamSolution) -> dict[str, sp.Expr | bool]:
    """Diagnostics for a Blasius solution.

    Returns the f''(0) values at every ℏ in the sweep grid so the
    caller can plot or inspect the ℏ-curve of the Blasius constant.
    The closest match to Howarth's reference is also reported.
    """
    sweep = _hbar_sweep_grid()
    fdd_at_grid = {h: f_double_prime_at_zero(solution, h) for h in sweep}
    best_hbar = min(
        sweep,
        key=lambda h: float(sp.Abs(fdd_at_grid[h] - HOWARTH_F_DOUBLE_PRIME_AT_ZERO)),
    )
    return {
        "f_double_prime_at_zero_hbar_neg_one": f_double_prime_at_zero(solution, sp.Integer(-1)),
        "f_double_prime_at_zero_best": fdd_at_grid[best_hbar],
        "best_hbar_for_f_double_prime": best_hbar,
        "hbar_curve_at_eta_max": hbar_curve_at(solution, ETA_MAX),
        "residual_at_best_hbar": residual(solution, best_hbar),
        "convergent_at_best_hbar": is_convergent(solution, best_hbar),
    }


if __name__ == "__main__":  # pragma: no cover -- runnable script entry point
    M = 5
    sol = solve_to(M)
    analysis = analyze(sol)
    print(f"Blasius boundary-layer equation, working order M = {M}")
    print(f"  η_max = {ETA_MAX}, Howarth f''(0) = {HOWARTH_F_DOUBLE_PRIME_AT_ZERO}")
    print()
    fdd_neg_one = float(analysis["f_double_prime_at_zero_hbar_neg_one"])
    fdd_best = float(analysis["f_double_prime_at_zero_best"])
    howarth = float(HOWARTH_F_DOUBLE_PRIME_AT_ZERO)
    print(f"  f''(0) at hbar = -1:           {fdd_neg_one:+.4f}  (diverges)")
    print(f"  best ℏ on the sweep grid:      {analysis['best_hbar_for_f_double_prime']}")
    print(f"  f''(0) at best ℏ:              {fdd_best:+.4f}")
    print(f"  |f''(0) - Howarth| at best ℏ:  {abs(fdd_best - howarth):+.4f}")
    print(f"  convergent gate at best ℏ:     {analysis['convergent_at_best_hbar']}")
    print()
    print("Note: the standard L² residual minimization is misleading for")
    print("this polynomial-basis Blasius — it has a false plateau at small")
    print("positive ℏ where the residual is tiny but f''(0) is far from")
    print("Howarth's reference. See examples/blasius.py docstring for details.")
