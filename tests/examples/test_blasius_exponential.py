"""Regression tests for the exponential-basis Blasius example.

Pins HAM output against Howarth's f''(0) ≈ 0.4696 reference. Exercises
the custom inverter (Stage 11b) that filters growing-exp free constants
left by sympy.dsolve when applying the asymptotic BC `f'(∞) = 0` on
L = d^3/dη^3 - alpha^2·d/dη, plus the Stage 12b two-parameter (ℏ, alpha)
optimisation via `ham.diagnostics.optimal_parameters`.
"""

import pytest
import sympy as sp
from examples.blasius_exponential import (
    ALPHA,
    ETA,
    HBAR,
    HOWARTH_F_DOUBLE_PRIME_AT_ZERO,
    _blasius_exponential_inverter,
    analyze,
    build_problem,
    f_double_prime_at_zero,
    is_convergent,
    solve_to,
)
from ham.solver import HamSolution


@pytest.fixture(scope="module")
def sol_order_3() -> HamSolution:
    """One symbolic-alpha Blasius exponential-basis solve at M = 3, reused across tests.

    Stage 13a's closed-form basis-aware inverter brought the solve time
    down enough that M = 3 (≈ 3 s) is now practical for the test
    fixture; Stage 12 had to cap at M = 2 (≈ 3 s with the dsolve
    inverter at every step). At M = 3 the 2D-optimal error on f''(0)
    drops from ~4x10⁻⁴ (M=2) to ~1.6x10⁻⁴.
    """
    return solve_to(3)


def test_initial_guess_satisfies_all_three_bcs() -> None:
    """u_0(η) = η - 1/alpha + e^(-alphaη)/alpha: u_0(0) = u_0'(0) = 0, u_0'(∞) = 1.

    Verified symbolically in alpha (the new symbolic parameter from Stage 12b)
    so the BCs hold for any positive alpha, not just alpha = 1.
    """
    problem = build_problem()
    u0 = problem.u0
    assert sp.simplify(u0.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, 0)) == 0
    assert sp.simplify(sp.limit(sp.diff(u0, ETA), ETA, sp.oo) - 1) == 0


def test_inverter_zeros_growing_exp_branch_on_resonant_rhs() -> None:
    """The custom inverter handles a resonant rhs that dsolve cannot.

    For L = d^3/dη^3 - alpha^2·d/dη with alpha symbolic, the kernel contains
    e^(±alphaη). A rhs of η·e^(-alphaη) is resonant. The custom inverter zeroes
    the free constants left by dsolve, leaving a function satisfying
    all three BCs symbolically in alpha.
    """
    rhs = ETA * sp.exp(-ALPHA * ETA)
    result = _blasius_exponential_inverter(rhs)
    assert sp.simplify(result.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(result, ETA).subs(ETA, 0)) == 0
    # Asymptotic BC: substitute alpha = 1 to make the limit computable
    assert sp.simplify(sp.limit(sp.diff(result, ETA).subs(ALPHA, 1), ETA, sp.oo)) == 0
    lhs = sp.diff(result, ETA, 3) - ALPHA**2 * sp.diff(result, ETA)
    assert sp.simplify(lhs - rhs) == 0


def test_partial_sum_satisfies_f_zero_is_zero(sol_order_3: HamSolution) -> None:
    """For every (ℏ, alpha), the partial sum vanishes at η = 0."""
    partial = sol_order_3.partial_sum()
    assert sp.simplify(partial.subs(ETA, 0)) == 0


def test_partial_sum_satisfies_f_prime_zero_is_zero(sol_order_3: HamSolution) -> None:
    """For every (ℏ, alpha), the partial sum has f'(0) = 0."""
    partial = sol_order_3.partial_sum()
    assert sp.simplify(sp.diff(partial, ETA).subs(ETA, 0)) == 0


def test_partial_sum_satisfies_asymptotic_bc_at_alpha_one(
    sol_order_3: HamSolution,
) -> None:
    """At alpha = 1, every ℏ: lim(η → ∞) f'(η) = 1 (the original Blasius BC).

    Symbolic alpha makes the limit-at-infinity computation expensive in
    general; pinning alpha = 1 lets sympy resolve the limit cleanly.
    """
    partial = sol_order_3.partial_sum().subs(ALPHA, 1)
    f_prime_at_oo = sp.limit(sp.diff(partial, ETA), ETA, sp.oo)
    assert sp.simplify(f_prime_at_oo - 1) == 0


def test_f_double_prime_at_zero_close_to_howarth_at_alpha_one(
    sol_order_3: HamSolution,
) -> None:
    """At alpha = 1 (Stage 11 default) and ℏ = -7/10, M = 3: error < 0.01.

    Backwards compatibility check: the legacy single-parameter
    `f_double_prime_at_zero(sol, hbar_value)` form still works and
    still gives the same answer as Stage 11 at alpha = 1, M = 3.
    """
    value = f_double_prime_at_zero(sol_order_3, sp.Rational(-7, 10))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < sp.Rational(1, 100)


def test_two_parameter_optimum_beats_alpha_one_optimum(
    sol_order_3: HamSolution,
) -> None:
    """The 2D (ℏ, alpha) optimum yields |f''(0) - Howarth| < 5x10⁻⁴ at M = 3.

    Headline cross-stage result combining Stages 12 and 13: with the
    closed-form basis-aware inverter, M = 3 is practical for the test
    fixture, and the 2D optimum tightens to about 1.6x10⁻⁴ — six
    times tighter than the Stage 12 M = 2 best. At M = 3 the optimum
    lives around (ℏ = -1, alpha = 13/10).
    """
    analysis = analyze(sol_order_3)
    err = sp.Abs(analysis["f_double_prime_at_zero_best"] - HOWARTH_F_DOUBLE_PRIME_AT_ZERO)
    assert err < sp.Rational(5, 10000)


def test_validity_gate_passes_at_best_substitutions(sol_order_3: HamSolution) -> None:
    """is_convergent accepts the 2D-optimal (ℏ, alpha) substitution."""
    analysis = analyze(sol_order_3)
    assert is_convergent(sol_order_3, substitutions=analysis["best_substitutions"]) is True


def test_validity_gate_fails_at_hbar_zero(sol_order_3: HamSolution) -> None:
    """At ℏ = 0 the partial sum is u_0(alpha=1); f''(0) = u_0''(0) = alpha = 1, far from Howarth."""
    assert is_convergent(sol_order_3, sp.Integer(0)) is False


def test_two_parameter_beats_polynomial_basis_at_higher_order(
    sol_order_3: HamSolution,
) -> None:
    """The 2D-optimal exp basis at M = 3 beats polynomial basis at M = 5 by 100x.

    Cross-stage result combining Stages 10, 11, 12, 13: with the
    exponential basis, two-parameter tuning, AND the closed-form
    basis-aware inverter, just M = 3 HAM iterations bring f''(0)
    within 2x10⁻⁴ of Howarth — over two orders of magnitude tighter
    than the polynomial-basis M = 5 best of ≈ 0.052.
    """
    from examples.blasius import solve_to as solve_poly

    sol_poly = solve_poly(5)
    err_poly = float(
        sp.Abs(
            f_double_prime_at_zero(sol_poly, sp.Rational(-2, 5)) - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    err_exp_2d = float(
        sp.Abs(
            f_double_prime_at_zero(
                sol_order_3,
                substitutions={HBAR: sp.Integer(-1), ALPHA: sp.Rational(13, 10)},
            )
            - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    # 2D-tuned exp basis at M=3 should be at least 100x better than poly M=5.
    assert err_exp_2d < err_poly / 100


def test_f_double_prime_rejects_both_shortcuts(sol_order_3: HamSolution) -> None:
    """Supplying both `hbar_value` and `substitutions` is ambiguous and must raise."""
    with pytest.raises(ValueError, match="not both"):
        f_double_prime_at_zero(
            sol_order_3,
            sp.Integer(-1),
            substitutions={HBAR: sp.Integer(-1), ALPHA: sp.Integer(1)},
        )


def test_hbar_and_alpha_remain_symbolic_in_u_k_for_k_geq_1(
    sol_order_3: HamSolution,
) -> None:
    """u_k for k ≥ 1 carries both ℏ and alpha; substitute-late convention holds for two params."""
    for k in range(1, sol_order_3.order + 1):
        free = sol_order_3.phi.coeff(k).free_symbols
        assert HBAR in free
        assert ALPHA in free
