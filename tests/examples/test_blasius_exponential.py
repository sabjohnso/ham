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
def sol_order_2() -> HamSolution:
    """One symbolic-alpha Blasius exponential-basis solve at M = 2, reused across tests.

    M = 2 (≈ 3 s) keeps the test runtime manageable while still
    demonstrating two-parameter (ℏ, alpha) convergence; the headline 2D
    optimum at M = 2 already gives |error| ≈ 4x10⁻⁴ on f''(0).
    """
    return solve_to(2)


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


def test_partial_sum_satisfies_f_zero_is_zero(sol_order_2: HamSolution) -> None:
    """For every (ℏ, alpha), the partial sum vanishes at η = 0."""
    partial = sol_order_2.partial_sum()
    assert sp.simplify(partial.subs(ETA, 0)) == 0


def test_partial_sum_satisfies_f_prime_zero_is_zero(sol_order_2: HamSolution) -> None:
    """For every (ℏ, alpha), the partial sum has f'(0) = 0."""
    partial = sol_order_2.partial_sum()
    assert sp.simplify(sp.diff(partial, ETA).subs(ETA, 0)) == 0


def test_partial_sum_satisfies_asymptotic_bc_at_alpha_one(
    sol_order_2: HamSolution,
) -> None:
    """At alpha = 1, every ℏ: lim(η → ∞) f'(η) = 1 (the original Blasius BC).

    Symbolic alpha makes the limit-at-infinity computation expensive in
    general; pinning alpha = 1 lets sympy resolve the limit cleanly.
    """
    partial = sol_order_2.partial_sum().subs(ALPHA, 1)
    f_prime_at_oo = sp.limit(sp.diff(partial, ETA), ETA, sp.oo)
    assert sp.simplify(f_prime_at_oo - 1) == 0


def test_f_double_prime_at_zero_close_to_howarth_at_alpha_one(
    sol_order_2: HamSolution,
) -> None:
    """At alpha = 1 (the Stage 11 default) and ℏ = -9/10, M = 2: error < 0.01.

    Backwards compatibility check: the legacy single-parameter
    `f_double_prime_at_zero(sol, hbar_value)` form still works and
    still gives the same answer as Stage 11 at alpha = 1.
    """
    value = f_double_prime_at_zero(sol_order_2, sp.Rational(-9, 10))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < sp.Rational(1, 100)


def test_two_parameter_optimum_beats_alpha_one_optimum(
    sol_order_2: HamSolution,
) -> None:
    """The 2D (ℏ, alpha) optimum yields |f''(0) - Howarth| < 10⁻³ at M = 2.

    Headline Stage 12b result: tuning alpha as a second free parameter
    drops the error by an order of magnitude versus tuning ℏ alone
    with alpha = 1. At M = 2 the 2D optimum is around (ℏ = -1/2,
    alpha = 7/10) with f''(0) ≈ 0.47003 — an error of about 4x10⁻⁴.
    """
    analysis = analyze(sol_order_2)
    err = sp.Abs(analysis["f_double_prime_at_zero_best"] - HOWARTH_F_DOUBLE_PRIME_AT_ZERO)
    assert err < sp.Rational(1, 1000)


def test_validity_gate_passes_at_best_substitutions(sol_order_2: HamSolution) -> None:
    """is_convergent accepts the 2D-optimal (ℏ, alpha) substitution."""
    analysis = analyze(sol_order_2)
    assert is_convergent(sol_order_2, substitutions=analysis["best_substitutions"]) is True


def test_validity_gate_fails_at_hbar_zero(sol_order_2: HamSolution) -> None:
    """At ℏ = 0 the partial sum is u_0(alpha=1); f''(0) = u_0''(0) = alpha = 1, far from Howarth."""
    assert is_convergent(sol_order_2, sp.Integer(0)) is False


def test_two_parameter_beats_polynomial_basis_at_higher_order() -> None:
    """The 2D-optimal exponential-basis at M = 2 beats polynomial-basis at M = 5.

    Cross-stage result combining Stages 10, 11, 12: with the
    exponential basis AND two-parameter tuning, just M = 2 HAM
    iterations bring f''(0) within 10⁻³ of Howarth — almost two
    orders of magnitude tighter than the polynomial-basis M = 5
    best of ≈ 0.052.
    """
    from examples.blasius import solve_to as solve_poly
    from examples.blasius_exponential import solve_to as solve_exp

    sol_poly = solve_poly(5)
    sol_exp = solve_exp(2)
    err_poly = float(
        sp.Abs(
            f_double_prime_at_zero(sol_poly, sp.Rational(-2, 5)) - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    err_exp_2d = float(
        sp.Abs(
            f_double_prime_at_zero(
                sol_exp,
                substitutions={HBAR: sp.Rational(-1, 2), ALPHA: sp.Rational(7, 10)},
            )
            - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    # Expect the 2D-tuned exponential basis to be at least 50x better.
    assert err_exp_2d < err_poly / 50


def test_f_double_prime_rejects_both_shortcuts(sol_order_2: HamSolution) -> None:
    """Supplying both `hbar_value` and `substitutions` is ambiguous and must raise."""
    with pytest.raises(ValueError, match="not both"):
        f_double_prime_at_zero(
            sol_order_2,
            sp.Integer(-1),
            substitutions={HBAR: sp.Integer(-1), ALPHA: sp.Integer(1)},
        )


def test_hbar_and_alpha_remain_symbolic_in_u_k_for_k_geq_1(
    sol_order_2: HamSolution,
) -> None:
    """u_k for k ≥ 1 carries both ℏ and alpha; substitute-late convention holds for two params."""
    for k in range(1, sol_order_2.order + 1):
        free = sol_order_2.phi.coeff(k).free_symbols
        assert HBAR in free
        assert ALPHA in free
