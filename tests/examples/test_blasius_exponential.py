"""Regression tests for the exponential-basis Blasius example.

Pins HAM output against Howarth's f''(0) ≈ 0.4696 reference. Exercises
the custom inverter (Stage 11b) that filters growing-exp free constants
left by sympy.dsolve when applying the asymptotic BC `f'(∞) = 0` on
L = d^3/dη^3 - alpha^2·d/dη.
"""

import pytest
import sympy as sp
from examples.blasius_exponential import (
    ALPHA,
    ETA,
    HBAR,
    HOWARTH_F_DOUBLE_PRIME_AT_ZERO,
    _blasius_exponential_inverter,
    build_problem,
    f_double_prime_at_zero,
    is_convergent,
    solve_to,
)
from ham.solver import HamSolution


@pytest.fixture(scope="module")
def sol_order_3() -> HamSolution:
    """One Blasius exponential-basis solve at M = 3, reused across tests."""
    return solve_to(3)


def test_initial_guess_satisfies_all_three_bcs() -> None:
    """u_0(η) = η - 1/alpha + e^(-alphaη)/alpha: u_0(0) = u_0'(0) = 0, u_0'(∞) = 1."""
    problem = build_problem()
    u0 = problem.u0
    assert sp.simplify(u0.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, 0)) == 0
    assert sp.simplify(sp.limit(sp.diff(u0, ETA), ETA, sp.oo) - 1) == 0


def test_inverter_zeros_growing_exp_branch_on_resonant_rhs() -> None:
    """The custom inverter handles a resonant rhs (η·exp(-η)) that dsolve cannot.

    For L = d^3/dη^3 - d/dη, the kernel contains exp(-η). A rhs of
    η·exp(-η) is resonant, producing terms with η^2·exp(-η) in the
    particular solution and growing-exp homogeneous components that
    sympy.dsolve leaves as free C constants. The custom inverter
    zeroes those free constants and returns a function satisfying
    all three BCs.
    """
    rhs = ETA * sp.exp(-ALPHA * ETA)
    result = _blasius_exponential_inverter(rhs)
    # All three BCs satisfied
    assert sp.simplify(result.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(result, ETA).subs(ETA, 0)) == 0
    assert sp.simplify(sp.limit(sp.diff(result, ETA), ETA, sp.oo)) == 0
    # And the ODE L[u] = rhs is satisfied
    lhs = sp.diff(result, ETA, 3) - ALPHA**2 * sp.diff(result, ETA)
    assert sp.simplify(lhs - rhs) == 0


def test_partial_sum_satisfies_f_zero_is_zero(sol_order_3: HamSolution) -> None:
    """For every ℏ, the partial sum vanishes at η = 0."""
    partial = sol_order_3.partial_sum()
    assert sp.simplify(partial.subs(ETA, 0)) == 0


def test_partial_sum_satisfies_f_prime_zero_is_zero(sol_order_3: HamSolution) -> None:
    """For every ℏ, the partial sum has f'(0) = 0."""
    partial = sol_order_3.partial_sum()
    assert sp.simplify(sp.diff(partial, ETA).subs(ETA, 0)) == 0


def test_partial_sum_satisfies_asymptotic_bc(sol_order_3: HamSolution) -> None:
    """For every ℏ, lim(η → ∞) f'(η) = 1 (the original Blasius BC).

    Each u_k for k ≥ 1 has f'(∞) = 0 (the homogeneous deformation
    BC enforced by the custom inverter), and u_0 contributes 1 to
    the limit. So the partial sum has f'(∞) = 1 for any ℏ.
    """
    partial = sol_order_3.partial_sum()
    f_prime_at_oo = sp.limit(sp.diff(partial, ETA), ETA, sp.oo)
    assert sp.simplify(f_prime_at_oo - 1) == 0


def test_f_double_prime_at_zero_close_to_howarth_at_best_hbar(
    sol_order_3: HamSolution,
) -> None:
    """At ℏ = -7/10, M = 3: f''(0) is within 0.01 of Howarth's 0.4696.

    Several times tighter than the polynomial-basis Blasius achieves at
    M = 5 (which gets within 0.05). The exponential basis is Liao's
    recommended setup for a reason — the rate of convergence is
    dramatically better.
    """
    value = f_double_prime_at_zero(sol_order_3, sp.Rational(-7, 10))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < sp.Rational(1, 100)


def test_validity_gate_passes_at_best_hbar(sol_order_3: HamSolution) -> None:
    """is_convergent accepts ℏ = -7/10 at M = 3."""
    assert is_convergent(sol_order_3, sp.Rational(-7, 10)) is True


def test_validity_gate_fails_at_hbar_zero(sol_order_3: HamSolution) -> None:
    """At ℏ = 0 the partial sum is u_0 alone; f''(0) = u_0''(0) = alpha = 1, far from Howarth."""
    assert is_convergent(sol_order_3, sp.Integer(0)) is False


def test_exponential_basis_beats_polynomial_basis_at_lower_order() -> None:
    """Compare exponential-basis error at M = 3 against polynomial-basis at M = 5.

    Both solutions exist in the library; this asserts that the
    exponential basis (chosen by Liao's Rule 1 for Blasius) reaches
    its closest f''(0) approximation in fewer HAM iterations than the
    polynomial basis does — the headline result of Stage 11.
    """
    from examples.blasius import solve_to as solve_poly
    from examples.blasius_exponential import solve_to as solve_exp

    sol_poly = solve_poly(5)
    sol_exp = solve_exp(3)
    err_poly = float(
        sp.Abs(
            f_double_prime_at_zero(sol_poly, sp.Rational(-2, 5)) - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    err_exp = float(
        sp.Abs(
            f_double_prime_at_zero(sol_exp, sp.Rational(-7, 10)) - HOWARTH_F_DOUBLE_PRIME_AT_ZERO
        )
    )
    assert err_exp < err_poly / 3  # exponential should be > 3x better


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1(sol_order_3: HamSolution) -> None:
    """u_k for k ≥ 1 carries the ℏ symbol; substitute-late convention holds."""
    for k in range(1, sol_order_3.order + 1):
        assert HBAR in sol_order_3.phi.coeff(k).free_symbols
