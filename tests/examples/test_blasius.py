"""Regression tests for the Blasius boundary-layer example.

Pins HAM output against Howarth's tabulated f''(0) ≈ 0.4696 and
against the truncated boundary conditions u_0 was constructed to
satisfy. Exercises a polynomial-basis HAM run where ℏ = -1 fails —
the third worked example to demonstrate that the right ℏ is
problem-specific.
"""

import pytest
import sympy as sp
from examples.blasius import (
    ETA,
    ETA_MAX,
    HBAR,
    HOWARTH_F_DOUBLE_PRIME_AT_ZERO,
    build_problem,
    f_double_prime_at_zero,
    is_convergent,
    solve_to,
)
from ham.solver import HamSolution


@pytest.fixture(scope="module")
def sol_order_5() -> HamSolution:
    """One Blasius solve at M = 5, reused across the tests below."""
    return solve_to(5)


def test_initial_guess_satisfies_truncated_bcs() -> None:
    """u_0 = η^2 / (2 η_max) must satisfy f(0)=0, f'(0)=0, f'(η_max)=1."""
    problem = build_problem()
    u0 = problem.u0
    assert sp.simplify(u0.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, ETA_MAX) - 1) == 0


def test_partial_sum_satisfies_f_zero_is_zero(sol_order_5: HamSolution) -> None:
    """For every ℏ, the partial sum vanishes at η = 0 (BC f(0) = 0)."""
    partial = sol_order_5.partial_sum()
    assert sp.simplify(partial.subs(ETA, 0)) == 0


def test_partial_sum_satisfies_f_prime_zero_is_zero(sol_order_5: HamSolution) -> None:
    """For every ℏ, the partial sum has f'(0) = 0 (BC f'(0) = 0)."""
    partial = sol_order_5.partial_sum()
    assert sp.simplify(sp.diff(partial, ETA).subs(ETA, 0)) == 0


def test_partial_sum_satisfies_truncated_asymptotic_bc(sol_order_5: HamSolution) -> None:
    """f'(η_max) = 1 exactly, for every ℏ — built in by construction.

    u_0 satisfies f'(η_max) = 1, and every higher u_k satisfies the
    homogeneous version of the same BC, so the sum is exactly 1 for
    any ℏ value.
    """
    partial = sol_order_5.partial_sum()
    f_prime_at_etamax = sp.diff(partial, ETA).subs(ETA, ETA_MAX)
    assert sp.simplify(f_prime_at_etamax - 1) == 0


def test_f_double_prime_at_zero_diverges_at_hbar_neg_one(
    sol_order_5: HamSolution,
) -> None:
    """At ℏ = -1, M = 5, f''(0) is far from Howarth's reference.

    The polynomial-basis Blasius series oscillates and diverges at
    ℏ = -1 — this is the example's pedagogical point. At M = 5 the
    f''(0) value is around -3.7, an order of magnitude larger than
    the Howarth reference 0.4696 and the wrong sign.
    """
    value = f_double_prime_at_zero(sol_order_5, sp.Integer(-1))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) > 1


def test_f_double_prime_at_zero_close_to_howarth_at_best_hbar(
    sol_order_5: HamSolution,
) -> None:
    """At a well-chosen ℏ (here -2/5), f''(0) is within 0.1 of Howarth's 0.4696.

    The polynomial-basis HAM converges slowly compared to Liao's
    exponential-basis treatment, so the tolerance is loose: 0.1
    absolute, about a 20% relative error.
    """
    value = f_double_prime_at_zero(sol_order_5, sp.Rational(-2, 5))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < sp.Rational(1, 10)


def test_validity_gate_fails_at_hbar_neg_one(sol_order_5: HamSolution) -> None:
    """is_convergent rejects ℏ = -1 where the series diverges."""
    assert is_convergent(sol_order_5, sp.Integer(-1)) is False


def test_validity_gate_passes_at_best_hbar(sol_order_5: HamSolution) -> None:
    """is_convergent accepts ℏ = -2/5 where f''(0) is close to Howarth."""
    assert is_convergent(sol_order_5, sp.Rational(-2, 5)) is True


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1(sol_order_5: HamSolution) -> None:
    """u_k for k >= 1 carries the ℏ symbol; substitute-late convention holds."""
    for k in range(1, sol_order_5.order + 1):
        assert HBAR in sol_order_5.phi.coeff(k).free_symbols
