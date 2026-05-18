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
    ORIGINAL_BCS,
    build_problem,
    f_double_prime_at_zero,
    is_convergent,
    solve_to,
)
from ham.contracts import verify_initial_guess
from ham.solver import HamSolution


def test_initial_guess_satisfies_original_bcs_via_verify_helper() -> None:
    """`ham.contracts.verify_initial_guess` accepts u_0 against the three BCs.

    Documentary cross-check that complements the explicit
    `test_initial_guess_satisfies_truncated_bcs` test by running the
    same three pointwise checks through the public helper.
    """
    verify_initial_guess(build_problem(), ORIGINAL_BCS)


@pytest.fixture(scope="module")
def sol_order_5() -> HamSolution[sp.Expr]:
    """One Blasius solve at M = 5, reused across the tests below."""
    return solve_to(5)


def test_initial_guess_satisfies_truncated_bcs() -> None:
    """u_0 = η^2 / (2 η_max) must satisfy f(0)=0, f'(0)=0, f'(η_max)=1."""
    problem = build_problem()
    u0 = problem.u0
    assert sp.simplify(u0.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(u0, ETA).subs(ETA, ETA_MAX) - 1) == 0


def test_partial_sum_satisfies_f_zero_is_zero(sol_order_5: HamSolution[sp.Expr]) -> None:
    """For every ℏ, the partial sum vanishes at η = 0 (BC f(0) = 0)."""
    partial = sol_order_5.partial_sum()
    assert sp.simplify(partial.subs(ETA, 0)) == 0


def test_partial_sum_satisfies_f_prime_zero_is_zero(sol_order_5: HamSolution[sp.Expr]) -> None:
    """For every ℏ, the partial sum has f'(0) = 0 (BC f'(0) = 0)."""
    partial = sol_order_5.partial_sum()
    assert sp.simplify(sp.diff(partial, ETA).subs(ETA, 0)) == 0


def test_partial_sum_satisfies_truncated_asymptotic_bc(sol_order_5: HamSolution[sp.Expr]) -> None:
    """f'(η_max) = 1 exactly, for every ℏ — built in by construction.

    u_0 satisfies f'(η_max) = 1, and every higher u_k satisfies the
    homogeneous version of the same BC, so the sum is exactly 1 for
    any ℏ value.
    """
    partial = sol_order_5.partial_sum()
    f_prime_at_etamax = sp.diff(partial, ETA).subs(ETA, ETA_MAX)
    assert sp.simplify(f_prime_at_etamax - 1) == 0


def test_f_double_prime_at_zero_diverges_at_hbar_neg_one(
    sol_order_5: HamSolution[sp.Expr],
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
    sol_order_5: HamSolution[sp.Expr],
) -> None:
    """At a well-chosen ℏ (here -2/5), f''(0) is within 0.1 of Howarth's 0.4696.

    The polynomial-basis HAM converges slowly compared to Liao's
    exponential-basis treatment, so the tolerance is loose: 0.1
    absolute, about a 20% relative error.
    """
    value = f_double_prime_at_zero(sol_order_5, sp.Rational(-2, 5))
    assert sp.Abs(value - HOWARTH_F_DOUBLE_PRIME_AT_ZERO) < sp.Rational(1, 10)


def test_validity_gate_fails_at_hbar_neg_one(sol_order_5: HamSolution[sp.Expr]) -> None:
    """is_convergent rejects ℏ = -1 where the series diverges."""
    assert is_convergent(sol_order_5, sp.Integer(-1)) is False


def test_validity_gate_passes_at_best_hbar(sol_order_5: HamSolution[sp.Expr]) -> None:
    """is_convergent accepts ℏ = -2/5 where f''(0) is close to Howarth."""
    assert is_convergent(sol_order_5, sp.Rational(-2, 5)) is True


def test_hbar_remains_symbolic_in_u_k_for_k_geq_1(sol_order_5: HamSolution[sp.Expr]) -> None:
    """u_k for k >= 1 carries the ℏ symbol; substitute-late convention holds."""
    for k in range(1, sol_order_5.order + 1):
        assert HBAR in sol_order_5.phi.coeff(k).free_symbols


# --- Spectral substrate (SHAM, truncated domain) -------------------------


def test_spectral_solve_recovers_sympy_partial_sum_at_same_hbar() -> None:
    """SHAM and the sympy path agree on Blasius at a fixed ℏ.

    The two substrates solve the same truncated HamProblem (BCs at
    eta=0 and eta=eta_max with the row-displacement convention for
    the double BC at eta=0); their grid values should match to a
    tolerance set by the spectral grid's resolution.
    """
    import numpy as np
    from examples.blasius import (
        ETA,
        solve_to,
        solve_to_spectral,
    )
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=40, domain=(0.0, 10.0))
    sol_sym = solve_to(2)
    sol_spec = solve_to_spectral(2, grid=grid, hbar_value=sp.Float(-0.4))

    partial_sym = sol_sym.evaluate_at_hbar(sp.Float(-0.4))
    sym_at_grid = np.array(
        [float(partial_sym.subs(ETA, sp.Float(n))) for n in grid.nodes],
        dtype=np.float64,
    )
    np.testing.assert_allclose(sol_spec.partial_sum(), sym_at_grid, atol=1e-10)


def test_spectral_f_double_prime_at_zero_close_to_howarth_at_best_hbar() -> None:
    """SHAM at the best ℏ on the sweep grid lands close to Howarth's f''(0).

    Same convergence story as the sympy path: polynomial-basis Blasius
    is slow; at M=5 the best ℏ on the sweep [-0.2, -0.4, -0.6, -0.8]
    is -0.4 with f''(0) ≈ 0.418 (~0.05 from Howarth's 0.4696). That
    matches the sympy analyze() output term for term.
    """
    from examples.blasius import (
        HOWARTH_F_DOUBLE_PRIME_AT_ZERO,
        f_double_prime_at_zero_spectral,
        solve_to_spectral,
    )
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=40, domain=(0.0, 10.0))
    howarth = float(HOWARTH_F_DOUBLE_PRIME_AT_ZERO)
    fdd_sweep = []
    for h in (-0.2, -0.4, -0.6, -0.8):
        sol = solve_to_spectral(5, grid=grid, hbar_value=sp.Float(h))
        fdd_sweep.append((h, f_double_prime_at_zero_spectral(sol, grid)))
    best_h, best_fdd = min(fdd_sweep, key=lambda p: abs(p[1] - howarth))
    assert best_h == -0.4
    assert abs(best_fdd - howarth) < 0.1


def test_spectral_inverter_handles_double_bc_at_same_point() -> None:
    """Two BCs at the same boundary node solve correctly via row displacement.

    Pins the row-displacement fix in `spectral_inverter`: Blasius has
    `f(0)=0` and `f'(0)=0` both at η=0 (the last grid node in
    Trefethen ordering). Without displacement, the second BC
    overwrites the first and the system is wrong. With displacement,
    the second BC goes to the adjacent row N-1, both BCs are
    enforced, and the partial sum at η=0 satisfies u_partial(0)=0
    along with u_partial'(0)=0.
    """
    from examples.blasius import solve_to_spectral
    from ham.grids import ChebGLGrid

    grid = ChebGLGrid(N=40, domain=(0.0, 10.0))
    sol = solve_to_spectral(3, grid=grid, hbar_value=sp.Float(-0.4))
    partial = sol.partial_sum()
    # Last node is eta=0 in Trefethen ordering.
    assert abs(partial[-1]) < 1e-10
    # f'(0) via D @ partial at the last node.
    fprime_at_zero = (grid.differentiation_matrix @ partial)[-1]
    assert abs(float(fprime_at_zero)) < 1e-10
