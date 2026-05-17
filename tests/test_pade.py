"""Tests for homotopy-Padé acceleration (Stage 8).

The classical [L/M] Padé approximant in q evaluated at q = 1; an
alternative to `HamSolution[sp.Expr].partial_sum()` that often converges when
the bare partial sum does not. Reference identities:

  - Geometric problem (u' = u², u(0) = 1) at ℏ = -1: [0/1] Padé from
    just u_0, u_1 = 1, x gives the exact closed-form 1/(1-x).
  - Exp problem (u' = u, u(0) = 1) at ℏ = -1 order 4: [2/2] Padé
    equals the classical (1 + x/2 + x²/12)/(1 - x/2 + x²/12).
"""

from functools import reduce
from operator import add

import pytest
import sympy as sp
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator
from ham.pade import homotopy_pade
from ham.solver import solve

from tests.strategies import X

U = sp.Function("u")
HBAR = sp.Symbol("hbar")


def _ivp_operator() -> LinearOperator[sp.Expr]:
    """L = d/dx with homogeneous BC u(0) = 0."""
    return LinearOperator(
        var=X,
        action=lambda e: sp.diff(e, X),
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )


def _exp_problem() -> HamProblem[sp.Expr]:
    """u' = u, u(0) = 1; exact solution exp(x)."""
    return HamProblem[sp.Expr](
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X), dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


def _geometric_problem() -> HamProblem[sp.Expr]:
    """u' = u², u(0) = 1; exact solution 1/(1-x)."""
    return HamProblem[sp.Expr](
        L=_ivp_operator(),
        N=NonlinearOperator(expr=U(X).diff(X) - U(X) ** 2, dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=sp.Integer(1),
    )


# --- M = 0: degenerate Padé equals the truncated partial sum --------------


def test_pade_with_zero_denominator_degree_equals_truncated_partial_sum() -> None:
    """[L/0] Padé = sum of phi.coeff(0..L) (no acceleration, just truncation)."""
    sol = solve(_exp_problem(), order=4)
    for ldeg in range(sol.order + 1):
        truncated = reduce(
            add,
            (sol.phi.coeff(k) for k in range(ldeg + 1)),
            sp.Integer(0),
        )
        result = homotopy_pade(sol, ldeg, 0)
        assert sp.simplify(result - truncated) == 0


def test_pade_at_max_l_zero_m_equals_full_partial_sum() -> None:
    """[N/0] Padé at L = solution.order matches solution.partial_sum()."""
    sol = solve(_exp_problem(), order=4)
    result = homotopy_pade(sol, sol.order, 0)
    assert sp.simplify(result - sol.partial_sum()) == 0


# --- Geometric problem: exact recovery from minimal HAM coefficients ------


def test_pade_recovers_geometric_solution_from_zero_over_one_at_order_1() -> None:
    """[0/1] Padé of {u_0=1, u_1=x} at ℏ=-1 collapses to 1/(1-x) exactly.

    The geometric series has perfect Padé structure: P(q)=1, Q(q)=1-xq,
    P(1)/Q(1) = 1/(1-x). Just two HAM steps recover the closed form.
    """
    sol = solve(_geometric_problem(), order=1)
    result = homotopy_pade(sol, 0, 1, sp.Integer(-1))
    expected = sp.Integer(1) / (sp.Integer(1) - X)
    assert sp.simplify(result - expected) == 0


def test_pade_recovers_geometric_solution_at_non_degenerate_orders() -> None:
    """Every non-degenerate [L/M] Padé of the truncated geometric gives 1/(1-x).

    The geometric series 1 + x + x² + ... has the exact closed form
    1/(1 - xq) in q, so [L/1] for any L ≥ 0 and [0/M] for any M ≥ 1
    both recover this rational function from the truncated HAM
    coefficients. Other [L/M] choices (e.g. [1/2], [2/2]) are
    /degenerate/: the M-by-M denominator system becomes singular because
    the geometric pole already lives at one specific [L/M], and
    higher-order denominators introduce a redundant equation. Those
    degenerate cases must be requested deliberately, and they raise
    sympy's NonInvertibleMatrixError rather than silently returning
    a wrong answer.
    """
    sol = solve(_geometric_problem(), order=4)
    expected = sp.Integer(1) / (sp.Integer(1) - X)
    cases = [(0, 1), (1, 1), (2, 1), (3, 1), (0, 2), (0, 3), (0, 4)]
    for ldeg, mdeg in cases:
        result = homotopy_pade(sol, ldeg, mdeg, sp.Integer(-1))
        assert sp.simplify(result - expected) == 0, (ldeg, mdeg)


def test_pade_singular_denominator_propagates_sympy_error() -> None:
    """A degenerate [L/M] choice (here [2/2] on the geometric series) raises.

    The Padé denominator system has determinant identically zero for
    this combination — falling back silently would give the caller a
    nan or a wrong rational function. We propagate sympy's
    NonInvertibleMatrixError as documented.
    """
    sol = solve(_geometric_problem(), order=4)
    with pytest.raises(sp.matrices.exceptions.NonInvertibleMatrixError):
        homotopy_pade(sol, 2, 2, sp.Integer(-1))


# --- Exp problem: classical [2/2] Padé approximant ------------------------


def test_pade_exp_problem_two_over_two_equals_classical_pade() -> None:
    """[2/2] Padé of exp's Taylor truncation at order 4 is (1+x/2+x²/12)/(1-x/2+x²/12)."""
    sol = solve(_exp_problem(), order=4)
    result = homotopy_pade(sol, 2, 2, sp.Integer(-1))
    numer = sp.Integer(1) + X / sp.Integer(2) + X**2 / sp.Integer(12)
    denom = sp.Integer(1) - X / sp.Integer(2) + X**2 / sp.Integer(12)
    expected = numer / denom
    assert sp.simplify(result - expected) == 0


def test_pade_exp_problem_one_over_one_equals_classical_pade() -> None:
    """[1/1] Padé of exp's Taylor truncation at order 2 is (1+x/2)/(1-x/2)."""
    sol = solve(_exp_problem(), order=2)
    result = homotopy_pade(sol, 1, 1, sp.Integer(-1))
    expected = (sp.Integer(1) + X / sp.Integer(2)) / (sp.Integer(1) - X / sp.Integer(2))
    assert sp.simplify(result - expected) == 0


# --- Validation ------------------------------------------------------------


def test_pade_rejects_negative_numerator_degree() -> None:
    """Negative numerator degree raises ValueError."""
    sol = solve(_exp_problem(), order=4)
    with pytest.raises(ValueError, match="non-negative"):
        homotopy_pade(sol, -1, 0)


def test_pade_rejects_negative_denominator_degree() -> None:
    """Negative denominator degree raises ValueError."""
    sol = solve(_exp_problem(), order=4)
    with pytest.raises(ValueError, match="non-negative"):
        homotopy_pade(sol, 0, -1)


def test_pade_rejects_total_degree_above_solution_order() -> None:
    """L + M exceeding solution.order raises ValueError."""
    sol = solve(_exp_problem(), order=4)
    with pytest.raises(ValueError, match=r"solution\.order"):
        homotopy_pade(sol, 3, 3)


# --- Symbolic ℏ mode ------------------------------------------------------


def test_pade_retains_hbar_when_no_value_supplied() -> None:
    """With hbar_value=None the Padé result keeps ℏ symbolic."""
    sol = solve(_exp_problem(), order=4)
    result = homotopy_pade(sol, 2, 2)
    assert HBAR in result.free_symbols


def test_pade_symbolic_then_substituted_matches_direct_call() -> None:
    """homotopy_pade(sol, L, M).subs(ℏ, v) == homotopy_pade(sol, L, M, v)."""
    sol = solve(_geometric_problem(), order=2)
    symbolic = homotopy_pade(sol, 1, 1)
    direct = homotopy_pade(sol, 1, 1, sp.Integer(-1))
    via_subs = symbolic.subs(HBAR, sp.Integer(-1))
    assert sp.simplify(via_subs - direct) == 0


# --- Spectral substrate rejection (PLAN.org S9) ---------------------------


def test_pade_rejects_spectral_solution() -> None:
    """homotopy_pade on a spectral solution raises NotImplementedError.

    Pinning the S9 decision: the spectral release ships with
    `homotopy_pade` sympy-only. A spectral solution's coefficients are
    grid vectors and the Padé construction would need a block-
    structured linear solve over them — a real follow-up but not the
    SHAM headline feature.
    """
    from ham.grids import ChebGLGrid
    from ham.operator import BoundaryCondition
    from ham.spectral import SpectralBackend, spectral_linear_operator

    grid = ChebGLGrid(N=8, domain=(0.0, 1.0))
    backend = SpectralBackend(grid, indep=X, scalar="float")
    L = spectral_linear_operator(  # noqa: N806
        U(X).diff(X),
        dependent=U,
        indep=X,
        grid=grid,
        scalar="float",
        bcs=(BoundaryCondition(point=sp.Integer(0), derivative_order=0),),
    )
    problem = HamProblem(
        L=L,
        N=NonlinearOperator(expr=U(X).diff(X) - U(X), dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=sp.Float(-1.0),
        u0=sp.Integer(1),
    )
    sol = solve(problem, order=3, backend=backend)
    with pytest.raises(NotImplementedError, match="spectral"):
        homotopy_pade(sol, 1, 1)
