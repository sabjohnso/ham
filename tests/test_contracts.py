"""Tests for the algebraic-contract checkers in `ham.contracts`.

The `LinearOperator` data type does not enforce linearity of its
`action` at construction time (a non-linear sympy callable cannot be
detected by inspection without evaluating it). `ham.contracts.verify_linearity`
gives the user an opt-in way to assert
~L[alpha*u + beta*v] = alpha*L[u] + beta*L[v]~ on hand- or
strategy-supplied samples before relying on a HamProblem[sp.Expr] built with
that L.

Liao's Rule of Solution Existence requires that `u_0` satisfy the
/original/ (not just deformation) boundary conditions; the deformation
BCs declared on `problem.L` are typically the homogeneous versions.
`ham.contracts.verify_initial_guess(problem, original_bcs)` is the
sibling checker that pins this contract at problem-construction time.
"""

import pytest
import sympy as sp
from ham.contracts import (
    InitialGuessViolation,
    LinearityViolation,
    verify_initial_guess,
    verify_linearity,
)
from ham.deformation import HamProblem
from ham.nonlinear import NonlinearOperator
from ham.operator import BoundaryCondition, LinearOperator

from tests.strategies import X


def _diff_x(e: sp.Expr) -> sp.Expr:
    return sp.diff(e, X)


def _diff_x_squared(e: sp.Expr) -> sp.Expr:
    return sp.diff(e, X, 2)


def _square_action(e: sp.Expr) -> sp.Expr:
    """A deliberately non-linear action used in negative tests."""
    return e**2


def _polynomial_samples() -> list[tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]]:
    """A few hand-picked (u, v, alpha, beta) quadruples."""
    return [
        (X, sp.Integer(1), sp.Integer(2), sp.Integer(3)),
        (X**2, X + sp.Integer(1), sp.Rational(1, 2), sp.Integer(-1)),
        (X**3 + X, sp.Integer(7), sp.Integer(-2), sp.Rational(5, 3)),
    ]


def test_verify_linearity_accepts_diff_x() -> None:
    """L = d/dx is linear; verify_linearity returns silently."""
    L = LinearOperator(var=X, action=_diff_x)  # noqa: N806  -- Liao's notation
    verify_linearity(L, _polynomial_samples())


def test_verify_linearity_accepts_second_derivative() -> None:
    """L = d^2/dx^2 is linear; verify_linearity passes."""
    L = LinearOperator(var=X, action=_diff_x_squared)  # noqa: N806
    verify_linearity(L, _polynomial_samples())


def test_verify_linearity_with_empty_samples_passes_vacuously() -> None:
    """An empty sample iterable is a no-op (vacuous truth).

    The helper exists to be called in user code; supporting empty
    samples matches Hypothesis-style usage where a parametrised
    sample set might currently be empty.
    """
    L = LinearOperator(var=X, action=_diff_x)  # noqa: N806
    verify_linearity(L, [])


def test_verify_linearity_rejects_nonlinear_action() -> None:
    """A square action fails linearity on the first non-trivial sample."""
    L = LinearOperator(var=X, action=_square_action)  # noqa: N806
    with pytest.raises(LinearityViolation):
        verify_linearity(L, _polynomial_samples())


def test_linearity_violation_is_value_error() -> None:
    """LinearityViolation derives from ValueError (per user decision)."""
    assert issubclass(LinearityViolation, ValueError)


def test_linearity_violation_carries_offending_sample() -> None:
    """The exception payload identifies the offending (u, v, alpha, beta)."""
    L = LinearOperator(var=X, action=_square_action)  # noqa: N806
    sample = (X, sp.Integer(1), sp.Integer(2), sp.Integer(3))
    with pytest.raises(LinearityViolation) as exc_info:
        verify_linearity(L, [sample])
    assert exc_info.value.sample == sample


def test_linearity_violation_carries_lhs_and_rhs() -> None:
    """The exception carries the computed LHS and RHS for the failing sample.

    For the square action with (u, v, alpha, beta) = (X, 1, 2, 3):
      lhs = (2X + 3)^2 = 4X^2 + 12X + 9
      rhs = 2 X^2 + 3 (1)^2 = 2 X^2 + 3
    Both numbers should be exposed via the exception so a debugging
    caller can compare them without reconstructing them.
    """
    L = LinearOperator(var=X, action=_square_action)  # noqa: N806
    sample = (X, sp.Integer(1), sp.Integer(2), sp.Integer(3))
    with pytest.raises(LinearityViolation) as exc_info:
        verify_linearity(L, [sample])
    err = exc_info.value
    expected_lhs = (2 * X + 3) ** 2
    expected_rhs = 2 * X**2 + 3
    assert sp.expand(err.lhs - expected_lhs) == 0
    assert sp.expand(err.rhs - expected_rhs) == 0


def test_verify_linearity_message_names_sample_and_sides() -> None:
    """The default message string mentions the sample and both sides."""
    L = LinearOperator(var=X, action=_square_action)  # noqa: N806
    sample = (X, sp.Integer(1), sp.Integer(2), sp.Integer(3))
    with pytest.raises(LinearityViolation) as exc_info:
        verify_linearity(L, [sample])
    msg = str(exc_info.value)
    assert "alpha" in msg
    assert "beta" in msg


def test_verify_linearity_accepts_custom_equal_comparator() -> None:
    """verify_linearity accepts an `equal` kwarg overriding the default sympy comparator.

    Per PLAN.org D-4, equality lives at the verification site, not on the
    Backend or LinearOperator. The default is sympy-flavoured (sp.expand-
    based); spectral backends (S5b+) inject their own (np.allclose-style)
    at the call site. To prove the kwarg is honoured we use a deliberately
    permissive comparator (`always_equal`) that hides a real linearity
    violation — if verify_linearity were still hard-coded to sp.expand it
    would raise regardless and the test would fail.
    """
    L = LinearOperator(var=X, action=_square_action)  # noqa: N806
    sample = (X, sp.Integer(1), sp.Integer(2), sp.Integer(3))

    with pytest.raises(LinearityViolation):
        verify_linearity(L, [sample])

    def always_equal(a: sp.Expr, b: sp.Expr) -> bool:
        return True

    verify_linearity(L, [sample], equal=always_equal)


def test_verify_linearity_default_equal_is_sympy_expand_based() -> None:
    """The default `equal` reproduces the pre-S3 sp.expand-based behaviour.

    Pins the back-compat invariant: callers who don't pass `equal=...`
    keep the exact behaviour of `sp.expand(lhs - rhs) == 0` they had
    before D-4 made the comparator injectable.
    """
    L = LinearOperator(var=X, action=_diff_x)  # noqa: N806
    # diff(2*X + 3*1) = 2 = 2*diff(X) + 3*diff(1) = 2*1 + 0 = 2, exact match.
    verify_linearity(L, [(X, sp.Integer(1), sp.Integer(2), sp.Integer(3))])


def test_verify_linearity_passes_on_quadratic_drag_example() -> None:
    """The quadratic-drag worked example's L = d/dt is linear; documentary check.

    Demonstrates intended usage: call verify_linearity on a worked
    example's L at problem-construction time to catch a misconfigured
    action before any deformation step runs.
    """
    from examples.quadratic_drag import T, build_problem

    L = build_problem().L  # noqa: N806
    samples = [
        (T, sp.Integer(1), sp.Integer(2), sp.Integer(-1)),
        (T**2, T + sp.Integer(3), sp.Rational(1, 4), sp.Integer(5)),
    ]
    verify_linearity(L, samples)


# --- verify_initial_guess -------------------------------------------------


U = sp.Function("u")
HBAR = sp.Symbol("hbar")


def _trivial_problem(u0: sp.Expr) -> HamProblem[sp.Expr]:
    """A HAM problem skeleton with the given u_0 and a placeholder N.

    verify_initial_guess only reads problem.u0 and problem.L.var, so the
    other fields can be whatever scaffolding is cheapest to build.
    """
    return HamProblem[sp.Expr](
        L=LinearOperator(var=X, action=lambda e: sp.diff(e, X)),
        N=NonlinearOperator(expr=U(X), dependent=U, indep=X),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=u0,
    )


def test_verify_initial_guess_accepts_point_value_bc() -> None:
    """u_0 = 1/2 satisfies u(0) = 1/2 (the logistic case)."""
    problem = _trivial_problem(sp.Rational(1, 2))
    original_bcs = (
        BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Rational(1, 2)),
    )
    verify_initial_guess(problem, original_bcs)


def test_verify_initial_guess_accepts_first_derivative_bc() -> None:
    """u_0 = X satisfies u(0) = 0 AND u'(0) = 1 (multi-BC case)."""
    problem = _trivial_problem(X)
    original_bcs = (
        BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0)),
        BoundaryCondition(point=sp.Integer(0), derivative_order=1, value=sp.Integer(1)),
    )
    verify_initial_guess(problem, original_bcs)


def test_verify_initial_guess_accepts_asymptotic_bc() -> None:
    """u_0 = x - 1 + exp(-x) satisfies u'(infty) = 1 (Blasius-style asymptotic BC).

    Derivative is 1 - exp(-x), with limit 1 as x -> infty. Exercises
    the sp.limit code path; .subs(X, sp.oo) would fail or yield an
    unevaluated expression here.
    """
    var = sp.Symbol("x_positive", positive=True)
    problem = HamProblem[sp.Expr](
        L=LinearOperator(var=var, action=lambda e: sp.diff(e, var)),
        N=NonlinearOperator(expr=U(var), dependent=U, indep=var),
        H=sp.Integer(1),
        hbar=HBAR,
        u0=var - sp.Integer(1) + sp.exp(-var),
    )
    original_bcs = (BoundaryCondition(point=sp.oo, derivative_order=1, value=sp.Integer(1)),)
    verify_initial_guess(problem, original_bcs)


def test_verify_initial_guess_rejects_wrong_value() -> None:
    """u_0 = 1 violates u(0) = 0: InitialGuessViolation raised."""
    problem = _trivial_problem(sp.Integer(1))
    original_bcs = (
        BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0)),
    )
    with pytest.raises(InitialGuessViolation):
        verify_initial_guess(problem, original_bcs)


def test_verify_initial_guess_rejects_wrong_derivative() -> None:
    """u_0 = X**2 violates u'(0) = 1 (the derivative at 0 is 0, not 1)."""
    problem = _trivial_problem(X**2)
    original_bcs = (
        BoundaryCondition(point=sp.Integer(0), derivative_order=1, value=sp.Integer(1)),
    )
    with pytest.raises(InitialGuessViolation):
        verify_initial_guess(problem, original_bcs)


def test_verify_initial_guess_empty_tuple_passes_vacuously() -> None:
    """Empty BC tuple is a no-op (vacuous truth)."""
    problem = _trivial_problem(sp.Integer(42))
    verify_initial_guess(problem, ())


def test_initial_guess_violation_is_value_error() -> None:
    """InitialGuessViolation derives from ValueError (sibling to LinearityViolation)."""
    assert issubclass(InitialGuessViolation, ValueError)


def test_initial_guess_violation_carries_bc_and_actual() -> None:
    """The exception carries the offending bc and the actual computed value."""
    problem = _trivial_problem(sp.Integer(5))
    bc = BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0))
    with pytest.raises(InitialGuessViolation) as exc_info:
        verify_initial_guess(problem, (bc,))
    err = exc_info.value
    assert err.bc == bc
    assert sp.simplify(err.actual - sp.Integer(5)) == 0


def test_verify_initial_guess_message_names_bc_and_actual() -> None:
    """The default message string mentions the bc and the actual value."""
    problem = _trivial_problem(sp.Integer(5))
    bc = BoundaryCondition(point=sp.Integer(0), derivative_order=0, value=sp.Integer(0))
    with pytest.raises(InitialGuessViolation) as exc_info:
        verify_initial_guess(problem, (bc,))
    msg = str(exc_info.value)
    assert "u_0" in msg or "initial guess" in msg.lower()
