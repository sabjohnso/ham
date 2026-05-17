"""Opt-in algebraic-contract checkers for HAM problem components.

The data types in `ham.operator`, `ham.nonlinear`, etc. accept any
sympy callable that /claims/ to satisfy the relevant algebraic law
(linearity of `L`, polynomial-in-u structure of `N`, ...). The
library does not enforce those laws at construction time because

  - the user's action may be symbolic and inspectable only by
    evaluation;
  - the cost of enforcement on every solver step is wasteful when
    the law is obvious to the implementer;
  - enforcement at construction would force users to materialise
    samples even when their action is trivially correct.

This module provides opt-in helpers users can call at problem-
construction time to assert the law on hand- or strategy-supplied
samples. Each helper is pure (no I/O, no global state) and raises a
specific exception with the offending sample attached so callers can
debug without re-running the action.

Currently exposed:

  - `verify_linearity(L, samples)` — assert that
    `L[alpha*u + beta*v] == alpha*L[u] + beta*L[v]` on each sample.
    Raises `LinearityViolation` on the first failing sample.
  - `verify_initial_guess(problem, original_bcs)` — assert that
    `problem.u0` satisfies each original (not deformation) boundary
    condition pointwise. Raises `InitialGuessViolation` on the first
    failing BC. Encodes Liao's "Rule of Solution Existence":
    `u_0` must already satisfy the original problem's BCs so that
    the deformation chain (which produces homogeneous higher
    iterates) yields a partial sum that still satisfies them.
"""

from collections.abc import Callable, Iterable

import sympy as sp

from ham.deformation import HamProblem
from ham.operator import BoundaryCondition, LinearOperator
from ham.series import SupportsCoefficientArith


class LinearityViolation(ValueError):  # noqa: N818  -- domain term, not "Error"
    """Raised when a `LinearOperator.action` fails to satisfy linearity.

    Subclass of `ValueError` because the failure indicates invalid
    user input (an action that does not satisfy its declared contract)
    rather than an internal invariant violation.

    Attributes:
      sample: the four-tuple `(u, v, alpha, beta)` on which the law
              failed.
      lhs:    the computed `L[alpha*u + beta*v]`.
      rhs:    the computed `alpha*L[u] + beta*L[v]`.
    """

    def __init__(
        self,
        sample: tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr],
        lhs: sp.Expr,
        rhs: sp.Expr,
    ) -> None:
        u, v, alpha, beta = sample
        super().__init__(
            f"LinearOperator action failed linearity on sample "
            f"u={u!r}, v={v!r}, alpha={alpha!r}, beta={beta!r}: "
            f"L[alpha*u + beta*v] = {lhs!r}; "
            f"alpha*L[u] + beta*L[v] = {rhs!r}."
        )
        self.sample = sample
        self.lhs = lhs
        self.rhs = rhs


def _sympy_equal(a: sp.Expr, b: sp.Expr) -> bool:
    """Default linearity comparator for the sympy backend: `sp.expand(a-b) == 0`.

    Lives at the verification site (not on `Backend` or `LinearOperator`)
    per PLAN.org D-4 — `LinearOperator` stays substrate-agnostic, and
    "close enough" is the caller's choice. The default reproduces the
    pre-S3 behaviour for back-compat with all existing call sites.
    """
    return bool(sp.expand(a - b) == 0)


def verify_linearity[C: SupportsCoefficientArith](
    L: LinearOperator[C],  # noqa: N803  -- Liao's notation
    samples: Iterable[tuple[C, C, C, C]],
    *,
    equal: Callable[[C, C], bool] = _sympy_equal,
) -> None:
    """Assert L[alpha*u + beta*v] == alpha*L[u] + beta*L[v] on each sample.

    Each sample is a four-tuple `(u, v, alpha, beta)` of coefficient
    values. Returns `None` when every sample satisfies the law; raises
    `LinearityViolation` on the first failing sample, with that sample
    and the computed LHS / RHS attached to the exception.

    Equality is decided by the injected `equal` comparator (PLAN.org D-4):
    the default `_sympy_equal` reproduces the pre-S3 `sp.expand(a-b)==0`
    behaviour, so sympy callers can omit it. SHAM call sites (S6+) pass
    `np.allclose` for float-spectral and a coefficient-wise sp.expand
    closure for the sympy-spectral scalar.

    An empty `samples` iterable is a no-op (vacuous truth), matching
    Hypothesis-style usage where a parametrised sample set might be
    empty for some configurations.
    """
    for sample in samples:
        u, v, alpha, beta = sample
        lhs = L.apply(alpha * u + beta * v)
        rhs = alpha * L.apply(u) + beta * L.apply(v)
        if not equal(lhs, rhs):
            raise LinearityViolation(sample=sample, lhs=lhs, rhs=rhs)


class InitialGuessViolation(ValueError):  # noqa: N818  -- domain term, not "Error"
    """Raised when `problem.u0` fails to satisfy an original boundary condition.

    Subclass of `ValueError` for the same reason as `LinearityViolation`:
    the failure indicates invalid user input rather than an internal
    invariant violation.

    Attributes:
      bc:     the offending `BoundaryCondition`.
      actual: the computed value of `u_0^(k)(point)` (where `k` is
              `bc.derivative_order` and `point` is `bc.point`).
    """

    def __init__(self, bc: BoundaryCondition, actual: sp.Expr) -> None:
        super().__init__(
            f"u_0 fails to satisfy the original boundary condition "
            f"u^({bc.derivative_order})({bc.point!r}) = {bc.value!r}: "
            f"actual value is {actual!r}."
        )
        self.bc = bc
        self.actual = actual


def verify_initial_guess(
    problem: HamProblem[sp.Expr],
    original_bcs: Iterable[BoundaryCondition],
) -> None:
    """Assert that `problem.u0` satisfies each original BC pointwise.

    Liao's Rule of Solution Existence requires that the initial guess
    `u_0` already satisfy the original boundary conditions; the
    higher-order deformation iterates `u_m (m >= 1)` are produced
    against /homogeneous/ versions of those BCs, so the partial sum
    `u^{(M)} = sum_k u_k` satisfies the original BCs iff `u_0` does.

    For each `bc` in `original_bcs`, this helper:

      1. Differentiates `u_0` w.r.t. `problem.L.var` to order
         `bc.derivative_order`.
      2. Evaluates the derivative at `bc.point`:
         - via `sp.limit` when `bc.point` is infinite, since `.subs`
           on `sp.oo` is unreliable for non-trivial expressions;
         - via `.subs` otherwise.
      3. Asserts `sp.simplify(actual - bc.value) == 0`.

    Raises `InitialGuessViolation` on the first failing `bc`, with
    the `bc` and the `actual` computed value attached. An empty
    `original_bcs` iterable is a no-op.

    Note: the BCs declared on `problem.L` are the /deformation/ BCs,
    which are typically the homogeneous versions of `original_bcs`.
    This helper exists because the original (possibly non-homogeneous)
    BCs are not encoded on `HamProblem[sp.Expr]` itself — they live in the
    user's mental model and must be supplied separately.
    """
    var = problem.L.var
    u0 = problem.u0
    for bc in original_bcs:
        derivative = sp.diff(u0, var, bc.derivative_order)
        if bc.point.is_infinite:
            actual = sp.limit(derivative, var, bc.point)
        else:
            actual = derivative.subs(var, bc.point)
        if sp.simplify(actual - bc.value) != 0:
            raise InitialGuessViolation(bc=bc, actual=actual)
