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
"""

from collections.abc import Iterable

import sympy as sp

from ham.operator import LinearOperator


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


def verify_linearity(
    L: LinearOperator,  # noqa: N803  -- Liao's notation
    samples: Iterable[tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr]],
) -> None:
    """Assert L[alpha*u + beta*v] == alpha*L[u] + beta*L[v] on each sample.

    Each sample is a four-tuple `(u, v, alpha, beta)` of sympy
    expressions. Returns `None` when every sample satisfies the law;
    raises `LinearityViolation` on the first failing sample, with
    that sample and the computed LHS / RHS attached to the exception.

    Equality is decided symbolically via `sp.expand(lhs - rhs) == 0`.
    Callers needing a stricter or looser comparator can call
    `L.apply` directly and compare via their own equality predicate.

    An empty `samples` iterable is a no-op (vacuous truth), matching
    Hypothesis-style usage where a parametrised sample set might be
    empty for some configurations.
    """
    for sample in samples:
        u, v, alpha, beta = sample
        lhs = L.apply(alpha * u + beta * v)
        rhs = alpha * L.apply(u) + beta * L.apply(v)
        if sp.expand(lhs - rhs) != 0:
            raise LinearityViolation(sample=sample, lhs=lhs, rhs=rhs)
