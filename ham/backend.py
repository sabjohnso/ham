"""Coefficient-substrate abstraction shared by symbolic and spectral HAM.

A `Backend[C]` packages the operations that `Series[C]` and
`NonlinearOperator` cannot perform polymorphically through Python `+
- *` arithmetic on the coefficient type `C`:

  - identities: `zero()` and `one()`;
  - lifting: `lift_xonly(expr)` turns an x-only `sympy.Expr` into a
    coefficient (identity for the sympy backend, evaluation on grid
    nodes for the spectral backend);
  - x-calculus: `diff_x(c, k)` and `integrate_x(c)`.

The Backend does NOT carry an equality comparator: per PLAN.org D-4,
equality is supplied at the verification site (sp.expand for sympy,
np.allclose for float-spectral, coefficient-wise sp.expand for sympy-
spectral). Keeping equality out of the Backend lets `LinearOperator`
stay backend-agnostic — it depends on `Callable[[C], C]`, not on the
substrate-specific notion of "close enough".

S0 wires the dataclass and the sympy backend only; downstream stages
introduce `SpectralBackend(grid, scalar)` against the same protocol.
"""

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class Backend[C]:
    """The five operations a coefficient substrate must supply.

    See module docstring for the contract on each field. Backends are
    constructed by the factory functions in this module (e.g.
    `SympyBackend(indep)`); the dataclass itself is the protocol-
    carrying record that consumers depend on.
    """

    zero: Callable[[], C]
    one: Callable[[], C]
    lift_xonly: Callable[[sp.Expr], C]
    diff_x: Callable[[C, int], C]
    integrate_x: Callable[[C], C]


def SympyBackend(indep: sp.Symbol) -> Backend[sp.Expr]:  # noqa: N802 -- constructor-style factory
    """Build a `Backend[sp.Expr]` whose x-calculus is keyed on `indep`.

    Coefficients are already sympy expressions, so `lift_xonly` is the
    identity. `integrate_x` is the definite integral from 0 to `indep`
    so the antiderivative vanishes at the lower bound by construction
    (matching the SHAM convention and the Volterra integral that
    `NonlinearOperator._compile_integral` uses).
    """

    def zero() -> sp.Expr:
        return sp.Integer(0)

    def one() -> sp.Expr:
        return sp.Integer(1)

    def lift_xonly(expr: sp.Expr) -> sp.Expr:
        return expr

    def diff_x(c: sp.Expr, k: int) -> sp.Expr:
        return sp.diff(c, indep, k)

    def integrate_x(c: sp.Expr) -> sp.Expr:
        return sp.integrate(c, (indep, sp.Integer(0), indep))

    return Backend(
        zero=zero,
        one=one,
        lift_xonly=lift_xonly,
        diff_x=diff_x,
        integrate_x=integrate_x,
    )
