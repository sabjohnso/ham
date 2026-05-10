"""The auxiliary linear operator L for HAM.

The HAM m-th order deformation equation has the shape

    L[u_m - chi_m * u_{m-1}] = hbar * H(x) * R_m(u_0, ..., u_{m-1})

so L participates in HAM at two points: forwards, where L is applied to a
known expression, and backwards, where L is inverted against a known
right-hand side to obtain u_m. Both directions live in this module.

`LinearOperator` bundles three things:

  - `var`:     the independent variable (e.g. sympy.Symbol('x')).
  - `action`:  a Callable[[Expr], Expr] computing L[u] forward.
  - `bcs`:     a tuple of BoundaryCondition values that L^{-1} must respect.
               In HAM the deformation BCs are homogeneous (u_0 already
               satisfies the original problem), so the default `value` is 0.
  - `inverter` (optional): a Callable[[Expr], Expr] that solves L[u] = rhs
               under the declared BCs. When None, `L.invert` raises
               NotImplementedError; Stage 2c will install a sympy.dsolve
               default. Hand-coded inverters for canonical operators (such
               as `antiderivative` below) live alongside this module.

Linearity of `action` and consistency of `inverter` with `bcs` are the
caller's contract; the wrapper does not enforce them. Property tests
verify both for known operators.
"""

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from ham.series import QSeries


@dataclass(frozen=True)
class BoundaryCondition:
    """A single boundary/initial condition: u^(k)(point) = value."""

    point: sp.Expr
    derivative_order: int
    value: sp.Expr = sp.S.Zero


@dataclass(frozen=True)
class LinearOperator:
    """A linear operator on sympy expressions, with forward and inverse paths.

    See module docstring for the contract on `action`, `bcs`, and `inverter`.
    """

    var: sp.Symbol
    action: Callable[[sp.Expr], sp.Expr]
    bcs: tuple[BoundaryCondition, ...] = ()
    inverter: Callable[[sp.Expr], sp.Expr] | None = None

    def apply(self, u: sp.Expr) -> sp.Expr:
        """Apply L forward to a sympy expression in `var`."""
        return self.action(u)

    def apply_series(self, s: QSeries) -> QSeries:
        """Apply L coefficient-wise to a QSeries (the q/x gluing law)."""
        return s.map_coeffs(self.action)

    def invert(self, rhs: sp.Expr) -> sp.Expr:
        """Solve L[u] = rhs subject to the declared BCs, returning u."""
        if self.inverter is None:
            raise NotImplementedError(
                "No inverter installed. Provide `inverter=...` at construction "
                "time, or wait for the Stage 2c sympy.dsolve default."
            )
        return self.inverter(rhs)


def antiderivative(var: sp.Symbol, t0: sp.Expr = sp.S.Zero) -> Callable[[sp.Expr], sp.Expr]:
    """Canonical-case inverter for L = d/d{var} with u({var}={t0}) = 0.

    Returns the closed-form definite-integration inverter
    `rhs |-> integral_{t0}^{var} rhs(s) ds`, which automatically satisfies
    the point condition u(t0) = 0 by construction.
    """

    def _invert(rhs: sp.Expr) -> sp.Expr:
        return sp.integrate(rhs, (var, t0, var))

    return _invert
