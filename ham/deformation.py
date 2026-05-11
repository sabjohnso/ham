"""The HAM deformation-equation builder (Stage 4).

Given an HAM problem (L, N, H, hbar, u_0) and a partial homotopy series
phi_{m-1} = u_0 + u_1 q + ... + u_{m-1} q^{m-1}, this module assembles
the right-hand side of the m-th-order deformation equation

    L[u_m - chi_m * u_{m-1}] = hbar * H(x) * R_m

with R_m the Taylor coefficient [q^{m-1}] N[phi]. The solve step itself
— calling L.invert on the RHS and recovering u_m — is Stage 5; Stage 4
stops at "produce the RHS".
"""

from dataclasses import dataclass

import sympy as sp

from ham.nonlinear import NonlinearOperator
from ham.operator import LinearOperator
from ham.series import QSeries


def chi_m(m: int) -> int:
    """The HAM heaviside: 0 for m <= 1, 1 for m >= 2.

    Drops the L[u_{m-1}] term out of the m = 1 deformation equation
    since u_0 is given rather than solved-for; switches on for m >= 2,
    where the recurrence relates u_m back to u_{m-1}.
    """
    return 0 if m <= 1 else 1


@dataclass(frozen=True)
class HamProblem:
    """The data of one HAM deformation problem.

    Fields:
      L:    auxiliary linear operator (with homogeneous BCs declared).
      N:    nonlinear operator wrapping the original problem N[u] = 0.
      H:    auxiliary function H(x), a sympy Expr (not an operator).
      hbar: convergence-control parameter, typically `sp.Symbol('hbar')`.
      u0:   initial guess u_0(x), a sympy Expr in L.var.

    Field names follow Liao's notation directly so the implementation
    reads as the math; PEP8 considerations are subordinated to that.
    """

    L: LinearOperator
    N: NonlinearOperator
    H: sp.Expr
    hbar: sp.Expr
    u0: sp.Expr

    def r_m(self, phi: QSeries, m: int) -> sp.Expr:
        """Liao's R_m: the Taylor coefficient `[q^{m-1}] N[phi]`.

        Stage 3's causality guarantees that only `phi.coeff(0..m-1)`
        enters the result, so `phi` may carry more coefficients than
        the deformation index needs.

        Raises ValueError when `m < 1` (deformation equations are
        indexed from 1) or `phi.order < m - 1` (phi does not yet
        carry u_{m-1} — silent zeros would be a quiet bug).
        """
        if m < 1:
            raise ValueError(
                f"r_m requires m >= 1 (HAM deformation equations are 1-indexed); got m = {m}."
            )
        if phi.order < m - 1:
            raise ValueError(
                f"r_m at index m = {m} requires phi.order >= {m - 1}; "
                f"got phi.order = {phi.order}. Extend phi or lower m."
            )
        return self.N.apply_series(phi).coeff(m - 1)

    def rhs_m(self, phi: QSeries, m: int) -> sp.Expr:
        """Right-hand side of the m-th deformation equation: `hbar * H * R_m`."""
        return self.hbar * self.H * self.r_m(phi, m)
