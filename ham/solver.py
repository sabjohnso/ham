"""The HAM solver loop (Stage 5).

Composes Stages 1-4 into a forward sweep:

  for m in 1..M:
      R_m   = [q^{m-1}] N[phi_{m-1}]              (Stage 3 via Stage 4)
      rhs   = hbar * H(x) * R_m                   (Stage 4)
      v     = L.invert(rhs)                       (Stage 2)
      u_m   = v + chi_m(m) * u_{m-1}              (this stage)

The series coefficients u_0, ..., u_M are bundled into a HamSolution
that retains the originating problem and the homotopy QSeries. The
convergence-control parameter ℏ is carried through symbolically, so
downstream consumers (Stage 6) can re-evaluate the partial sum at
different ℏ values without re-running the loop.

No claim is made here about convergence at q = 1. Stage 6 owns that.
"""

from dataclasses import dataclass
from functools import reduce
from operator import add

import sympy as sp

from ham.backend import SympyBackend
from ham.deformation import HamProblem, chi_m
from ham.series import QSeries


@dataclass(frozen=True)
class HamSolution:
    """The output of an HAM solve to working order M.

    Fields:
      problem: the originating HamProblem (so ℏ, H, L, etc. stay reachable).
      phi:     the homotopy QSeries, with coeff(k) = u_k for k = 0..M.

    Use `partial_sum()` for the formal Σ_k u_k(x) with ℏ symbolic, and
    `evaluate_at_hbar(value)` for the same sum with ℏ substituted.
    """

    problem: HamProblem
    phi: QSeries

    @property
    def order(self) -> int:
        """The working order M = phi.order."""
        return self.phi.order

    def partial_sum(self) -> sp.Expr:
        """Formal partial sum u^{(M)}(x) = Σ_{k=0..M} u_k(x); ℏ kept symbolic."""
        terms = [self.phi.coeff(k) for k in range(self.order + 1)]
        return sp.expand(reduce(add, terms, sp.Integer(0)))

    def evaluate_at_hbar(self, value: sp.Expr) -> sp.Expr:
        """Partial sum with ℏ substituted to `value`."""
        return sp.expand(self.partial_sum().subs(self.problem.hbar, value))


def solve_step(problem: HamProblem, phi: QSeries, m: int) -> sp.Expr:
    """Compute u_m from phi = u_0 + u_1 q + ... + u_{m-1} q^{m-1}.

    Builds the m-th deformation RHS via `problem.rhs_m`, inverts L,
    then adds back the χ_m u_{m-1} correction. The result is canonical-
    ised via `phi.backend.normalize` (=`sp.expand` for sympy, identity
    for the future spectral backend) so downstream tests can compare
    with structural `==` without re-canonicalising.
    """
    rhs = problem.rhs_m(phi, m)
    v = problem.L.invert(rhs)
    return phi.backend.normalize(v + chi_m(m) * phi.coeff(m - 1))


def solve(problem: HamProblem, order: int) -> HamSolution:
    """Run the HAM solver to working order M, returning a HamSolution.

    Builds phi incrementally: phi_0 = constant(u_0), and at each step
    m = 1..order, extends with u_m = solve_step(problem, phi_{m-1}, m).
    The ℏ symbol in `problem.hbar` is carried through every coefficient
    so the result can be evaluated at any ℏ without re-running the loop.

    The constructed `phi` carries a `SympyBackend(problem.L.var)` so
    that `N.apply_series`'s coefficient-wise diff_x / integrate_x
    operate w.r.t. the problem's independent variable rather than the
    `QSeries` default (which assumes `sp.Symbol("x")`). Examples like
    Volterra (`t`) and Blasius (`eta`) depend on this.
    """
    if order < 0:
        raise ValueError(f"solve requires order >= 0; got order = {order}.")
    backend = SympyBackend(problem.L.var)
    coeffs: list[sp.Expr] = [backend.normalize(backend.lift_xonly(problem.u0))]
    for m in range(1, order + 1):
        phi = QSeries(coeffs, order=m - 1, backend=backend)
        coeffs.append(solve_step(problem, phi, m))
    return HamSolution(problem=problem, phi=QSeries(coeffs, order=order, backend=backend))
