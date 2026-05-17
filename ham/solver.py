"""The HAM solver loop (Stage 5, generalised through S7).

Composes Stages 1-4 into a forward sweep:

  for m in 1..M:
      R_m   = [q^{m-1}] N[phi_{m-1}]              (Stage 3 via Stage 4)
      rhs   = hbar * H(x) * R_m                   (Stage 4)
      v     = L.invert(rhs)                       (Stage 2)
      u_m   = v + chi_m(m) * u_{m-1}              (this stage)

The series coefficients u_0, ..., u_M are bundled into a `HamSolution[C]`
that retains the originating problem and the homotopy `Series[C]`. The
loop is substrate-agnostic: pass `backend=SympyBackend(...)` for symbolic
HAM, `backend=SpectralBackend(...)` for SHAM. If `backend` is omitted
the solver constructs a `SympyBackend(problem.L.var)` for back-compat
with pre-S7 callers.

The convergence-control parameter ℏ is carried through every coefficient.
For the sympy / sympy-scalar paths, ℏ stays symbolic and downstream
consumers can re-evaluate the partial sum at different ℏ values without
re-running the loop. For the spectral float path, ℏ is pre-substituted
on `problem.hbar` (typically `sp.Float(value)`) and the partial sum is
already at that ℏ.

No claim is made here about convergence at q = 1. Stage 6 / 8 own that.
"""

from dataclasses import dataclass
from functools import reduce
from operator import add
from typing import cast

import sympy as sp

from ham.backend import Backend, SympyBackend
from ham.deformation import HamProblem, chi_m
from ham.series import Series, SupportsCoefficientArith


@dataclass(frozen=True)
class HamSolution[C: SupportsCoefficientArith]:
    """The output of an HAM solve to working order M, generic over substrate C.

    Fields:
      problem: the originating HamProblem (so ℏ, H, L, etc. stay reachable).
      phi:     the homotopy Series, with coeff(k) = u_k for k = 0..M.

    Use `partial_sum()` for the formal Σ_k u_k; for the sympy substrate
    this is a sympy.Expr in x and possibly ℏ, and `evaluate_at_hbar` is
    the sp.subs-based shortcut. For the spectral substrate it's an
    `np.ndarray` of grid values (with each entry a sympy expression in
    ℏ when `scalar="sympy"`, or a numeric float when `scalar="float"`).
    """

    problem: HamProblem[C]
    phi: Series[C]

    @property
    def order(self) -> int:
        """The working order M = phi.order."""
        return self.phi.order

    def partial_sum(self) -> C:
        """Formal partial sum u^{(M)} = Σ_{k=0..M} u_k.

        Returns a single value in the coefficient substrate: a
        `sympy.Expr` for the sympy backend, an `np.ndarray` of grid
        values for the spectral backend. The result is canonicalised
        via `phi.backend.normalize` (= `sp.expand` for sympy, identity
        for the float spectral backend, element-wise `sp.expand` for
        the sympy-scalar spectral backend).
        """
        backend = self.phi.backend
        terms = [self.phi.coeff(k) for k in range(self.order + 1)]
        return backend.normalize(reduce(add, terms, backend.zero()))

    def evaluate_at_hbar(self, value: sp.Expr) -> sp.Expr:
        """Partial sum with ℏ substituted to `value` — sympy-substrate only.

        Calls `.subs(problem.hbar, value)` on the partial sum. Works
        when `C = sp.Expr` (the sympy backend); raises `AttributeError`
        on the float spectral path (numpy arrays have no `.subs`) and
        is unnecessary on the sympy-scalar spectral path (which carries
        ℏ inside every grid entry — callers there should substitute
        element-wise or use `partial_sum()` directly).
        """
        return sp.expand(self.partial_sum().subs(self.problem.hbar, value))  # type: ignore[attr-defined]


def solve_step[C: SupportsCoefficientArith](problem: HamProblem[C], phi: Series[C], m: int) -> C:
    """Compute u_m from phi = u_0 + u_1 q + ... + u_{m-1} q^{m-1}.

    Builds the m-th deformation RHS via `problem.rhs_m`, inverts L,
    then adds back the χ_m u_{m-1} correction. The result is canonical-
    ised via `phi.backend.normalize` (=`sp.expand` for sympy, identity
    for the spectral float backend) so downstream tests can compare
    with structural `==` without re-canonicalising.
    """
    rhs = problem.rhs_m(phi, m)
    v = problem.L.invert(rhs)
    return phi.backend.normalize(v + chi_m(m) * phi.coeff(m - 1))


def solve[C: SupportsCoefficientArith](
    problem: HamProblem[C],
    order: int,
    *,
    backend: Backend[C] | None = None,
) -> HamSolution[C]:
    """Run the HAM solver to working order M, returning a `HamSolution[C]`.

    Builds phi incrementally: phi_0 = constant(u_0 lifted to C), and at
    each step m = 1..order, extends with u_m = solve_step(problem,
    phi_{m-1}, m). The `backend` parameter selects the substrate; when
    omitted, the solver constructs `SympyBackend(problem.L.var)` so
    legacy sympy-only callers continue to work unchanged.

    For the spectral path, pass `backend=SpectralBackend(grid, indep,
    scalar)`; `problem.L` must then be a `LinearOperator[np.ndarray]`
    built via `spectral_linear_operator(...)` over the same grid.
    """
    if order < 0:
        raise ValueError(f"solve requires order >= 0; got order = {order}.")
    if backend is None:
        # Sympy default for back-compat. SympyBackend keyed on
        # `problem.L.var` so the calculus inside `N.apply_series`
        # operates w.r.t. the problem's indep, not QSeries's "x".
        backend = cast("Backend[C]", SympyBackend(problem.L.var))
    coeffs: list[C] = [backend.normalize(backend.lift_xonly(problem.u0))]
    for m in range(1, order + 1):
        phi: Series[C] = Series(coeffs, order=m - 1, backend=backend)
        coeffs.append(solve_step(problem, phi, m))
    final_phi: Series[C] = Series(coeffs, order=order, backend=backend)
    return HamSolution(problem=problem, phi=final_phi)
