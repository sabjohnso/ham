"""The auxiliary linear operator L for HAM.

The HAM m-th order deformation equation has the shape

    L[u_m - chi_m * u_{m-1}] = hbar * H(x) * R_m(u_0, ..., u_{m-1})

so L participates in HAM at two points: forwards, where L is applied to a
known coefficient, and backwards, where L is inverted against a known
right-hand side to obtain u_m. Both directions live in this module.

`LinearOperator[C]` is generic over the coefficient type C: `sp.Expr`
for symbolic HAM, `numpy.ndarray` for SHAM (S5b+). It bundles:

  - `var`:     the independent variable as a sympy.Symbol (kept symbolic
               for BC interpretation and for legacy diagnostics; the
               substrate's actual computation lives in `action` and
               `inverter`).
  - `action`:  a `Callable[[C], C]` computing L[u] forward.
  - `bcs`:     a tuple of `BoundaryCondition` values that L^{-1} must
               respect. In HAM the deformation BCs are homogeneous, so
               the default `value` is 0.
  - `inverter` (optional): a `Callable[[C], C]` that solves L[u] = rhs
               under the declared BCs. When None, `L.invert` falls back
               to the sympy.dsolve-backed inverter — meaningful only
               when C = sp.Expr; non-sympy callers must supply an
               explicit inverter (S6 ships `spectral_inverter` in the
               same factory shape).

`sympy_dsolve_inverter(var, action, bcs)` exposes the dsolve inverter
as a public factory so callers can surface the inverter explicitly
(for testing / profiling / composition) instead of relying on the
implicit fallback. This is the symbolic-side counterpart to S6's
`spectral_inverter` factory.

Linearity of `action` and consistency of `inverter` with `bcs` are the
caller's contract; the wrapper does not enforce them. Property tests
verify both for known operators. Users who want a runtime check on
their own `action` can call `ham.contracts.verify_linearity(L, samples)`
at problem-construction time; it raises `LinearityViolation` with the
offending sample if the law fails. The comparator used to decide
"violated" is injectable via the `equal` kwarg (PLAN.org D-4) so the
same checker serves sympy and spectral backends from one site.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import sympy as sp

from ham.series import Series, SupportsCoefficientArith


@dataclass(frozen=True)
class BoundaryCondition:
    """A single boundary/initial condition: u^(k)(point) = value."""

    point: sp.Expr
    derivative_order: int
    value: sp.Expr = sp.S.Zero


@dataclass(frozen=True)
class LinearOperator[C: SupportsCoefficientArith]:
    """A linear operator on coefficient values, with forward and inverse paths.

    See module docstring for the contract on `action`, `bcs`, and
    `inverter`. The class is generic over C so the same scaffolding
    serves sympy (C = sp.Expr) and SHAM (C = np.ndarray) with no code
    duplication; substrate-specific behaviour lives entirely in the
    Callables passed in.
    """

    var: sp.Symbol
    action: Callable[[C], C]
    bcs: tuple[BoundaryCondition, ...] = ()
    inverter: Callable[[C], C] | None = None

    def apply(self, u: C) -> C:
        """Apply L forward to a coefficient value."""
        return self.action(u)

    def apply_series(self, s: Series[C]) -> Series[C]:
        """Apply L coefficient-wise to a Series (the q/x gluing law)."""
        return s.map_coeffs(self.action)

    def invert(self, rhs: C) -> C:
        """Solve L[u] = rhs subject to the declared BCs, returning u.

        Delegates to `self.inverter` when present, otherwise falls back
        to the sympy.dsolve-backed inverter. The fallback is meaningful
        only when C = sp.Expr; non-sympy callers must supply an
        explicit `inverter` (typically via the substrate's inverter
        factory — `sympy_dsolve_inverter` here, `spectral_inverter`
        in S6).
        """
        if self.inverter is not None:
            return self.inverter(rhs)
        # Sympy-specific fallback. The cast reflects the contract: the
        # default path only makes sense when C = sp.Expr, in which case
        # the dsolve callable returns sp.Expr, which is C. Non-sympy
        # callers are required (per module docstring) to supply an
        # explicit inverter and never reach this branch.
        return cast("C", sympy_dsolve_inverter(self.var, self.action, self.bcs)(rhs))


def sympy_dsolve_inverter(
    var: sp.Symbol,
    action: Callable[[sp.Expr], sp.Expr],
    bcs: tuple[BoundaryCondition, ...],
) -> Callable[[sp.Expr], sp.Expr]:
    """Factory: build the sympy.dsolve-backed inverter from (var, action, bcs).

    Returns a callable `rhs -> u` solving `action(u) = rhs` under the
    declared BCs via `sympy.dsolve`. Equivalent to the implicit
    fallback used by `LinearOperator.invert` when `inverter=None`; the
    explicit factory exists so callers can surface the inverter (for
    testing, profiling, or composition) and so SHAM's `spectral_inverter`
    in S6 has a matching factory shape — "build an inverter callable
    from var, action, bcs" is the substrate-agnostic interface, the
    body is what varies.
    """

    def _invert(rhs: sp.Expr) -> sp.Expr:
        u = sp.Function("u")
        u_of_var = u(var)
        equation = sp.Eq(action(u_of_var), rhs)
        ics = {_bc_to_ic(u_of_var, var, bc): bc.value for bc in bcs}
        solution = sp.dsolve(equation, u_of_var, ics=ics)
        return solution.rhs

    return _invert


def _bc_to_ic(u_of_var: sp.Expr, var: sp.Symbol, bc: BoundaryCondition) -> sp.Expr:
    """Translate a BoundaryCondition into the LHS form sympy.dsolve expects.

    For derivative_order = k, returns `u(var).diff(var, k).subs(var, bc.point)`,
    which sympy.dsolve recognizes as "u^(k)(point)" in its ics dictionary.
    """
    expr = u_of_var
    for _ in range(bc.derivative_order):
        expr = sp.diff(expr, var)
    return expr.subs(var, bc.point)


def antiderivative(var: sp.Symbol, t0: sp.Expr = sp.S.Zero) -> Callable[[sp.Expr], sp.Expr]:
    """Canonical-case inverter for L = d/d{var} with u({var}={t0}) = 0.

    Returns the closed-form definite-integration inverter
    `rhs |-> integral_{t0}^{var} rhs(s) ds`, which automatically satisfies
    the point condition u(t0) = 0 by construction.
    """

    def _invert(rhs: sp.Expr) -> sp.Expr:
        return sp.integrate(rhs, (var, t0, var))

    return _invert
