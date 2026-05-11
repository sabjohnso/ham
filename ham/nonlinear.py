"""The nonlinear operator N for HAM.

The HAM zeroth-order deformation equation reads

    (1 - q) L[phi - u_0] = q * hbar * H(x) * N[phi]

so N is applied to the homotopy series phi(x; q) = sum_k u_k(x) q^k. In
this module, N is constructed from a sympy Expr in u(x), u'(x), u''(x), ...
and supports two evaluations:

  - `apply_scalar(u_concrete)`: substitute a concrete function for u and
    evaluate any derivatives that appear, returning a sympy Expr in x.
  - `apply_series(phi)`: substitute a QSeries for u, evaluating the
    expression with QSeries arithmetic. Provided in 3b/3c.

Polynomial expressions in u, u', ... are compiled to QSeries arithmetic;
transcendental dependencies (sin u, exp u, ...) raise NotImplementedError
at the offending node when 3d lands.
"""

from dataclasses import dataclass

import sympy as sp


@dataclass(frozen=True)
class NonlinearOperator:
    """The nonlinear operator N as a sympy expression in u(x), u'(x), ...

    Fields:
      - `expr`:      the sympy expression defining N[u].
      - `dependent`: the Function symbol standing for u (e.g. `sp.Function('u')`).
      - `indep`:     the independent variable (e.g. `sp.Symbol('x')`).
    """

    expr: sp.Expr
    dependent: sp.Function
    indep: sp.Symbol

    def apply_scalar(self, u_concrete: sp.Expr) -> sp.Expr:
        """Substitute u(indep) -> u_concrete and evaluate derivatives.

        Uses `sp.Lambda(indep, u_concrete)` as the substitution target so
        sympy propagates the substitution through any `Derivative` nodes
        that appear in `expr`, then `.doit()` evaluates them.
        """
        substituted = self.expr.subs(self.dependent, sp.Lambda(self.indep, u_concrete))
        return substituted.doit()
