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

from ham.series import QSeries


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

    def apply_series(self, phi: QSeries) -> QSeries:
        """Substitute phi for u in `expr` and evaluate with QSeries arithmetic.

        Recursive tree-walker that compiles `expr` into QSeries operations
        against `phi`. After each multiplication the intermediate result is
        truncated back to `phi.order` (eager-truncation policy): HAM only
        ever reads coefficients up to q^{M-1} of N[phi] at working order M,
        so growing the Cauchy tail past phi.order is wasted work.

        Polynomial-in-u nodes (with optional x-only scalar factors) are
        compiled directly. Any node the walker does not recognise — most
        notably derivatives (3c) and transcendentals like sin/exp (3d) —
        raises NotImplementedError naming the offending subexpression.
        """
        return self._compile(self.expr, phi)

    def _compile(self, node: sp.Expr, phi: QSeries) -> QSeries:
        """Compile a sympy node into a QSeries against phi (recursive)."""
        if not node.has(self.dependent):
            return QSeries.constant(node, order=phi.order)
        if node == self.dependent(self.indep):
            return phi
        if isinstance(node, sp.Add):
            result = QSeries.zero(order=phi.order)
            for arg in node.args:
                result = result + self._compile(arg, phi)
            return result
        if isinstance(node, sp.Mul):
            result = QSeries.constant(sp.Integer(1), order=phi.order)
            for arg in node.args:
                factor = self._compile(arg, phi)
                result = (result * factor).trunc(phi.order)
            return result
        if isinstance(node, sp.Pow) and isinstance(node.exp, sp.Integer) and node.exp >= 0:
            base = self._compile(node.base, phi)
            result = QSeries.constant(sp.Integer(1), order=phi.order)
            for _ in range(int(node.exp)):
                result = (result * base).trunc(phi.order)
            return result
        raise NotImplementedError(
            f"NonlinearOperator.apply_series cannot compile subexpression "
            f"{node!r} (sympy type {type(node).__name__}). Polynomial-in-u "
            f"with no derivatives is supported in Stage 3b; derivatives "
            f"land in 3c; transcendentals (sin u, exp u, ...) in 3d."
        )
