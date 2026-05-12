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
        """Substitute u(indep) -> u_concrete and evaluate derivatives / integrals.

        Uses `expr.replace(dependent, Lambda(indep, u_concrete))` rather than
        `.subs`. The two produce identical results for Derivative subnodes,
        but `.subs` fails on `Integral` subnodes with the obscure sympy
        error "TypeError: 'property' object is not iterable" — `.replace`
        walks the tree more carefully and handles the Integral case
        correctly. `.doit()` then evaluates any Derivative / Integral
        subnodes that emerged from the substitution.
        """
        substituted = self.expr.replace(self.dependent, sp.Lambda(self.indep, u_concrete))
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
            # `.doit()` evaluates any symbolic Integral / Derivative subnodes
            # in the u-free path so they reach the QSeries as concrete values
            # rather than as unevaluated sympy objects in coefficient slot 0.
            return QSeries.constant(node.doit(), order=phi.order)
        if node == self.dependent(self.indep):
            return phi
        if isinstance(node, sp.Derivative):
            return self._compile_derivative(node, phi)
        if isinstance(node, sp.Integral):
            return self._compile_integral(node, phi)
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
            f"{node!r} (sympy type {type(node).__name__}). The compiler "
            f"handles polynomial-in-u expressions (and integer-power "
            f"composites) with x-derivatives of u; other dependencies on "
            f"u — transcendentals (sin u, exp u, log u, ...), rational "
            f"powers (sqrt u), reciprocals (1/u) — need formal-series "
            f"composition and are not yet implemented. Workaround: "
            f"replace the offending term with a truncated Taylor series "
            f"in u before constructing the NonlinearOperator."
        )

    def _compile_derivative(self, node: sp.Derivative, phi: QSeries) -> QSeries:
        """Compile a Derivative(u(indep), indep, k) node into coefficient-wise diff.

        Only handles derivatives applied directly to `dependent(indep)` and
        taken in `indep` alone. A derivative whose inner expression is any
        product, sum, or other composition involving u (e.g.
        `Derivative(u(x)*x, x)`) raises NotImplementedError — the compiler
        is not a product-rule engine. Users should expand such derivatives
        symbolically before constructing the NonlinearOperator.
        """
        bare_u = self.dependent(self.indep)
        only_indep = len(node.variable_count) == 1 and node.variable_count[0][0] == self.indep
        if node.expr != bare_u or not only_indep:
            raise NotImplementedError(
                f"NonlinearOperator.apply_series only handles derivatives of "
                f"the form Derivative({self.dependent}({self.indep}), "
                f"{self.indep}, k); got {node!r}. Expand the derivative "
                f"symbolically before constructing the NonlinearOperator."
            )
        k = int(node.variable_count[0][1])
        return phi.map_coeffs(lambda c: sp.diff(c, self.indep, k))

    def _compile_integral(self, node: sp.Integral, phi: QSeries) -> QSeries:
        """Compile an `Integral(integrand, (dummy, 0, indep))` node into
        coefficient-wise integration on the integrand's compiled QSeries.

        Supports the canonical Volterra form: a single-variable definite
        integral from 0 to `indep` whose integrand depends on the dummy
        only through `dependent(dummy)` (and its derivatives in the dummy).
        Other shapes — multi-variable integrals, lower bounds other than 0,
        upper bounds other than `indep`, or integrands with explicit
        non-`dependent` dummy dependence — raise NotImplementedError.

        Algorithm: substitute `dummy -> indep` in the integrand (this is
        sound when the dummy appears only inside `dependent(dummy)` calls),
        recursively compile the resulting expression to obtain the
        QSeries `f(phi)`, then `f(phi).map_coeffs(integrate from 0 to indep)`.
        """
        if len(node.limits) != 1:
            raise NotImplementedError(
                f"NonlinearOperator.apply_series only handles single-variable "
                f"integrals; got {node!r} with {len(node.limits)} integration "
                f"variables."
            )
        limit = node.limits[0]
        if len(limit) != 3:
            raise NotImplementedError(
                f"NonlinearOperator.apply_series requires definite integrals "
                f"with both lower and upper bounds; got {node!r}."
            )
        dummy, lower, upper = limit
        if lower != sp.Integer(0):
            raise NotImplementedError(
                f"NonlinearOperator.apply_series only handles integrals with "
                f"lower bound 0; got {node!r} with lower bound {lower}."
            )
        if upper != self.indep:
            raise NotImplementedError(
                f"NonlinearOperator.apply_series only handles integrals with "
                f"upper bound = {self.indep!r} (the independent variable); "
                f"got {node!r} with upper bound {upper}."
            )
        integrand_at_indep = node.function.subs(dummy, self.indep)
        integrand_qseries = self._compile(integrand_at_indep, phi)
        return integrand_qseries.map_coeffs(
            lambda c: sp.integrate(c, (self.indep, sp.Integer(0), self.indep))
        )
