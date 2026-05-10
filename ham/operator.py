"""The auxiliary linear operator L for HAM.

Stage 2a ships only the forward action: a thin wrapper around a
Callable[[Expr], Expr] that lifts coefficient-wise to a QSeries via the
functor map. Boundary-condition data and the inversion strategy are added
in 2b and 2c (see PLAN.org).

Linearity is not enforced by the wrapper — the caller supplies a Callable
they claim is linear. Property tests in tests/test_operator.py verify the
wrapper preserves linearity for known-linear actions; downstream code that
relies on linearity should always be written against the law, not the
wrapper's runtime check (there is none).
"""

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from ham.series import QSeries


@dataclass(frozen=True)
class LinearOperator:
    """A linear operator on sympy expressions.

    Forward direction only at this stage; see module docstring.
    """

    action: Callable[[sp.Expr], sp.Expr]

    def apply(self, u: sp.Expr) -> sp.Expr:
        """Apply the operator to a sympy expression in the independent variable."""
        return self.action(u)

    def apply_series(self, s: QSeries) -> QSeries:
        """Apply the operator coefficient-wise to a QSeries (the q/x gluing law)."""
        return s.map_coeffs(self.action)
