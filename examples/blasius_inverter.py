"""Closed-form basis-aware inverter for the Blasius exponential-basis L.

The Stage 11 `_blasius_exponential_inverter` uses `sympy.dsolve` on the
full HAM RHS at every deformation step. With alpha kept symbolic (Stage 12),
that dsolve call grows substantially: M = 2 takes ≈ 3 s, M = 3 ≈ 12 s,
M = 4 minutes.

This module replaces the dsolve-based inverter with a closed-form one
that:

1. **Decomposes** the RHS into the natural basis for the Blasius
   exponential setup:

        rhs = Σ c_i · η^(j_i) · exp(-k_i · alpha · η)

   with `c_i` η-free (it may still depend on alpha, ℏ, and any other
   problem-level symbols). Terms outside this basis raise
   `NotImplementedError`.

2. **Caches** the L^{-1} of each basis element `η^j · exp(-k alpha η)`
   under the three BCs `u(0) = u'(0) = u'(∞) = 0`. The first invocation
   for each `(j, k)` pair calls `sympy.dsolve` on a trivial single-term
   RHS (fast, even with alpha symbolic); subsequent invocations hit the
   cache. The Stage 11 "zero out free constants" trick still applies
   to silence the asymptotic-BC ambiguity for the resonant `k = 1`
   case.

3. **Assembles** the full inverse as
   `Σ c_i · _basis_l_inverse(j_i, k_i)` — by construction every basis
   term already satisfies the three BCs, so the sum does too, with
   no per-step BC correction needed.

The cache key is `(j, k)`; alpha appears in the cached expression as a
sympy symbol. Across a typical HAM run the cache fills quickly and
later orders amortise to seconds. The Stage 13 regression tests pin
this against the Stage 11 dsolve-based inverter on identical RHSes.
"""

from functools import cache
from typing import Protocol

import sympy as sp


class BlasiusInverter(Protocol):
    """Callable for `LinearOperator.inverter` plus the two introspection helpers.

    `__call__` is the inverter entry point used by HAM. `basis_l_inverse`
    and `decompose` are exposed so regression tests can pin the closed-
    form per-basis-element solutions and the RHS decomposition in
    isolation.
    """

    def __call__(self, rhs: sp.Expr) -> sp.Expr: ...
    def basis_l_inverse(self, j: int, k: int) -> sp.Expr: ...
    def decompose(self, rhs: sp.Expr) -> list[tuple[sp.Expr, int, int]]: ...


def make_blasius_inverter(
    eta: sp.Symbol,
    alpha: sp.Symbol,
) -> BlasiusInverter:
    """Return a basis-aware inverter for `L = d^3/dη^3 - alpha^2 · d/dη`.

    The returned callable conforms to `ham.operator.LinearOperator.inverter`'s
    `Callable[[Expr], Expr]` shape. It expects RHSes lying in the
    Blasius exponential basis (linear combinations of
    `η^j · exp(-kalphaη)`); anything else raises `NotImplementedError`.

    The inverter satisfies `f(0) = 0`, `f'(0) = 0`, `f'(∞) = 0` on its
    output (the homogeneous-deformation form of the Blasius BCs);
    Stage 11's dsolve-zero-free-constants workaround is built into
    each cached basis element.
    """

    @cache
    def basis_l_inverse(j: int, k: int) -> sp.Expr:
        """L^{-1}(η^j · exp(-kalphaη)) under the Blasius homogeneous BCs.

        Cached per (j, k); the result is symbolic in alpha. Uses the same
        Stage 11 dsolve + free-constant-zeroing trick that worked for
        the assembled RHS, but the single-term RHS keeps each dsolve
        call cheap.

        Requires `k >= 1`: a non-exponentially-decaying RHS like η^j
        admits no L^{-1} satisfying f'(∞) = 0 under this L, because
        the particular solution itself contains a non-decaying η-power
        term that no element of the kernel {1, e^(-alphaη)} can cancel.
        Calling with k = 0 raises NotImplementedError.
        """
        if k < 1:
            raise NotImplementedError(
                f"basis_l_inverse requires k >= 1; got k = {k}. The Blasius "
                f"homogeneous BCs (f(0)=f'(0)=f'(infty)=0) admit no finite "
                f"L^(-1) for a non-exponentially-decaying RHS (k = 0)."
            )
        rhs_term = eta**j * sp.exp(-k * alpha * eta)
        u_func = sp.Function("_u_basis")
        u = u_func(eta)
        ode = sp.Eq(u.diff(eta, 3) - alpha**2 * u.diff(eta), rhs_term)
        ics = {
            u.subs(eta, 0): 0,
            u.diff(eta).subs(eta, 0): 0,
        }
        sol = sp.dsolve(ode, u, ics=ics)
        result = sol.rhs
        free = result.free_symbols - {eta, alpha}
        if free:
            result = result.subs(dict.fromkeys(free, sp.Integer(0)))
        return sp.simplify(result)

    def decompose(rhs: sp.Expr) -> list[tuple[sp.Expr, int, int]]:
        """Split rhs into a list of (coeff, j, k) tuples.

        Each tuple represents `coeff · η^j · exp(-kalphaη)` where `coeff` is
        η-free (it may still contain alpha, ℏ, or other free symbols). Zero
        terms are skipped; the empty rhs decomposes to an empty list.
        """
        rhs_expanded = sp.expand(rhs)
        decomposed: list[tuple[sp.Expr, int, int]] = []
        for term in sp.Add.make_args(rhs_expanded):
            if term == sp.Integer(0):
                continue
            coeff, eta_part = term.as_independent(eta)
            j = 0
            k = 0
            if eta_part == sp.Integer(1):
                # η-free term (constant in η, possibly with alpha/ℏ in coeff).
                decomposed.append((coeff, 0, 0))
                continue
            for factor in sp.Mul.make_args(eta_part):
                if factor == eta:
                    j += 1
                elif factor.is_Pow and factor.base == eta and isinstance(factor.exp, sp.Integer):
                    if factor.exp < 0:
                        raise NotImplementedError(
                            f"Blasius inverter cannot decompose η^n with n < 0; got {term!r}."
                        )
                    j += int(factor.exp)
                elif factor.func == sp.exp:
                    arg = factor.args[0]
                    k_candidate = sp.simplify(-arg / (alpha * eta))
                    if not isinstance(k_candidate, sp.Integer) or k_candidate < 0:
                        raise NotImplementedError(
                            f"Blasius inverter cannot decompose exp argument {arg!r} "
                            f"in {term!r}; expected -k*alpha*η with k a non-negative integer."
                        )
                    k += int(k_candidate)
                else:
                    raise NotImplementedError(
                        f"Blasius inverter cannot decompose factor {factor!r} "
                        f"of term {term!r}; expected η, η^n, or exp(-k alpha η)."
                    )
            decomposed.append((coeff, j, k))
        return decomposed

    def invert(rhs: sp.Expr) -> sp.Expr:
        """Apply L^{-1} to rhs by decomposition + cached basis lookup."""
        total: sp.Expr = sp.Integer(0)
        for coeff, j, k in decompose(rhs):
            total = total + coeff * basis_l_inverse(j, k)
        return sp.expand(total)

    # Expose the helpers on the closure so tests can introspect them.
    invert.basis_l_inverse = basis_l_inverse  # type: ignore[attr-defined]
    invert.decompose = decompose  # type: ignore[attr-defined]
    return invert  # type: ignore[return-value]
