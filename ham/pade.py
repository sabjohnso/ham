"""Homotopy-Padé acceleration for HAM partial sums (Stage 8).

The bare partial sum `u^{(M)}(x) = Σ_{k=0..M} u_k(x)` collapses the
homotopy series `phi(x; q) = Σ u_k(x) q^k` at `q = 1`. When the formal
series in q has radius of convergence less than 1 in q, that collapse
diverges even though phi is well-defined as a formal series.

The homotopy-Padé alternative is to form the [L/M] Padé approximant
`P(q) / Q(q)` of phi treated as a polynomial in q, then evaluate at
q = 1. The Padé approximant analytically continues the formal series
past its radius of convergence in q and often delivers a finite,
correct value where the bare partial sum cannot. For the canonical
geometric problem `u' = u², u(0) = 1` at `ℏ = -1`, the [0/1]
homotopy-Padé returns the exact closed-form `1/(1-x)` from just two
HAM coefficients.

Pure post-processing: depends only on `solution.phi.coeff(0..order)`.
No new algebra is introduced — the algorithm is the standard linear-
system Padé construction with `Q(0) = 1` normalisation.
"""

from functools import reduce
from operator import add

import sympy as sp

from ham.solver import HamSolution


def homotopy_pade(
    solution: HamSolution,
    numerator_degree: int,
    denominator_degree: int,
    hbar_value: sp.Expr | None = None,
) -> sp.Expr:
    """Compute the [L/M] homotopy-Padé approximant evaluated at q = 1.

    Treats `phi = Σ_{k=0..N} solution.phi.coeff(k) q^k` as a polynomial
    in q and constructs the rational approximant `P(q) / Q(q)` with
    `deg P = numerator_degree`, `deg Q = denominator_degree`, and
    `Q(0) = 1`. Returns `P(1) / Q(1)`, the homotopy-Padé acceleration
    of `solution.partial_sum()`.

    Validation:
      - both degrees must be non-negative;
      - `numerator_degree + denominator_degree <= solution.order`
        (the Padé construction needs at least L + M + 1 series
        coefficients to be well-defined).

    A singular denominator system raises sympy's
    `NonInvertibleMatrixError` rather than falling back silently — a
    degenerate Padé choice should fail loudly so the caller picks a
    different [L/M].

    With `hbar_value = None` the ℏ symbol from `solution.problem.hbar`
    is retained, mirroring the convention in `HamSolution.partial_sum`.
    """
    if numerator_degree < 0 or denominator_degree < 0:
        raise ValueError(
            f"homotopy_pade requires non-negative degrees; got "
            f"numerator_degree = {numerator_degree}, "
            f"denominator_degree = {denominator_degree}."
        )
    total_degree = numerator_degree + denominator_degree
    if total_degree > solution.order:
        raise ValueError(
            f"homotopy_pade requires numerator_degree + denominator_degree "
            f"<= solution.order; got {numerator_degree} + "
            f"{denominator_degree} = {total_degree} > {solution.order}."
        )

    phi = solution.phi
    series_coeffs = [phi.coeff(k) for k in range(solution.order + 1)]
    if hbar_value is not None:
        # Substitute ℏ /before/ building the Padé system so a degenerate
        # [L/M] choice at this specific ℏ surfaces as sympy's
        # NonInvertibleMatrixError rather than as a silent nan from
        # post-substituting a vanishing symbolic denominator.
        series_coeffs = [c.subs(solution.problem.hbar, hbar_value) for c in series_coeffs]

    if denominator_degree == 0:
        return reduce(
            add,
            series_coeffs[: numerator_degree + 1],
            sp.Integer(0),
        )
    return _pade_value_at_q_one(
        series_coeffs,
        numerator_degree,
        denominator_degree,
    )


def _pade_value_at_q_one(
    series_coeffs: list[sp.Expr],
    numerator_degree: int,
    denominator_degree: int,
) -> sp.Expr:
    """Solve the Padé linear system for q_1..q_M, build p_0..p_L, return P(1)/Q(1).

    `series_coeffs` are the input HAM q-coefficients `c_k = phi.coeff(k)`.
    The Padé denominator coefficients `q_j` solve the M-by-M system
        Σ_{j=1..M} c_{k-j} q_j = -c_k    for k = L+1..L+M,
    with q_0 = 1. The Padé numerator coefficients are then
        p_k = Σ_{j=0..min(k,M)} c_{k-j} q_j    for k = 0..L,
    and the value at q = 1 is `(p_0 + ... + p_L) / (1 + q_1 + ... + q_M)`.
    """
    max_index = len(series_coeffs) - 1
    size = denominator_degree
    matrix = sp.zeros(size, size)
    rhs = sp.zeros(size, 1)
    for row in range(size):
        for col in range(size):
            idx = numerator_degree + row - col
            if 0 <= idx <= max_index:
                matrix[row, col] = series_coeffs[idx]
        rhs[row, 0] = -series_coeffs[numerator_degree + 1 + row]
    denominator_corrections = matrix.LUsolve(rhs)
    q_coeffs: list[sp.Expr] = [sp.Integer(1)] + [denominator_corrections[j, 0] for j in range(size)]

    p_coeffs: list[sp.Expr] = []
    for k in range(numerator_degree + 1):
        partial: sp.Expr = sp.Integer(0)
        for j in range(min(k, denominator_degree) + 1):
            partial = partial + series_coeffs[k - j] * q_coeffs[j]
        p_coeffs.append(partial)

    numer = reduce(add, p_coeffs, sp.Integer(0))
    denom = reduce(add, q_coeffs, sp.Integer(0))
    return numer / denom
