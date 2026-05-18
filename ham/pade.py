"""Homotopy-Padé acceleration for HAM partial sums (Stage 8 + S9 spectral).

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

Sympy substrate: the Padé denominator coefficients `q_1..q_M` form
a single (MxM) linear system over sympy expressions, solved via
`sp.Matrix.LUsolve`.

Spectral substrate (S9 follow-up): each `c_k` is a grid vector, and
the construction is **block-structured** — the Padé denominator
coefficients become per-grid-node vectors `q_j[i]`, and the linear
system decomposes into N+1 independent MxM solves (one per grid
node). For the float scalar these are batched via `np.linalg.solve`;
for the sympy scalar they are looped via `sp.Matrix.LUsolve`. The
result is a grid vector of `P(1)/Q(1)` values, one per node.
"""

from collections.abc import Sequence
from functools import reduce
from operator import add
from typing import Any, cast

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ham.series import SupportsCoefficientArith
from ham.solver import HamSolution


def homotopy_pade[C: SupportsCoefficientArith](
    solution: HamSolution[C],
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

    Sympy substrate: returns a sympy expression (rational function
    in `solution.problem.L.var`, with ℏ retained symbolically unless
    `hbar_value` is supplied).

    Spectral substrate: returns a grid vector (`np.ndarray` of
    length N+1) of `P(1)/Q(1)` values, computed by the
    block-structured construction — one MxM solve per grid node,
    vectorised via `np.linalg.solve` on the float scalar and looped
    via `sp.Matrix.LUsolve` on the sympy scalar. `hbar_value` is
    rejected on the float scalar (ℏ is pre-substituted at problem
    construction; no symbols remain to substitute), and substituted
    element-wise before the Padé build on the sympy scalar.

    A singular denominator system raises sympy's
    `NonInvertibleMatrixError` (sympy substrate) or
    `numpy.linalg.LinAlgError` (spectral float). On the spectral
    sympy scalar each grid node's solve uses sympy, so the failure
    mode there is also `NonInvertibleMatrixError`.

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

    if isinstance(solution.phi.backend.zero(), np.ndarray):
        return cast(
            "sp.Expr",
            _spectral_homotopy_pade(
                cast("HamSolution[NDArray[Any]]", solution),
                numerator_degree,
                denominator_degree,
                hbar_value,
            ),
        )

    sympy_solution = cast("HamSolution[sp.Expr]", solution)
    phi = sympy_solution.phi
    series_coeffs = [phi.coeff(k) for k in range(sympy_solution.order + 1)]
    if hbar_value is not None:
        # Substitute ℏ /before/ building the Padé system so a degenerate
        # [L/M] choice at this specific ℏ surfaces as sympy's
        # NonInvertibleMatrixError rather than as a silent nan from
        # post-substituting a vanishing symbolic denominator.
        series_coeffs = [c.subs(sympy_solution.problem.hbar, hbar_value) for c in series_coeffs]

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


# --- Spectral path -------------------------------------------------------


def _spectral_homotopy_pade(
    solution: HamSolution[NDArray[Any]],
    numerator_degree: int,
    denominator_degree: int,
    hbar_value: sp.Expr | None,
) -> NDArray[Any]:
    """Block-structured Padé over a spectral solution's grid-vector coefficients.

    Detects float vs sympy scalar via the dtype of the first
    coefficient. The float path uses `np.linalg.solve` on a batched
    `(N+1, M, M)` matrix stack; the sympy path loops sympy `LUsolve`
    over grid nodes.
    """
    phi = solution.phi
    sample_coeff = phi.coeff(0)
    is_float = sample_coeff.dtype != object

    if is_float and hbar_value is not None:
        raise ValueError(
            "homotopy_pade with hbar_value on a float-scalar spectral "
            "solution: ℏ is pre-substituted at HamProblem construction; no "
            "symbols remain to substitute. Either omit hbar_value, or use "
            "scalar='sympy' on the SpectralBackend to keep ℏ symbolic in "
            "the partial sum."
        )

    series_coeffs = [phi.coeff(k) for k in range(solution.order + 1)]
    if hbar_value is not None:
        # Sympy scalar path with ℏ supplied: substitute element-wise.
        series_coeffs = [
            np.array(
                [entry.subs(solution.problem.hbar, hbar_value) for entry in c],
                dtype=object,
            )
            for c in series_coeffs
        ]

    if denominator_degree == 0:
        # P(q) = Σ c_k, Q(q) = 1.
        truncated = series_coeffs[0].copy()
        for k in range(1, numerator_degree + 1):
            truncated = truncated + series_coeffs[k]
        return truncated

    if is_float:
        return _spectral_pade_value_at_q_one_float(
            series_coeffs, numerator_degree, denominator_degree
        )
    return _spectral_pade_value_at_q_one_sympy(series_coeffs, numerator_degree, denominator_degree)


def _spectral_pade_value_at_q_one_float(
    series_coeffs: Sequence[NDArray[np.float64]],
    numerator_degree: int,
    denominator_degree: int,
) -> NDArray[np.float64]:
    """Batched Padé construction for float-scalar spectral coefficients.

    Builds the `(N+1, M, M)` matrix stack `A_batch[i, row, col] =
    c_{L+row-col}[i]` and `(N+1, M)` rhs `b_batch[i, row] =
    -c_{L+1+row}[i]`, solves all N+1 systems in one
    `np.linalg.solve` call, then assembles P(1)/Q(1) element-wise
    on the grid. A degenerate Padé at any grid node propagates the
    singular-matrix error from numpy rather than fall back silently.
    """
    max_index = len(series_coeffs) - 1
    size = denominator_degree
    n_plus_1 = series_coeffs[0].shape[0]

    a_batch = np.zeros((n_plus_1, size, size), dtype=np.float64)
    rhs_batch = np.zeros((n_plus_1, size, 1), dtype=np.float64)
    for row in range(size):
        for col in range(size):
            idx = numerator_degree + row - col
            if 0 <= idx <= max_index:
                a_batch[:, row, col] = series_coeffs[idx]
        rhs_batch[:, row, 0] = -series_coeffs[numerator_degree + 1 + row]

    # np.linalg.solve broadcasts over leading batch dims of A and b;
    # b must be at least 2-D (`(..., M, K)`) so we keep a singleton
    # last axis for the per-node solve and squeeze it back afterwards.
    denominator_corrections = np.linalg.solve(a_batch, rhs_batch)[:, :, 0]

    # q_coeffs[i, 0] = 1 and q_coeffs[i, 1:] = solved corrections.
    q_coeffs = np.concatenate(
        [np.ones((n_plus_1, 1), dtype=np.float64), denominator_corrections],
        axis=1,
    )

    p_coeffs = np.zeros((n_plus_1, numerator_degree + 1), dtype=np.float64)
    for k in range(numerator_degree + 1):
        for j in range(min(k, denominator_degree) + 1):
            p_coeffs[:, k] += series_coeffs[k - j] * q_coeffs[:, j]

    numer = p_coeffs.sum(axis=1)
    denom = q_coeffs.sum(axis=1)
    result: NDArray[np.float64] = numer / denom
    return result


def _spectral_pade_value_at_q_one_sympy(
    series_coeffs: Sequence[NDArray[Any]],
    numerator_degree: int,
    denominator_degree: int,
) -> NDArray[Any]:
    """Per-grid-node sympy Padé for sympy-scalar spectral coefficients.

    Each grid node's MxM linear system is over sympy expressions
    (typically polynomial in ℏ). We reuse `_pade_value_at_q_one`
    by extracting the scalar coefficients at one grid index at a
    time. Slower than the float path by a factor of `N+1 x
    sympy_overhead`, but produces a grid of sympy rational
    functions that the caller can substitute / plot / optimise.
    """
    n_plus_1 = series_coeffs[0].shape[0]
    result = np.empty(n_plus_1, dtype=object)
    for i in range(n_plus_1):
        scalar_coeffs = [cast("sp.Expr", c[i]) for c in series_coeffs]
        result[i] = _pade_value_at_q_one(scalar_coeffs, numerator_degree, denominator_degree)
    return result
