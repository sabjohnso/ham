"""Spectral backend for SHAM — `Backend[np.ndarray]` over a spectral grid.

`SpectralBackend(grid, indep, scalar)` is generic over the per-element
scalar ring (PLAN.org D-1):

  - `scalar = "float"` — classical SHAM. Arrays are `float64`; the
    linear solver in `integrate_x` is `np.linalg.solve`; ℏ is a
    numeric parameter set externally and ℏ-curves are produced by
    re-running the spectral solve over a grid of ℏ values.
  - `scalar = "sympy"` — symbolic ℏ over numeric x. Arrays are
    `object` dtype carrying `sympy.Expr` per entry; the linear solver
    uses `sp.Matrix.LUsolve`; `hbar_curve_at` works directly because
    every grid entry is already a polynomial in ℏ.

Both scalars share the entire Backend interface; the only points of
variation are the array dtype and the linear-solve strategy in
`integrate_x`. `indep` is the sympy symbol the user writes their
problem in; it's used by `lift_xonly` to translate sympy expressions
into grid values (via `sp.lambdify` for float, element-wise `.subs`
for sympy).

S5b ships the backend in isolation — tests-only, no consumer wires it
into Series yet. That happens in S7.
"""

from typing import Any, Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ham.backend import Backend
from ham.grids import Grid

Scalar = Literal["float", "sympy"]
_NDArrAny = NDArray[Any]


def SpectralBackend(  # noqa: N802 -- constructor-style factory
    grid: Grid,
    indep: sp.Symbol,
    scalar: Scalar = "float",
) -> Backend[_NDArrAny]:
    """Build a `Backend[np.ndarray]` over the given `grid` and scalar.

    See module docstring for the contract. Raises `ValueError` if
    `scalar` is not `"float"` or `"sympy"`.
    """
    if scalar not in ("float", "sympy"):
        raise ValueError(f"SpectralBackend scalar must be 'float' or 'sympy'; got {scalar!r}.")

    n_plus_1 = grid.nodes.shape[0]

    if scalar == "float":
        dtype: type = np.float64
        zero_scalar: object = 0.0
        one_scalar: object = 1.0
    else:
        dtype = object
        zero_scalar = sp.Integer(0)
        one_scalar = sp.Integer(1)

    def zero() -> _NDArrAny:
        return np.zeros(n_plus_1, dtype=dtype)

    def one() -> _NDArrAny:
        if scalar == "sympy":
            return np.array([sp.Integer(1)] * n_plus_1, dtype=object)
        return np.ones(n_plus_1, dtype=dtype)

    def lift_xonly(expr: sp.Expr) -> _NDArrAny:
        if scalar == "float":
            f = sp.lambdify(indep, expr, modules="numpy")
            raw = f(grid.nodes)
            if np.isscalar(raw):
                return np.full(n_plus_1, raw, dtype=dtype)
            return np.asarray(raw, dtype=dtype)
        # Sympy scalar: substitute symbolically at each node so any
        # non-indep symbols (e.g. ℏ) survive into the result.
        return np.array(
            [expr.subs(indep, sp.Float(node)) for node in grid.nodes],
            dtype=object,
        )

    def diff_x(c: _NDArrAny, k: int) -> _NDArrAny:
        if k == 0:
            return c
        d_k = grid.differentiation_matrix_power(k)
        if scalar == "sympy":
            # numpy refuses float @ object cleanly; promoting D to
            # object dtype lets the element-wise float x sp.Expr
            # multiplication go through sympy's __mul__. Honest cost:
            # one O(N²) cast per call; an alternative cached
            # object-dtype D would double memory.
            return d_k.astype(object) @ c
        return d_k @ c

    def integrate_x(c: _NDArrAny) -> _NDArrAny:
        """Antiderivative from the left boundary of the grid's domain.

        Solve `D · F = c` with the constraint `F[left_idx] = 0`. The
        constraint replaces row `left_idx` of `D` with the unit row
        `e_{left_idx}` and entry `left_idx` of `rhs` with `0`. In
        Trefethen's grid ordering the left boundary node is at index
        `N` (since `nodes[0] = b`, `nodes[N] = a`).
        """
        d_mod = grid.differentiation_matrix.copy()
        rhs = c.copy()
        left_idx = n_plus_1 - 1
        if scalar == "sympy":
            d_mod = d_mod.astype(object)
            if rhs.dtype != object:
                rhs = rhs.astype(object)
        d_mod[left_idx, :] = 0
        d_mod[left_idx, left_idx] = 1
        rhs[left_idx] = zero_scalar
        if scalar == "float":
            return np.linalg.solve(d_mod, rhs)
        return _sympy_linear_solve(d_mod, rhs)

    def normalize(c: _NDArrAny) -> _NDArrAny:
        if scalar == "float":
            return c  # numpy floats have no canonical-form notion
        return np.array([sp.expand(entry) for entry in c], dtype=object)

    # Silence unused-binding warning on one_scalar in the closure;
    # one() returns ones either way and doesn't need the scalar.
    del one_scalar

    return Backend(
        zero=zero,
        one=one,
        lift_xonly=lift_xonly,
        diff_x=diff_x,
        integrate_x=integrate_x,
        normalize=normalize,
    )


def _sympy_linear_solve(matrix: _NDArrAny, rhs: _NDArrAny) -> _NDArrAny:
    """Solve `matrix · x = rhs` over sympy expressions via `sp.Matrix.LUsolve`.

    The result is returned as an object-dtype ndarray so it composes
    with the rest of the spectral pipeline (which uses object arrays
    end-to-end when `scalar = "sympy"`).
    """
    matrix_sym = sp.Matrix(matrix.tolist())
    rhs_sym = sp.Matrix(rhs.tolist())
    solution = matrix_sym.LUsolve(rhs_sym)
    n = rhs.shape[0]
    return np.array([solution[i] for i in range(n)], dtype=object)
