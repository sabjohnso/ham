"""Spectral backend + spectral LinearOperator factory for SHAM.

`SpectralBackend(grid, indep, scalar)` is the `Backend[np.ndarray]`
substrate: zero / one / lift_xonly / diff_x / integrate_x / normalize
over a spectral grid, generic in the per-element scalar ring (PLAN.org
D-1):

  - `scalar = "float"` — classical SHAM. Arrays are `float64`; the
    linear solver in `integrate_x` is `np.linalg.solve`; ℏ is a
    numeric parameter set externally and ℏ-curves are produced by
    re-running the spectral solve over a grid of ℏ values.
  - `scalar = "sympy"` — symbolic ℏ over numeric x. Arrays are
    `object` dtype carrying `sympy.Expr` per entry; the linear solver
    uses `sp.Matrix.LUsolve`; `hbar_curve_at` works directly because
    every grid entry is already a polynomial in ℏ.

`spectral_linear_operator(expr_in_u, dependent, indep, grid, scalar,
bcs)` builds a `LinearOperator[NDArray]` from a sympy expression that
is linear in `dependent(indep)` and its x-derivatives. It assembles
the dense `L_matrix` as `sum_k diag(coeff_k(nodes)) @ D^k` and pairs
it with `spectral_inverter`, which solves `L · u = rhs` under the
declared BCs by replacing one row per BC with the row that evaluates
`u^(bc.derivative_order)` at the boundary node and pinning the
matching RHS entry to `bc.value`. Symbolic-side counterpart is
`sympy_dsolve_inverter` in `ham.operator`; both have the same factory
shape (PLAN.org S3 / S6).
"""

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import sympy as sp
from numpy.typing import NDArray

from ham.backend import Backend
from ham.grids import Grid
from ham.operator import BoundaryCondition, LinearOperator

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
    dtype: type = np.float64 if scalar == "float" else object
    zero_scalar: object = 0.0 if scalar == "float" else sp.Integer(0)

    def zero() -> _NDArrAny:
        return np.zeros(n_plus_1, dtype=dtype)

    def one() -> _NDArrAny:
        if scalar == "sympy":
            return np.array([sp.Integer(1)] * n_plus_1, dtype=object)
        return np.ones(n_plus_1, dtype=dtype)

    def lift_xonly(expr: sp.Expr) -> _NDArrAny:
        return _evaluate_at_nodes(expr, indep, grid.nodes, scalar)

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

    return Backend(
        zero=zero,
        one=one,
        lift_xonly=lift_xonly,
        diff_x=diff_x,
        integrate_x=integrate_x,
        normalize=normalize,
    )


# --- LinearOperator factory ----------------------------------------------


def spectral_linear_operator(
    expr_in_u: sp.Expr,
    dependent: sp.Function,
    indep: sp.Symbol,
    grid: Grid,
    scalar: Scalar = "float",
    *,
    bcs: tuple[BoundaryCondition, ...] = (),
) -> LinearOperator[_NDArrAny]:
    """Build a `LinearOperator[NDArray]` from a linear-in-`u` sympy expression.

    `expr_in_u` must be linear in `dependent(indep)` and its
    x-derivatives — i.e. a sum of terms of the form
    `coeff(indep) * d^k u / dx^k`. The factory decomposes it into
    `(coeff, k)` pairs, evaluates each `coeff` on `grid.nodes`, and
    assembles `L_matrix = Σ_k diag(coeff_k(nodes)) · D^k`. Non-linear
    expressions (e.g. `u**2`) and constant-term expressions (e.g.
    `u + 1`) are rejected by a reconstruction check after extraction.

    `bcs` is interpreted by `spectral_inverter`: each `bc` replaces
    one row of `L_matrix` at solve time so the inverter honours
    `u^(bc.derivative_order)(bc.point) = bc.value`. The forward
    `action` (the bare matvec) is unaffected; BCs only affect the
    inverter, matching the standard spectral practice and the sympy
    side's `sympy_dsolve_inverter`.
    """
    pairs = _parse_linear_in_u(expr_in_u, dependent, indep)

    n_plus_1 = grid.nodes.shape[0]
    matrix_dtype: type = np.float64 if scalar == "float" else object
    l_matrix: _NDArrAny = np.zeros((n_plus_1, n_plus_1), dtype=matrix_dtype)
    for coeff_expr, order in pairs:
        coeff_at_nodes = _evaluate_at_nodes(coeff_expr, indep, grid.nodes, scalar)
        d_k = grid.differentiation_matrix_power(order)
        if scalar == "sympy":
            d_k = d_k.astype(object)
        # diag(coeff_at_nodes) @ D^k via broadcasting on the row index.
        l_matrix = l_matrix + coeff_at_nodes.reshape(-1, 1) * d_k

    inverter = spectral_inverter(l_matrix, bcs, grid, scalar)

    def action(c: _NDArrAny) -> _NDArrAny:
        return l_matrix @ c

    return LinearOperator(var=indep, action=action, bcs=bcs, inverter=inverter)


def spectral_inverter(
    l_matrix: _NDArrAny,
    bcs: tuple[BoundaryCondition, ...],
    grid: Grid,
    scalar: Scalar,
) -> Callable[[_NDArrAny], _NDArrAny]:
    """Build the inverter that solves `L · u = rhs` under the declared BCs.

    For each `bc` the row /content/ is `D^(bc.derivative_order)`
    evaluated at the grid node closest to `bc.point` — that row,
    dotted with the unknown vector `u`, computes
    `u^(bc.derivative_order)` at that node, so setting the
    corresponding RHS entry to `bc.value` enforces the BC. The
    /placement/ row of `l_matrix` to overwrite is the closest unused
    row to that evaluation index, which lets multiple BCs at the
    same boundary node coexist (Blasius has both `f(0)=0` and
    `f'(0)=0` at η=0; the second one displaces inward to the next
    grid row, following Trefethen *Spectral Methods in MATLAB*
    Program 30). The modified system is then solved with
    `np.linalg.solve` (float) or `sp.Matrix.LUsolve` (sympy).

    Asymptotic BCs (`bc.point.is_infinite`) are rejected — a finite
    grid has no infinite node to anchor them to. Semi-infinite
    problems with asymptotic BCs need `RationalChebGrid` plus
    `spectral_inverter` extensions (the rational-Cheb D matrix has
    a zero row at the infinity node, so additional BC handling is
    required there — tracked as a PLAN.org follow-up).
    """
    n_plus_1 = grid.nodes.shape[0]
    used_rows: set[int] = set()
    replacements: list[tuple[int, NDArray[np.float64], sp.Expr]] = []
    for bc in bcs:
        if not bc.point.is_finite:
            raise ValueError(
                f"spectral_inverter requires a finite BC point on a finite grid; "
                f"got bc.point = {bc.point!r}. Semi-infinite problems need a "
                f"rational-Chebyshev grid (S5b ships only ChebGLGrid)."
            )
        point_val = float(bc.point)
        a, b = grid.domain
        if not a <= point_val <= b:
            raise ValueError(f"BC point {point_val} lies outside the grid's domain {grid.domain}.")
        eval_idx = int(np.argmin(np.abs(grid.nodes - point_val)))
        replacement_row = grid.differentiation_matrix_power(bc.derivative_order)[eval_idx, :]
        # Pick the closest unused matrix row for placement; additional
        # BCs at the same boundary node displace inward to adjacent rows.
        distances_from_eval = np.abs(np.arange(n_plus_1) - eval_idx)
        candidate_order = np.argsort(distances_from_eval, kind="stable")
        try:
            placement_idx = next(int(i) for i in candidate_order if int(i) not in used_rows)
        except StopIteration as exc:
            raise ValueError(
                f"spectral_inverter cannot place more BCs than grid rows; "
                f"got {len(bcs)} BCs on a grid with {n_plus_1} nodes."
            ) from exc
        used_rows.add(placement_idx)
        replacements.append((placement_idx, replacement_row, bc.value))

    def invert(rhs: _NDArrAny) -> _NDArrAny:
        l_mod = l_matrix.copy()
        rhs_mod = rhs.copy()
        if scalar == "sympy":
            if l_mod.dtype != object:
                l_mod = l_mod.astype(object)
            if rhs_mod.dtype != object:
                rhs_mod = rhs_mod.astype(object)
        for idx, row, value in replacements:
            l_mod[idx, :] = row.astype(object) if scalar == "sympy" else row
            rhs_mod[idx] = value if scalar == "sympy" else float(value)
        if scalar == "float":
            return np.linalg.solve(l_mod, rhs_mod)
        return _sympy_linear_solve(l_mod, rhs_mod)

    return invert


# --- helpers --------------------------------------------------------------


def _evaluate_at_nodes(
    expr: sp.Expr,
    indep: sp.Symbol,
    nodes: NDArray[np.float64],
    scalar: Scalar,
) -> _NDArrAny:
    """Evaluate `expr(indep)` at each `node`, dtype-appropriate to `scalar`.

    For float: lambdify and call; broadcast scalar results to the
    grid's node count. For sympy: substitute symbolically at each
    node so any non-indep symbols (e.g. ℏ) survive into the result.
    """
    n_plus_1 = nodes.shape[0]
    if scalar == "float":
        f = sp.lambdify(indep, expr, modules="numpy")
        raw = f(nodes)
        if np.isscalar(raw):
            return np.full(n_plus_1, raw, dtype=np.float64)
        return np.asarray(raw, dtype=np.float64)
    return np.array(
        [expr.subs(indep, sp.Float(node)) for node in nodes],
        dtype=object,
    )


def _parse_linear_in_u(
    expr_in_u: sp.Expr,
    dependent: sp.Function,
    indep: sp.Symbol,
) -> list[tuple[sp.Expr, int]]:
    """Decompose a linear-in-`u` sympy expression into `(coeff(x), derivative_order)` pairs.

    Walks the expanded form of `expr_in_u`, finds every distinct
    derivative order of `dependent(indep)` that appears, and extracts
    the (degree-1) coefficient of each via `sp.Expr.coeff`. A
    reconstruction check (`Σ coeff_k · u^(k)` must equal the input
    after `sp.expand`) catches non-linearity, constant terms, and
    other shapes the matrix builder cannot handle.
    """
    expanded = sp.expand(expr_in_u)
    u_of_x = dependent(indep)

    orders: set[int] = set()
    for sub in sp.preorder_traversal(expanded):
        if sub == u_of_x:
            orders.add(0)
        elif isinstance(sub, sp.Derivative) and sub.expr == u_of_x:
            orders.add(int(sub.variable_count[0][1]))

    pairs: list[tuple[sp.Expr, int]] = []
    reconstructed: sp.Expr = sp.Integer(0)
    for order in sorted(orders):
        target = u_of_x if order == 0 else sp.Derivative(u_of_x, indep, order)
        coeff = expanded.coeff(target)
        pairs.append((coeff, order))
        reconstructed = reconstructed + coeff * target

    if sp.expand(reconstructed - expanded) != 0:
        raise ValueError(
            f"spectral_linear_operator: expression is not linear in "
            f"{dependent}({indep}), or carries a u-free constant term. "
            f"Input: {expr_in_u!r}; reconstructed linear part: {reconstructed!r}; "
            f"difference: {sp.expand(expanded - reconstructed)!r}."
        )

    return pairs


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
