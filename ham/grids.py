"""Spectral grids for the SHAM substrate.

A `Grid` is a small object packaging the data a spectral backend needs:
nodes on a domain, a dense first-derivative matrix `D`, Clenshaw-
Curtis quadrature weights, and a memoised `D^k` for higher
derivatives. Different problem types demand different grid families
(Chebyshev-Gauss-Lobatto for finite intervals, rational-Chebyshev for
semi-infinite domains, Fourier for periodic problems); the Protocol
surface keeps the spectral backend, the spectral linear-operator
factory, and downstream diagnostics from caring which family they're
working with (PLAN.org D-3).

S5a ships `ChebGLGrid` only; other grid families are added by
implementing the protocol without touching the Backend or any consumer.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Grid(Protocol):
    """A spectral grid: nodes + differentiation matrix + quadrature + domain.

    The protocol is purposely small. `D^k` for higher derivatives is
    obtained through `differentiation_matrix_power(k)` so the grid can
    cache repeated requests (the spectral backend hits the same `k`
    many times per solve).
    """

    @property
    def nodes(self) -> NDArray[np.float64]: ...

    @property
    def differentiation_matrix(self) -> NDArray[np.float64]: ...

    @property
    def quadrature_weights(self) -> NDArray[np.float64]: ...

    @property
    def domain(self) -> tuple[float, float]: ...

    def differentiation_matrix_power(self, k: int) -> NDArray[np.float64]: ...


class ChebGLGrid:
    """Chebyshev-Gauss-Lobatto grid on `[a, b]`.

    Nodes follow Trefethen's convention (decreasing from `b` at index 0
    to `a` at index `N`):

        x_j = (a + b)/2 + (b - a)/2 · cos(π j / N),    j = 0..N.

    The differentiation matrix is from Trefethen's
    /Spectral Methods in MATLAB/ Program 6 (`cheb.m`), domain-scaled
    by `2/(b - a)` to translate the reference-grid derivative to the
    target domain. The quadrature weights are Clenshaw-Curtis on
    `[-1, 1]` (Trefethen's `clencurt.m`), scaled by `(b - a)/2` for
    the change of variables.

    `D^k` is memoised on the instance so the spectral backend's
    repeated calls during a solve share one computed matrix per power.
    """

    def __init__(self, N: int, domain: tuple[float, float] = (-1.0, 1.0)) -> None:  # noqa: N803 -- N is mathematical
        if N < 1:
            raise ValueError(f"ChebGLGrid requires N >= 1 (need at least 2 nodes); got N={N}.")
        a, b = domain
        if not a < b:
            raise ValueError(f"ChebGLGrid requires a < b; got domain=({a}, {b}).")
        self._N = N
        self._domain = (float(a), float(b))
        self._nodes = self._compute_nodes()
        self._D = self._compute_differentiation_matrix()
        self._weights = self._compute_quadrature_weights()
        # Pre-populate the cache with the identity (k=0) and the bare
        # differentiation matrix (k=1) so `differentiation_matrix_power`
        # can return them by identity, satisfying the memoisation
        # contract on first access.
        self._D_powers: dict[int, NDArray[np.float64]] = {
            0: np.eye(N + 1),
            1: self._D,
        }

    @property
    def nodes(self) -> NDArray[np.float64]:
        return self._nodes

    @property
    def differentiation_matrix(self) -> NDArray[np.float64]:
        return self._D

    @property
    def quadrature_weights(self) -> NDArray[np.float64]:
        return self._weights

    @property
    def domain(self) -> tuple[float, float]:
        return self._domain

    def differentiation_matrix_power(self, k: int) -> NDArray[np.float64]:
        """Return `D^k`, memoised on the grid instance."""
        if k < 0:
            raise ValueError(
                f"differentiation_matrix_power requires k >= 0; got k={k}. "
                f"Integration is handled by the spectral inverter, not by D^{-1}."
            )
        if k not in self._D_powers:
            # Build iteratively from the highest cached predecessor so
            # we don't re-multiply through every intermediate power on
            # each call.
            base_k = max(j for j in self._D_powers if j < k)
            result = self._D_powers[base_k]
            for _ in range(k - base_k):
                result = result @ self._D
            self._D_powers[k] = result
        return self._D_powers[k]

    # --- numerics ----------------------------------------------------------

    def _compute_nodes(self) -> NDArray[np.float64]:
        a, b = self._domain
        ref = np.cos(np.pi * np.arange(self._N + 1) / self._N)
        return (a + b) / 2.0 + (b - a) / 2.0 * ref

    def _compute_differentiation_matrix(self) -> NDArray[np.float64]:
        n = self._N
        a, b = self._domain
        # Reference grid on [-1, 1] in Trefethen ordering.
        j = np.arange(n + 1)
        x_ref = np.cos(np.pi * j / n)
        c = np.ones(n + 1)
        c[0] = 2.0
        c[n] = 2.0
        c *= (-1.0) ** j
        x_mat = np.tile(x_ref.reshape(-1, 1), (1, n + 1))
        dx_mat = x_mat - x_mat.T
        d_ref = np.outer(c, 1.0 / c) / (dx_mat + np.eye(n + 1))
        d_ref -= np.diag(d_ref.sum(axis=1))
        # Scale d/dt -> d/dx via dt/dx = 2/(b - a).
        result: NDArray[np.float64] = d_ref * (2.0 / (b - a))
        return result

    def _compute_quadrature_weights(self) -> NDArray[np.float64]:
        """Clenshaw-Curtis weights on [a, b]; Trefethen `clencurt.m` + domain scale."""
        n = self._N
        a, b = self._domain
        theta = np.pi * np.arange(n + 1) / n
        w_ref = np.zeros(n + 1)
        if n % 2 == 0:
            w_ref[0] = 1.0 / (n**2 - 1)
            w_ref[n] = w_ref[0]
            v = np.ones(n - 1)
            for k in range(1, n // 2):
                v -= 2.0 * np.cos(2 * k * theta[1:n]) / (4 * k**2 - 1)
            v -= np.cos(n * theta[1:n]) / (n**2 - 1)
        else:
            w_ref[0] = 1.0 / n**2
            w_ref[n] = w_ref[0]
            v = np.ones(n - 1)
            for k in range(1, (n - 1) // 2 + 1):
                v -= 2.0 * np.cos(2 * k * theta[1:n]) / (4 * k**2 - 1)
        w_ref[1:n] = 2.0 * v / n
        return w_ref * (b - a) / 2.0
