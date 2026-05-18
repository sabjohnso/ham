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

Two concrete grids ship:

  - `ChebGLGrid(N, domain=(a, b))` — Chebyshev-Gauss-Lobatto on a
    finite interval. The default for SHAM on bounded domains.
  - `RationalChebGrid(N, L=1.0)` — algebraic-map rational-Chebyshev
    on [0, infinity). Useful when the problem's natural domain is
    semi-infinite and a finite truncation would distort the solve.
    Supports rational-Cheb quadrature (`quadrature_weights`) and is
    paired with `spectral_inverter`'s asymptotic-BC handling for
    value-at-infinity BCs (`f(infinity) = A`) and basis-auto
    homogeneous derivative BCs (`f^(k)(infinity) = 0` for k >= 1).
    Nonzero asymptotic-derivative BCs (`f'(infinity) = 1` à la
    Blasius) need a variable transformation by the user that reduces
    them to the basis-auto homogeneous case; the inverter rejects
    them with an explanatory error.
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
        a, b = self._domain
        w_ref = _clenshaw_curtis_weights_reference(self._N)
        return w_ref * (b - a) / 2.0


class RationalChebGrid:
    """Rational-Chebyshev grid on `[0, infinity)` via the algebraic map.

    Maps the Chebyshev reference interval `[-1, 1]` to `[0, infinity)`
    via `x = L · (1 + xi) / (1 - xi)`, with `L > 0` a characteristic
    length controlling node density near the origin (smaller `L`
    clusters more nodes near 0; larger `L` stretches the grid
    further into `x`). Nodes follow Trefethen's ordering (decreasing
    from `xi = 1` at index 0 to `xi = -1` at index N):

        xi_j = cos(pi · j / N),    x_j = L · (1 + xi_j) / (1 - xi_j).

    At index 0 (`xi = 1`) the physical node is at infinity; it is
    stored as `np.inf` in `nodes`.

    The differentiation matrix is built by the chain rule. With
    `rho_j = dxi/dx|_{xi=xi_j} = (1 - xi_j)^2 / (2 L)`,

        D_x = diag(rho) @ D_xi,

    where `D_xi` is the standard ChebGL differentiation matrix on
    `[-1, 1]`. At index 0 the row is identically zero because
    `dxi/dx = 0` at the infinity image — a fact this implementation
    exposes rather than papers over, since `spectral_inverter`'s
    `bc.point.is_finite` guard already rules out the asymptotic BCs
    that would require special handling there.

    The quadrature weights transform Clenshaw-Curtis on `[-1, 1]`
    via the Jacobian `dx/dxi = 2L/(1-xi)^2`:

        w_x_j = w_xi_j · 2 L / (1 - xi_j)^2,    j = 1..N

    with `w_x_0 = 0` by convention — the integrand is assumed to
    decay at infinity (the SHAM residual on a converged solve does),
    so the j=0 contribution is dropped rather than tripping a
    `0 · inf` indeterminate. For integrands that grow at infinity
    this quadrature returns nonsense; users should pick problem
    formulations whose residual decays at the outflow boundary.

    Asymptotic BCs:

      - `f(infinity) = A`: supported via `spectral_inverter`
        (identity row at the infinity index, RHS = A).
      - `f^(k)(infinity) = 0` for k >= 1: basis-automatic. The
        rational-Cheb polynomial `F(xi)` of degree N gives
        `f^(k)(x) → 0` as `x → infinity` by construction (chain-rule
        factors of `(1-xi)^(2k)` kill the derivative at xi=1). The
        inverter accepts these BCs and silently skips them.
      - `f^(k)(infinity) = A != 0` for k >= 1: NOT supported by
        this basis (would conflict with the basis assumption above).
        Examples like Blasius `f'(infinity) = 1` need a variable
        transformation by the user (e.g. `f = x + g`) that reduces
        them to the basis-auto homogeneous case; the inverter
        rejects nonzero asymptotic-derivative BCs with an
        explanatory error.

    `lift_xonly` of expressions that diverge at infinity (e.g.,
    `x` itself) will produce `np.inf` / `nan` at index 0; choose
    `u_0` and `H` with finite limits at infinity.
    """

    def __init__(self, N: int, L: float = 1.0) -> None:  # noqa: N803 -- N is mathematical
        if N < 1:
            raise ValueError(f"RationalChebGrid requires N >= 1; got N={N}.")
        if L <= 0:
            raise ValueError(f"RationalChebGrid requires L > 0; got L={L}.")
        self._N = N
        self._L = float(L)
        self._xi = np.cos(np.pi * np.arange(N + 1) / N)
        with np.errstate(divide="ignore"):
            self._nodes = self._L * (1.0 + self._xi) / (1.0 - self._xi)
        self._D = self._compute_differentiation_matrix()
        self._weights = self._compute_quadrature_weights()
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
        return (0.0, float("inf"))

    @property
    def L(self) -> float:  # noqa: N802 -- mathematical name
        """The characteristic-length parameter controlling node density."""
        return self._L

    def differentiation_matrix_power(self, k: int) -> NDArray[np.float64]:
        """Return `D_x^k`, memoised on the grid instance."""
        if k < 0:
            raise ValueError(f"differentiation_matrix_power requires k >= 0; got k={k}.")
        if k not in self._D_powers:
            base_k = max(j for j in self._D_powers if j < k)
            result = self._D_powers[base_k]
            for _ in range(k - base_k):
                result = result @ self._D
            self._D_powers[k] = result
        return self._D_powers[k]

    def _compute_differentiation_matrix(self) -> NDArray[np.float64]:
        # Reference Chebyshev-GL differentiation matrix in xi-space.
        n = self._N
        j = np.arange(n + 1)
        c = np.ones(n + 1)
        c[0] = 2.0
        c[n] = 2.0
        c *= (-1.0) ** j
        x_mat = np.tile(self._xi.reshape(-1, 1), (1, n + 1))
        dx_mat = x_mat - x_mat.T
        d_xi = np.outer(c, 1.0 / c) / (dx_mat + np.eye(n + 1))
        d_xi -= np.diag(d_xi.sum(axis=1))
        # Chain rule: D_x = diag(rho) @ D_xi with rho = dxi/dx.
        rho = (1.0 - self._xi) ** 2 / (2.0 * self._L)
        result: NDArray[np.float64] = rho.reshape(-1, 1) * d_xi
        return result

    def _compute_quadrature_weights(self) -> NDArray[np.float64]:
        """Rational-Cheb quadrature weights via Jacobian transformation.

        Clenshaw-Curtis weights on `[-1, 1]` in xi-space, scaled by
        the Jacobian `dx/dxi = 2L / (1 - xi)^2`. At j=0 (xi=1) the
        Jacobian is infinite; the weight there is set to 0 by
        convention since the integrand is assumed to decay at
        infinity. Integrands that grow at infinity violate that
        assumption and this quadrature will return nonsense for
        them.
        """
        w_xi = _clenshaw_curtis_weights_reference(self._N)
        with np.errstate(divide="ignore", invalid="ignore"):
            jacobian = 2.0 * self._L / (1.0 - self._xi) ** 2
        weights = w_xi * jacobian
        weights[0] = 0.0  # by convention; integrand assumed to decay at infty
        return weights


def _clenshaw_curtis_weights_reference(n: int) -> NDArray[np.float64]:
    """Clenshaw-Curtis weights on `[-1, 1]` for the N+1 Cheb-GL nodes.

    Trefethen's `clencurt.m` (*Spectral Methods in MATLAB*, Program 30).
    Shared between `ChebGLGrid` (domain-scaled) and `RationalChebGrid`
    (Jacobian-scaled), keeping a single implementation site for the
    O(N) Fourier-cosine series that the weights factor through.
    """
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
    return w_ref
