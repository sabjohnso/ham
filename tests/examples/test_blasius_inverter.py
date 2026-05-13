"""Tests for the closed-form basis-aware Blasius inverter (Stage 13a).

Pins the per-basis-element L^{-1}, the RHS decomposer, and end-to-end
equivalence with the Stage 11 dsolve-based inverter on the same RHSes.
"""

import pytest
import sympy as sp
from examples.blasius_exponential import _blasius_exponential_inverter
from examples.blasius_inverter import make_blasius_inverter

ETA = sp.Symbol("eta", positive=True)
ALPHA = sp.Symbol("alpha", positive=True)
HBAR = sp.Symbol("hbar")


def _make() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return (eta, alpha, hbar) for tests that need fresh symbols."""
    return ETA, ALPHA, HBAR


# --- basis_l_inverse: closed-form per-basis-element inversions ------------


def _verify_basis_solution(result: sp.Expr, rhs: sp.Expr) -> None:
    """Assert `result` satisfies L[result] = rhs and the three Blasius BCs."""
    assert sp.simplify(result.subs(ETA, 0)) == 0
    assert sp.simplify(sp.diff(result, ETA).subs(ETA, 0)) == 0
    # f'(∞) = 0 — sympy can resolve this for the basis elements after alpha-sub
    deriv = sp.diff(result, ETA).subs(ALPHA, 1)
    assert sp.simplify(sp.limit(deriv, ETA, sp.oo)) == 0
    lhs = sp.diff(result, ETA, 3) - ALPHA**2 * sp.diff(result, ETA)
    assert sp.simplify(lhs - rhs) == 0


def test_basis_l_inverse_rejects_k_zero() -> None:
    """L^{-1}(η^j) (k = 0) does not admit f'(∞) = 0; basis_l_inverse raises.

    Under L = d^3/dη^3 - alpha^2·d/dη, the particular solution of L[u] = η^j
    has a non-decaying η^(j+1) term and the kernel {1, e^(-alphaη)} cannot
    cancel it. For HAM Blasius this case never arises (every RHS term
    has k ≥ 1), so raising is the correct loud-failure mode.
    """
    invert = make_blasius_inverter(ETA, ALPHA)
    with pytest.raises(NotImplementedError, match="k >= 1"):
        invert.basis_l_inverse(0, 0)
    with pytest.raises(NotImplementedError, match="k >= 1"):
        invert.basis_l_inverse(2, 0)


def test_basis_l_inverse_exp_decay_non_resonant() -> None:
    """L^{-1}(e^(-2alphaη)) with the homogeneous Blasius BCs."""
    invert = make_blasius_inverter(ETA, ALPHA)
    rhs = sp.exp(-2 * ALPHA * ETA)
    result = invert.basis_l_inverse(0, 2)
    _verify_basis_solution(result, rhs)


def test_basis_l_inverse_resonant_case() -> None:
    """L^{-1}(η·e^(-alphaη)) — the resonant k=1 case sympy.dsolve cannot
    symbolically resolve via the asymptotic BC. The cached inverter
    inherits the Stage 11 zero-free-constants workaround and produces a
    function satisfying L[u]=rhs and all three BCs.
    """
    invert = make_blasius_inverter(ETA, ALPHA)
    rhs = ETA * sp.exp(-ALPHA * ETA)
    result = invert.basis_l_inverse(1, 1)
    _verify_basis_solution(result, rhs)


def test_basis_l_inverse_cached() -> None:
    """Two calls to basis_l_inverse(j, k) return the *same* object — lru_cache hit."""
    invert = make_blasius_inverter(ETA, ALPHA)
    first = invert.basis_l_inverse(0, 2)
    second = invert.basis_l_inverse(0, 2)
    assert first is second


# --- decompose: rhs -> [(coeff, j, k)] ------------------------------------


def test_decompose_single_term_no_eta() -> None:
    """An η-free term decomposes as (coeff, 0, 0)."""
    invert = make_blasius_inverter(ETA, ALPHA)
    result = invert.decompose(sp.Rational(3, 2) * HBAR)
    assert result == [(sp.Rational(3, 2) * HBAR, 0, 0)]


def test_decompose_eta_times_exp() -> None:
    """The term `(alpha·ℏ/2)·η·exp(-alphaη)` decomposes as (alpha·ℏ/2, 1, 1)."""
    invert = make_blasius_inverter(ETA, ALPHA)
    term = ALPHA * HBAR * ETA * sp.exp(-ALPHA * ETA) / 2
    result = invert.decompose(term)
    assert len(result) == 1
    coeff, j, k = result[0]
    assert j == 1
    assert k == 1
    assert sp.simplify(coeff - ALPHA * HBAR / 2) == 0


def test_decompose_sum_of_four_terms_at_m1_rhs() -> None:
    """The actual HAM-step-1 RHS decomposes into the four basis terms."""
    invert = make_blasius_inverter(ETA, ALPHA)
    u0 = ETA - 1 / ALPHA + sp.exp(-ALPHA * ETA) / ALPHA
    n_u0 = sp.diff(u0, ETA, 3) + sp.Rational(1, 2) * u0 * sp.diff(u0, ETA, 2)
    rhs = sp.expand(HBAR * n_u0)
    result = invert.decompose(rhs)
    assert len(result) == 4
    # Each tuple has integer j, k and a coefficient free of η.
    for coeff, j, k in result:
        assert isinstance(j, int) and j >= 0
        assert isinstance(k, int) and k >= 0
        assert ETA not in coeff.free_symbols


def test_decompose_rejects_transcendental_factor() -> None:
    """A term containing sin(η) is outside the basis; raise."""
    invert = make_blasius_inverter(ETA, ALPHA)
    with pytest.raises(NotImplementedError, match="cannot decompose"):
        invert.decompose(sp.sin(ETA) * sp.exp(-ALPHA * ETA))


def test_decompose_rejects_negative_power_of_eta() -> None:
    """A term with η^(-1) is outside the basis; raise."""
    invert = make_blasius_inverter(ETA, ALPHA)
    with pytest.raises(NotImplementedError, match="η\\^n with n < 0"):
        invert.decompose(1 / ETA * sp.exp(-ALPHA * ETA))


def test_decompose_rejects_positive_exponential() -> None:
    """A term with exp(+alphaη) is outside the basis (k must be ≥ 0); raise."""
    invert = make_blasius_inverter(ETA, ALPHA)
    with pytest.raises(NotImplementedError, match="exp argument"):
        invert.decompose(sp.exp(ALPHA * ETA))


# --- end-to-end: closed-form inverter matches Stage 11 dsolve inverter ----


def test_invert_matches_dsolve_on_ham_step_1_rhs() -> None:
    """invert(rhs_m1) equals the Stage 11 dsolve-based inverter on the same RHS."""
    invert = make_blasius_inverter(ETA, ALPHA)
    u0 = ETA - 1 / ALPHA + sp.exp(-ALPHA * ETA) / ALPHA
    n_u0 = sp.diff(u0, ETA, 3) + sp.Rational(1, 2) * u0 * sp.diff(u0, ETA, 2)
    rhs = sp.expand(HBAR * n_u0)
    via_basis = invert(rhs)
    via_dsolve = _blasius_exponential_inverter(rhs)
    assert sp.simplify(via_basis - via_dsolve) == 0


def test_invert_of_simple_exponential_matches_known_closed_form() -> None:
    """invert(exp(-2alphaη)) equals the Liao closed-form.

    -1/(6 alpha^3) + e^(-alphaη)/(3 alpha^3) - e^(-2alphaη)/(6 alpha^3).
    """
    invert = make_blasius_inverter(ETA, ALPHA)
    result = invert(sp.exp(-2 * ALPHA * ETA))
    expected = (
        -sp.Rational(1, 6) / ALPHA**3
        + sp.exp(-ALPHA * ETA) / (3 * ALPHA**3)
        - sp.exp(-2 * ALPHA * ETA) / (6 * ALPHA**3)
    )
    assert sp.simplify(result - expected) == 0


def test_invert_of_zero_returns_zero() -> None:
    """invert(0) returns 0 (empty decomposition produces empty sum)."""
    invert = make_blasius_inverter(ETA, ALPHA)
    assert sp.simplify(invert(sp.Integer(0))) == 0
