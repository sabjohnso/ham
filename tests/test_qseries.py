"""Property tests for QSeries — the truncated formal power series in q.

Stage 1 of the HAM build (see PLAN.org). Each property here corresponds to
an algebraic law that QSeries must satisfy.
"""

import sympy as sp
from ham.series import QSeries
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import (
    MAX_ORDER,
    X,
    one_qseries,
    qseries_polynomial_coeffs,
    three_qseries_same_order,
    two_qseries_same_order,
)


@given(raw_coeffs=st.lists(st.integers(min_value=-100, max_value=100), min_size=1, max_size=8))
def test_construction_round_trip(raw_coeffs: list[int]) -> None:
    """coeff(k) returns the k-th coefficient supplied at construction."""
    coeffs = [sp.Integer(c) for c in raw_coeffs]
    series = QSeries(coeffs, order=len(coeffs) - 1)
    for k, c in enumerate(coeffs):
        assert series.coeff(k) == c


# --- Additive abelian group laws -------------------------------------------


@given(pair=two_qseries_same_order())
def test_addition_is_coefficientwise(pair: tuple[QSeries, QSeries]) -> None:
    """[q^k](a + b) == [q^k]a + [q^k]b for all k <= order."""
    a, b = pair
    s = a + b
    for k in range(a.order + 1):
        assert s.coeff(k) == a.coeff(k) + b.coeff(k)


@given(pair=two_qseries_same_order())
def test_addition_is_commutative(pair: tuple[QSeries, QSeries]) -> None:
    """a + b == b + a, verified coefficient-wise."""
    a, b = pair
    for k in range(a.order + 1):
        assert (a + b).coeff(k) == (b + a).coeff(k)


@given(triple=three_qseries_same_order())
def test_addition_is_associative(triple: tuple[QSeries, QSeries, QSeries]) -> None:
    """(a + b) + c == a + (b + c), verified coefficient-wise."""
    a, b, c = triple
    for k in range(a.order + 1):
        assert ((a + b) + c).coeff(k) == (a + (b + c)).coeff(k)


@given(a=one_qseries())
def test_zero_is_additive_identity(a: QSeries) -> None:
    """a + 0 == a == 0 + a, verified coefficient-wise."""
    zero = QSeries.zero(a.order)
    for k in range(a.order + 1):
        assert (a + zero).coeff(k) == a.coeff(k)
        assert (zero + a).coeff(k) == a.coeff(k)


@given(a=one_qseries())
def test_negation_is_additive_inverse(a: QSeries) -> None:
    """a + (-a) == 0, verified coefficient-wise."""
    s = a + (-a)
    for k in range(a.order + 1):
        assert s.coeff(k) == sp.Integer(0)


# --- Scalar multiplication (module over sympy.Expr) ------------------------


@given(
    a=one_qseries(),
    raw_alpha=st.integers(min_value=-50, max_value=50),
)
def test_scalar_multiplication_is_coefficientwise(a: QSeries, raw_alpha: int) -> None:
    """[q^k](alpha * s) == alpha * [q^k]s for all k <= order."""
    alpha = sp.Integer(raw_alpha)
    s = alpha * a
    for k in range(a.order + 1):
        assert s.coeff(k) == alpha * a.coeff(k)


@given(
    a=one_qseries(),
    raw_alpha=st.integers(min_value=-50, max_value=50),
)
def test_scalar_multiplication_left_right_agree(a: QSeries, raw_alpha: int) -> None:
    """alpha * s == s * alpha (sympy.Expr coefficient ring is commutative)."""
    alpha = sp.Integer(raw_alpha)
    for k in range(a.order + 1):
        assert (alpha * a).coeff(k) == (a * alpha).coeff(k)


@given(
    pair=two_qseries_same_order(),
    raw_alpha=st.integers(min_value=-50, max_value=50),
)
def test_scalar_distributes_over_series_addition(
    pair: tuple[QSeries, QSeries], raw_alpha: int
) -> None:
    """alpha * (a + b) == alpha * a + alpha * b, verified coefficient-wise."""
    a, b = pair
    alpha = sp.Integer(raw_alpha)
    lhs = alpha * (a + b)
    rhs = alpha * a + alpha * b
    for k in range(a.order + 1):
        assert lhs.coeff(k) == rhs.coeff(k)


@given(a=one_qseries())
def test_scalar_one_is_identity(a: QSeries) -> None:
    """1 · s == s, verified coefficient-wise."""
    for k in range(a.order + 1):
        assert (sp.Integer(1) * a).coeff(k) == a.coeff(k)


@given(a=one_qseries())
def test_scalar_zero_annihilates(a: QSeries) -> None:
    """0 · s has all-zero coefficients."""
    s = sp.Integer(0) * a
    for k in range(a.order + 1):
        assert s.coeff(k) == sp.Integer(0)


# --- Multiplication (Cauchy product, truncated) ----------------------------


@given(pair=two_qseries_same_order())
def test_cauchy_product_order(pair: tuple[QSeries, QSeries]) -> None:
    """(a * b).order == a.order + b.order (polynomial-product semantics)."""
    a, b = pair
    assert (a * b).order == a.order + b.order


@given(pair=two_qseries_same_order())
def test_cauchy_product_coefficient_formula(pair: tuple[QSeries, QSeries]) -> None:
    """[q^k](a * b) == sum_{j=0..k} a_j * b_{k-j}."""
    a, b = pair
    p = a * b
    for k in range(p.order + 1):
        expected = sum(
            (a.coeff(j) * b.coeff(k - j) for j in range(k + 1)),
            sp.Integer(0),
        )
        assert sp.expand(p.coeff(k) - expected) == 0


@given(pair=two_qseries_same_order())
def test_cauchy_product_is_commutative(pair: tuple[QSeries, QSeries]) -> None:
    """a * b == b * a (the sympy.Expr coefficient ring is commutative)."""
    a, b = pair
    p1 = a * b
    p2 = b * a
    for k in range(p1.order + 1):
        assert sp.expand(p1.coeff(k) - p2.coeff(k)) == 0


@given(triple=three_qseries_same_order())
def test_cauchy_product_is_associative(
    triple: tuple[QSeries, QSeries, QSeries],
) -> None:
    """(a * b) * c == a * (b * c), verified coefficient-wise."""
    a, b, c = triple
    p1 = (a * b) * c
    p2 = a * (b * c)
    for k in range(p1.order + 1):
        assert sp.expand(p1.coeff(k) - p2.coeff(k)) == 0


@given(triple=three_qseries_same_order())
def test_cauchy_product_distributes_over_addition(
    triple: tuple[QSeries, QSeries, QSeries],
) -> None:
    """a * (b + c) == a * b + a * c, verified coefficient-wise."""
    a, b, c = triple
    lhs = a * (b + c)
    rhs = a * b + a * c
    for k in range(lhs.order + 1):
        assert sp.expand(lhs.coeff(k) - rhs.coeff(k)) == 0


@given(a=one_qseries())
def test_unit_series_is_multiplicative_identity(a: QSeries) -> None:
    """The constant series `1` acts as identity on a, coefficient-wise."""
    unit = QSeries([sp.Integer(1)], order=0)
    p = unit * a
    for k in range(a.order + 1):
        assert p.coeff(k) == a.coeff(k)


# --- Truncation ------------------------------------------------------------


@given(
    a=one_qseries(),
    n=st.integers(min_value=0, max_value=2 * MAX_ORDER),
)
def test_truncation_lowers_only(a: QSeries, n: int) -> None:
    """trunc(n).order == min(n, a.order); trunc never extends the order."""
    assert a.trunc(n).order == min(n, a.order)


@given(
    a=one_qseries(),
    n=st.integers(min_value=0, max_value=2 * MAX_ORDER),
)
def test_truncation_idempotent(a: QSeries, n: int) -> None:
    """trunc_n(trunc_n(s)) == trunc_n(s), coefficient-wise."""
    once = a.trunc(n)
    twice = once.trunc(n)
    assert once.order == twice.order
    for k in range(once.order + 1):
        assert once.coeff(k) == twice.coeff(k)


@given(
    a=one_qseries(),
    n=st.integers(min_value=0, max_value=2 * MAX_ORDER),
)
def test_truncation_preserves_low_coefficients(a: QSeries, n: int) -> None:
    """trunc(n) preserves coefficients up to the new order."""
    t = a.trunc(n)
    for k in range(t.order + 1):
        assert t.coeff(k) == a.coeff(k)


# --- Differentiation and integration in q ----------------------------------


@given(a=one_qseries())
def test_diff_q_lowers_order(a: QSeries) -> None:
    """diff_q(s).order == max(0, s.order - 1)."""
    assert a.diff_q().order == max(0, a.order - 1)


@given(a=one_qseries())
def test_diff_q_coefficient_formula(a: QSeries) -> None:
    """[q^k] diff_q(s) == (k+1) * [q^{k+1}] s."""
    d = a.diff_q()
    for k in range(d.order + 1):
        expected = sp.Integer(k + 1) * a.coeff(k + 1)
        assert sp.expand(d.coeff(k) - expected) == 0


@given(a=one_qseries())
def test_integrate_q_raises_order(a: QSeries) -> None:
    """integrate_q(s).order == s.order + 1."""
    assert a.integrate_q().order == a.order + 1


@given(a=one_qseries())
def test_integrate_q_coefficient_formula(a: QSeries) -> None:
    """[q^0] integrate_q(s) == 0, and [q^k] integrate_q(s) == [q^{k-1}] s / k for k >= 1."""
    i = a.integrate_q()
    assert i.coeff(0) == sp.Integer(0)
    for k in range(1, i.order + 1):
        expected = a.coeff(k - 1) / sp.Integer(k)
        assert sp.expand(i.coeff(k) - expected) == 0


@given(a=one_qseries())
def test_integrate_then_diff_is_identity(a: QSeries) -> None:
    """diff_q(integrate_q(s)) == s, coefficient-wise. Right inverse, no caveats."""
    result = a.integrate_q().diff_q()
    assert result.order == a.order
    for k in range(a.order + 1):
        assert sp.expand(result.coeff(k) - a.coeff(k)) == 0


@given(a=one_qseries())
def test_diff_then_integrate_modulo_constant(a: QSeries) -> None:
    """integrate_q(diff_q(s)) equals s, except [q^0] is reset to 0."""
    result = a.diff_q().integrate_q()
    assert result.coeff(0) == sp.Integer(0)
    for k in range(1, a.order + 1):
        assert sp.expand(result.coeff(k) - a.coeff(k)) == 0


# --- Coefficient-wise mapping (q/x gluing law) -----------------------------


def _diff_x(c: sp.Expr) -> sp.Expr:
    """Differentiate a coefficient with respect to the test symbol X."""
    return sp.diff(c, X)


@given(a=qseries_polynomial_coeffs())
def test_map_coeffs_preserves_order(a: QSeries) -> None:
    """map_coeffs preserves the q-order; it acts within each coefficient."""
    assert a.map_coeffs(_diff_x).order == a.order


@given(a=qseries_polynomial_coeffs())
def test_map_coeffs_commutes_with_coeff_extraction(a: QSeries) -> None:
    """[q^k] map_coeffs(f)(s) == f([q^k] s) — the q/x gluing law.

    Exercised here with f = d/dx, so the test is non-trivial only when
    coefficients depend on X (hence the polynomial-coefficient strategy).
    """
    mapped = a.map_coeffs(_diff_x)
    for k in range(a.order + 1):
        assert sp.expand(mapped.coeff(k) - _diff_x(a.coeff(k))) == 0


@given(a=qseries_polynomial_coeffs())
def test_map_coeffs_identity_is_no_op(a: QSeries) -> None:
    """map_coeffs(identity) == s, coefficient-wise (functor identity law)."""
    mapped = a.map_coeffs(lambda c: c)
    assert mapped.order == a.order
    for k in range(a.order + 1):
        assert mapped.coeff(k) == a.coeff(k)
