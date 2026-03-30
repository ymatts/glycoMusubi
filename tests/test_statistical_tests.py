"""Tests for statistical testing utilities.

Tests cover:
  - auto_test returns dict with p_value, test_name, statistic keys
  - Identical distributions give p > 0.05
  - Very different distributions give p < 0.05
  - Holm-Bonferroni correction: adjusted p-values >= unadjusted
  - Holm-Bonferroni: single p-value unchanged
  - Cohen's d: identical groups give d ~ 0
  - Cohen's d: known offset gives expected d
  - Bootstrap CI: contains true mean for normal distribution
  - Bootstrap CI: lower < upper
  - DeLong's test: identical predictions give p > 0.05
"""

from __future__ import annotations

import numpy as np
import pytest

from glycoMusubi.evaluation.statistical_tests import (
    auto_test,
    bootstrap_ci,
    cohens_d,
    delong_test,
    holm_bonferroni,
)


# ======================================================================
# TestAutoTest
# ======================================================================


class TestAutoTest:
    """Tests for the auto_test function."""

    def test_returns_required_keys(self) -> None:
        """Result dict contains statistic, p_value, test, normality_p."""
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87])
        b = np.array([0.8, 0.82, 0.79, 0.83, 0.81])
        result = auto_test(a, b)
        assert "statistic" in result
        assert "p_value" in result
        assert "test" in result
        assert "normality_p" in result

    def test_identical_distributions_high_p_value(self) -> None:
        """Samples drawn from the same distribution should give p > 0.05."""
        rng = np.random.RandomState(7)
        # Two independent samples from the same distribution
        a = rng.normal(0.9, 0.05, size=20)
        b = rng.normal(0.9, 0.05, size=20)
        result = auto_test(a, b)
        assert result["p_value"] > 0.05

    def test_very_different_distributions_low_p_value(self) -> None:
        """Clearly different scores should give p < 0.05."""
        a = np.array([0.95, 0.93, 0.94, 0.96, 0.92, 0.95, 0.94, 0.93])
        b = np.array([0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.52, 0.48])
        result = auto_test(a, b)
        assert result["p_value"] < 0.05

    def test_test_type_is_numeric(self) -> None:
        """test field is 0.0 (t-test) or 1.0 (Wilcoxon)."""
        a = np.array([0.9, 0.85, 0.88, 0.91, 0.87])
        b = np.array([0.8, 0.82, 0.79, 0.83, 0.81])
        result = auto_test(a, b)
        assert result["test"] in (0.0, 1.0)

    def test_mismatched_lengths_raises(self) -> None:
        """Arrays of different lengths raise ValueError."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="equal length"):
            auto_test(a, b)

    def test_small_sample_defaults_to_nonparametric(self) -> None:
        """With fewer than 3 samples, normality test is skipped -> Wilcoxon."""
        a = np.array([0.9, 0.8])
        b = np.array([0.7, 0.6])
        result = auto_test(a, b)
        assert result["normality_p"] == 0.0
        assert result["test"] == 1.0  # Wilcoxon

    def test_p_value_between_0_and_1(self) -> None:
        """p_value should always be in [0, 1]."""
        rng = np.random.RandomState(42)
        a = rng.normal(0.8, 0.05, size=10)
        b = rng.normal(0.82, 0.05, size=10)
        result = auto_test(a, b)
        assert 0.0 <= result["p_value"] <= 1.0


# ======================================================================
# TestHolmBonferroni
# ======================================================================


class TestHolmBonferroni:
    """Tests for the holm_bonferroni correction."""

    def test_adjusted_geq_unadjusted(self) -> None:
        """Adjusted p-values are always >= unadjusted p-values."""
        p_values = [0.01, 0.04, 0.03, 0.08, 0.002]
        adjusted = holm_bonferroni(p_values)
        for raw, adj in zip(p_values, adjusted):
            assert adj >= raw, f"adjusted {adj} < raw {raw}"

    def test_single_p_value_unchanged(self) -> None:
        """A single p-value is returned unchanged (multiplied by 1)."""
        p_values = [0.03]
        adjusted = holm_bonferroni(p_values)
        assert len(adjusted) == 1
        assert adjusted[0] == pytest.approx(0.03)

    def test_adjusted_capped_at_one(self) -> None:
        """Adjusted p-values are capped at 1.0."""
        p_values = [0.5, 0.6, 0.7, 0.8, 0.9]
        adjusted = holm_bonferroni(p_values)
        for adj in adjusted:
            assert adj <= 1.0

    def test_empty_input(self) -> None:
        """Empty list returns empty list."""
        assert holm_bonferroni([]) == []

    def test_known_example(self) -> None:
        """Verify against a manually computed example.

        p = [0.01, 0.04, 0.03]
        sorted: [0.01, 0.03, 0.04]
        adjusted (sorted): [0.01*3, 0.03*2, 0.04*1] = [0.03, 0.06, 0.04]
        with monotonicity enforcement: [0.03, 0.06, 0.06]
        map back to original order: [0.03, 0.06, 0.06]
        """
        p_values = [0.01, 0.04, 0.03]
        adjusted = holm_bonferroni(p_values)
        assert adjusted[0] == pytest.approx(0.03)
        assert adjusted[1] == pytest.approx(0.06)
        assert adjusted[2] == pytest.approx(0.06)

    def test_all_significant_remain_significant(self) -> None:
        """Very small p-values remain significant after correction."""
        p_values = [0.001, 0.002, 0.003]
        adjusted = holm_bonferroni(p_values)
        for adj in adjusted:
            assert adj < 0.05


# ======================================================================
# TestCohensD
# ======================================================================


class TestCohensD:
    """Tests for the cohens_d function."""

    def test_identical_groups_give_zero(self) -> None:
        """Identical groups should give d approximately 0."""
        group = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = cohens_d(group, group)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_known_offset(self) -> None:
        """Groups with known mean difference give expected d.

        Two groups with same std=1 and mean difference=1 give d=1.0.
        """
        rng = np.random.RandomState(42)
        n = 10000
        group1 = rng.normal(1.0, 1.0, size=n)
        group2 = rng.normal(0.0, 1.0, size=n)
        d = cohens_d(group1, group2)
        assert d == pytest.approx(1.0, abs=0.1)

    def test_sign_convention(self) -> None:
        """d is positive when group1 mean > group2 mean."""
        group1 = np.array([10.0, 11.0, 12.0, 13.0])
        group2 = np.array([1.0, 2.0, 3.0, 4.0])
        d = cohens_d(group1, group2)
        assert d > 0

    def test_reversed_sign(self) -> None:
        """d is negative when group1 mean < group2 mean."""
        group1 = np.array([1.0, 2.0, 3.0, 4.0])
        group2 = np.array([10.0, 11.0, 12.0, 13.0])
        d = cohens_d(group1, group2)
        assert d < 0

    def test_too_few_observations_raises(self) -> None:
        """Each group must have at least 2 observations."""
        with pytest.raises(ValueError, match="at least 2 observations"):
            cohens_d(np.array([1.0]), np.array([2.0, 3.0]))

    def test_zero_variance_returns_zero(self) -> None:
        """Groups with zero variance and equal means give d=0."""
        group = np.array([5.0, 5.0, 5.0])
        d = cohens_d(group, group)
        assert d == pytest.approx(0.0)


# ======================================================================
# TestBootstrapCI
# ======================================================================


class TestBootstrapCI:
    """Tests for the bootstrap_ci function."""

    def test_contains_true_mean(self) -> None:
        """95% CI for a normal distribution should contain the true mean.

        Using a large sample so the CI is tight around the true mean.
        """
        rng = np.random.RandomState(42)
        true_mean = 5.0
        data = rng.normal(true_mean, 1.0, size=500)

        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=5000, ci=0.95)
        assert lower < true_mean < upper

    def test_lower_less_than_upper(self) -> None:
        """Lower bound is strictly less than upper bound."""
        rng = np.random.RandomState(42)
        data = rng.normal(0.0, 1.0, size=100)
        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=2000)
        assert lower < upper

    def test_narrow_ci_with_low_variance(self) -> None:
        """Low variance data produces a narrow confidence interval."""
        data = np.array([1.0, 1.001, 0.999, 1.002, 0.998] * 20)
        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=2000)
        assert upper - lower < 0.01

    def test_wide_ci_with_high_variance(self) -> None:
        """High variance data produces a wider confidence interval."""
        rng = np.random.RandomState(42)
        data = rng.normal(0.0, 100.0, size=50)
        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=2000)
        assert upper - lower > 1.0

    def test_works_with_median(self) -> None:
        """Bootstrap CI works with median as the statistic."""
        rng = np.random.RandomState(42)
        data = rng.normal(10.0, 2.0, size=200)
        lower, upper = bootstrap_ci(np.median, data, n_bootstrap=2000)
        assert lower < 10.0 < upper

    def test_returns_floats(self) -> None:
        """CI bounds are Python floats."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lower, upper = bootstrap_ci(np.mean, data, n_bootstrap=100)
        assert isinstance(lower, float)
        assert isinstance(upper, float)


# ======================================================================
# TestDeLongTest
# ======================================================================


class TestDeLongTest:
    """Tests for delong_test."""

    def test_identical_predictions_high_p_value(self) -> None:
        """Identical predictions should give p > 0.05 (no difference)."""
        rng = np.random.RandomState(42)
        n = 100
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        scores = rng.rand(n)
        p = delong_test(y_true, scores, scores)
        assert p > 0.05

    def test_different_predictions_low_p_value(self) -> None:
        """A good model vs random model should give low p-value."""
        rng = np.random.RandomState(42)
        n = 200
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])

        # Good model: positive examples get high scores
        scores_good = np.where(y_true == 1, rng.uniform(0.7, 1.0, n), rng.uniform(0.0, 0.3, n))
        # Random model
        scores_random = rng.rand(n)

        p = delong_test(y_true, scores_good, scores_random)
        assert p < 0.05

    def test_returns_float(self) -> None:
        """DeLong's test returns a float p-value."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        scores_a = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.4])
        scores_b = np.array([0.8, 0.7, 0.4, 0.3, 0.6, 0.5])
        p = delong_test(y_true, scores_a, scores_b)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_requires_both_classes(self) -> None:
        """Raises ValueError if only one class present."""
        y_true = np.ones(10)
        scores = np.random.rand(10)
        with pytest.raises(ValueError, match="positive and negative"):
            delong_test(y_true, scores, scores)

    def test_symmetric(self) -> None:
        """DeLong's test is symmetric: p(a, b) == p(b, a)."""
        rng = np.random.RandomState(42)
        n = 50
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        scores_a = rng.rand(n)
        scores_b = rng.rand(n)

        p_ab = delong_test(y_true, scores_a, scores_b)
        p_ba = delong_test(y_true, scores_b, scores_a)
        assert p_ab == pytest.approx(p_ba, abs=1e-10)

    def test_perfect_vs_random(self) -> None:
        """Perfect model vs random should be highly significant."""
        n = 100
        y_true = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
        # Perfect model
        scores_perfect = y_true.astype(float)
        # Random model
        rng = np.random.RandomState(42)
        scores_random = rng.rand(n)

        p = delong_test(y_true, scores_perfect, scores_random)
        assert p < 0.01
