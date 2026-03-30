"""Statistical testing utilities for model comparison.

Provides normality-aware test selection, multiple comparison correction,
effect size computation, bootstrap confidence intervals, and DeLong's
test for comparing AUC values.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def auto_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Auto-select paired t-test or Wilcoxon based on normality.

    Uses the Shapiro-Wilk test on the paired differences to decide
    between a parametric (paired t-test) or non-parametric (Wilcoxon
    signed-rank) test.

    Parameters
    ----------
    scores_a, scores_b : np.ndarray
        Paired score arrays of equal length (e.g. per-fold metrics).
    alpha : float
        Significance level for the Shapiro-Wilk normality test.

    Returns
    -------
    dict[str, float]
        Keys: ``"statistic"``, ``"p_value"``, ``"test"`` (0 = t-test,
        1 = Wilcoxon), ``"normality_p"`` (Shapiro-Wilk p-value).
    """
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Arrays must have equal length: {len(scores_a)} vs {len(scores_b)}"
        )

    diffs = scores_a - scores_b

    # Shapiro-Wilk requires n >= 3
    if len(diffs) >= 3:
        _, normality_p = stats.shapiro(diffs)
    else:
        normality_p = 0.0  # too few samples; default to non-parametric

    if normality_p >= alpha:
        # Differences are approximately normal -> paired t-test
        stat, p_value = stats.ttest_rel(scores_a, scores_b)
        test_type = 0.0  # t-test
    else:
        # Non-normal -> Wilcoxon signed-rank
        # Handle edge case where all differences are zero
        if np.allclose(diffs, 0.0):
            stat, p_value = 0.0, 1.0
        else:
            stat, p_value = stats.wilcoxon(diffs)
        test_type = 1.0  # Wilcoxon

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "test": test_type,
        "normality_p": float(normality_p),
    }


def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> List[float]:
    """Holm-Bonferroni correction for multiple comparisons.

    Adjusts p-values using the step-down Holm-Bonferroni method,
    which is uniformly more powerful than the classical Bonferroni
    correction while still controlling the family-wise error rate.

    Parameters
    ----------
    p_values : list of float
        Raw (unadjusted) p-values.
    alpha : float
        Target family-wise error rate (for reference; adjustment is
        applied regardless).

    Returns
    -------
    list of float
        Adjusted p-values, capped at 1.0.
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted = np.zeros(n)
    for i, idx in enumerate(sorted_indices):
        adjusted[idx] = sorted_p[i] * (n - i)

    # Enforce monotonicity: each adjusted p must be >= all previous
    result = np.empty(n)
    result[sorted_indices[0]] = min(adjusted[sorted_indices[0]], 1.0)
    running_max = adjusted[sorted_indices[0]]
    for i in range(1, n):
        idx = sorted_indices[i]
        running_max = max(running_max, adjusted[idx])
        result[idx] = min(running_max, 1.0)

    return result.tolist()


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size for two independent groups.

    Uses the pooled standard deviation as the denominator.

    Parameters
    ----------
    group1, group2 : np.ndarray
        Score arrays for the two groups.

    Returns
    -------
    float
        Cohen's d.  Positive when ``mean(group1) > mean(group2)``.
    """
    group1 = np.asarray(group1, dtype=np.float64)
    group2 = np.asarray(group2, dtype=np.float64)

    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Each group must have at least 2 observations.")

    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0.0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def bootstrap_ci(
    statistic_fn: Callable,
    data: np.ndarray,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    statistic_fn : callable
        Function that computes a scalar statistic from a 1-D array.
    data : np.ndarray
        Observed data (1-D array).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci : float
        Confidence level, e.g. 0.95 for 95% CI.
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    tuple of (float, float)
        Lower and upper bounds of the confidence interval.
    """
    data = np.asarray(data, dtype=np.float64)
    rng = np.random.RandomState(rng_seed)

    bootstrap_stats = np.empty(n_bootstrap)
    n = len(data)
    for i in range(n_bootstrap):
        sample = data[rng.randint(0, n, size=n)]
        bootstrap_stats[i] = statistic_fn(sample)

    lower_pct = (1.0 - ci) / 2.0 * 100
    upper_pct = (1.0 + ci) / 2.0 * 100

    lower = float(np.percentile(bootstrap_stats, lower_pct))
    upper = float(np.percentile(bootstrap_stats, upper_pct))

    return lower, upper


def benjamini_hochberg(
    p_values: List[float],
    alpha: float = 0.05,
) -> Tuple[List[bool], List[float]]:
    """Benjamini-Hochberg procedure for False Discovery Rate control.

    Controls the expected proportion of false discoveries among rejected
    hypotheses, which is less conservative than Holm-Bonferroni (FWER
    control) and yields more power in large-scale testing scenarios.

    Parameters
    ----------
    p_values : list of float
        Raw (unadjusted) p-values.
    alpha : float
        Target FDR level (default 0.05).

    Returns
    -------
    tuple of (list of bool, list of float)
        ``(rejected, adjusted_p_values)`` where ``rejected[i]`` indicates
        whether hypothesis *i* is rejected at the given FDR level, and
        ``adjusted_p_values[i]`` is the BH-adjusted p-value (capped at 1.0).

    References
    ----------
    Benjamini, Y. and Hochberg, Y., 1995. Controlling the false discovery
    rate: a practical and powerful approach to multiple testing. Journal of
    the Royal Statistical Society: Series B, 57(1), pp.289-300.
    """
    n = len(p_values)
    if n == 0:
        return [], []

    p_arr = np.asarray(p_values, dtype=np.float64)

    # Sort p-values in ascending order
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Compute adjusted p-values: p_adj[i] = p[i] * n / (rank)
    # where rank is 1-indexed position in sorted order
    ranks = np.arange(1, n + 1, dtype=np.float64)
    adjusted_sorted = sorted_p * n / ranks

    # Enforce monotonicity from the right (step-up):
    # adjusted[i] = min(adjusted[i], adjusted[i+1]) working backwards
    for i in range(n - 2, -1, -1):
        adjusted_sorted[i] = min(adjusted_sorted[i], adjusted_sorted[i + 1])

    # Cap at 1.0
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Map back to original order
    adjusted = np.empty(n)
    adjusted[sorted_indices] = adjusted_sorted

    # Determine rejections
    rejected = [bool(adj <= alpha) for adj in adjusted]

    return rejected, adjusted.tolist()


def delong_test(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """DeLong's test for comparing two AUC values.

    Tests the null hypothesis that the AUCs of two classifiers
    (evaluated on the same dataset) are equal.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    scores_a : np.ndarray
        Predicted scores from model A.
    scores_b : np.ndarray
        Predicted scores from model B.

    Returns
    -------
    float
        Two-sided p-value.

    References
    ----------
    DeLong, E.R., DeLong, D.M. and Clarke-Pearson, D.L., 1988.
    Comparing the areas under two or more correlated receiver operating
    characteristic curves: a nonparametric approach. Biometrics, 44(3),
    pp.837-845.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    # Separate positive and negative indices
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    m = len(pos_idx)  # number of positives
    n = len(neg_idx)  # number of negatives

    if m == 0 or n == 0:
        raise ValueError("Both positive and negative samples are required.")

    # Compute structural components (placement values)
    def _placement_values(scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute DeLong placement values for positives and negatives."""
        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]

        # V_pos[i] = fraction of negatives with score < pos_scores[i]
        # + 0.5 * fraction of negatives with score == pos_scores[i]
        v_pos = np.zeros(m)
        for i in range(m):
            v_pos[i] = (
                np.sum(neg_scores < pos_scores[i])
                + 0.5 * np.sum(neg_scores == pos_scores[i])
            ) / n

        # V_neg[j] = fraction of positives with score > neg_scores[j]
        # + 0.5 * fraction of positives with score == neg_scores[j]
        v_neg = np.zeros(n)
        for j in range(n):
            v_neg[j] = (
                np.sum(pos_scores > neg_scores[j])
                + 0.5 * np.sum(pos_scores == neg_scores[j])
            ) / m

        return v_pos, v_neg

    v_pos_a, v_neg_a = _placement_values(scores_a)
    v_pos_b, v_neg_b = _placement_values(scores_b)

    # AUC estimates
    auc_a = np.mean(v_pos_a)
    auc_b = np.mean(v_pos_b)

    # Covariance matrix of (AUC_a, AUC_b)
    # S10: covariance from positive placements
    s10 = np.cov(
        np.stack([v_pos_a, v_pos_b]), ddof=1
    )  # 2x2

    # S01: covariance from negative placements
    s01 = np.cov(
        np.stack([v_neg_a, v_neg_b]), ddof=1
    )  # 2x2

    # Combined covariance
    s = s10 / m + s01 / n

    # Variance of AUC_a - AUC_b
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        return 1.0  # cannot distinguish

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2.0 * stats.norm.sf(np.abs(z))

    return float(p_value)
