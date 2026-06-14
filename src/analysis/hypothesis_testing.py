"""
Statistical hypothesis testing suite for churn analysis.

Tests performed:
  H1 — Chi-square: contract type vs churn
  H2 — Welch's t-test: monthly charges by churn group
  H3 — Mann-Whitney U: tenure by churn group
  H4 — Chi-square: senior citizen status vs churn
  H5 — Mann-Whitney U: service count by churn group
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


ALPHA = 0.05


@dataclass
class HypothesisResult:
    name: str
    description: str
    test_type: str
    statistic: float
    p_value: float
    alpha: float = ALPHA
    effect_size: float | None = None
    effect_size_label: str | None = None
    additional: dict[str, Any] = field(default_factory=dict)

    @property
    def rejected(self) -> bool:
        return self.p_value < self.alpha

    @property
    def conclusion(self) -> str:
        if self.rejected:
            return (
                f"Reject H₀ (p={self.p_value:.4f} < α={self.alpha}). "
                f"{self.description} — statistically significant."
            )
        return (
            f"Fail to reject H₀ (p={self.p_value:.4f} ≥ α={self.alpha}). "
            f"Insufficient evidence for {self.description.lower()}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "test_type": self.test_type,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 6),
            "alpha": self.alpha,
            "rejected": self.rejected,
            "effect_size": round(self.effect_size, 4) if self.effect_size is not None else None,
            "effect_size_label": self.effect_size_label,
            "conclusion": self.conclusion,
            **self.additional,
        }


def _cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramér's V effect size for chi-square tests."""
    return float(np.sqrt(chi2 / (n * (min(k, r) - 1))))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent groups."""
    pooled_std = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def _rank_biserial_r(u: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation for Mann-Whitney U."""
    return float(1 - (2 * u) / (n1 * n2))


def _label_effect(value: float, thresholds: tuple[float, float, float]) -> str:
    small, medium, large = thresholds
    if abs(value) < small:
        return "negligible"
    if abs(value) < medium:
        return "small"
    if abs(value) < large:
        return "medium"
    return "large"


class HypothesisTester:
    def __init__(self, df: pd.DataFrame, alpha: float = ALPHA) -> None:
        self.df = df.copy()
        self.alpha = alpha
        self._add_service_count()

    def _add_service_count(self) -> None:
        if "service_count" not in self.df.columns:
            cols = [
                "online_security", "online_backup", "device_protection",
                "tech_support", "streaming_tv", "streaming_movies", "multiple_lines",
            ]
            self.df["service_count"] = sum(
                (self.df[c] == "Yes").astype(int) for c in cols if c in self.df.columns
            )

    # ------------------------------------------------------------------ #
    # H1: Contract type → churn (Chi-square)
    # ------------------------------------------------------------------ #
    def test_contract_vs_churn(self) -> HypothesisResult:
        ct = pd.crosstab(self.df["contract"], self.df["churn"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = _cramers_v(chi2, n, ct.shape[1], ct.shape[0])
        effect_label = _label_effect(v, (0.1, 0.3, 0.5))

        churn_by_contract = (
            self.df.groupby("contract")["churn"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "churn_rate", "count": "n"})
            .round(4)
            .to_dict(orient="index")
        )

        logger.info(f"[H1] Chi-square={chi2:.3f}, p={p:.4f}, V={v:.3f}")
        return HypothesisResult(
            name="contract_vs_churn",
            description="Contract type significantly affects churn rate",
            test_type="Chi-square",
            statistic=chi2,
            p_value=p,
            alpha=self.alpha,
            effect_size=v,
            effect_size_label=f"Cramér's V = {v:.3f} ({effect_label})",
            additional={"dof": dof, "churn_by_contract": churn_by_contract},
        )

    # ------------------------------------------------------------------ #
    # H2: Monthly charges → churn (Welch's t-test)
    # ------------------------------------------------------------------ #
    def test_monthly_charges_vs_churn(self) -> HypothesisResult:
        churned = self.df.loc[self.df["churn"] == 1, "monthly_charges"].dropna()
        retained = self.df.loc[self.df["churn"] == 0, "monthly_charges"].dropna()
        t, p = stats.ttest_ind(churned, retained, equal_var=False)
        d = _cohens_d(churned.values, retained.values)
        effect_label = _label_effect(d, (0.2, 0.5, 0.8))

        logger.info(f"[H2] t={t:.3f}, p={p:.4f}, d={d:.3f}")
        return HypothesisResult(
            name="monthly_charges_vs_churn",
            description="Churners pay higher monthly charges than retained customers",
            test_type="Welch's t-test",
            statistic=t,
            p_value=p,
            alpha=self.alpha,
            effect_size=d,
            effect_size_label=f"Cohen's d = {d:.3f} ({effect_label})",
            additional={
                "churned_mean": round(float(churned.mean()), 2),
                "retained_mean": round(float(retained.mean()), 2),
                "churned_std": round(float(churned.std()), 2),
                "retained_std": round(float(retained.std()), 2),
            },
        )

    # ------------------------------------------------------------------ #
    # H3: Tenure → churn (Mann-Whitney U)
    # ------------------------------------------------------------------ #
    def test_tenure_vs_churn(self) -> HypothesisResult:
        churned = self.df.loc[self.df["churn"] == 1, "tenure"].dropna()
        retained = self.df.loc[self.df["churn"] == 0, "tenure"].dropna()
        u, p = stats.mannwhitneyu(churned, retained, alternative="less")
        r = _rank_biserial_r(u, len(churned), len(retained))
        effect_label = _label_effect(r, (0.1, 0.3, 0.5))

        logger.info(f"[H3] U={u:.1f}, p={p:.4f}, r={r:.3f}")
        return HypothesisResult(
            name="tenure_vs_churn",
            description="Churners have shorter tenure than retained customers",
            test_type="Mann-Whitney U",
            statistic=u,
            p_value=p,
            alpha=self.alpha,
            effect_size=r,
            effect_size_label=f"Rank-biserial r = {r:.3f} ({effect_label})",
            additional={
                "churned_median": float(churned.median()),
                "retained_median": float(retained.median()),
                "churned_mean": round(float(churned.mean()), 2),
                "retained_mean": round(float(retained.mean()), 2),
            },
        )

    # ------------------------------------------------------------------ #
    # H4: Senior citizen → churn (Chi-square)
    # ------------------------------------------------------------------ #
    def test_senior_citizen_vs_churn(self) -> HypothesisResult:
        ct = pd.crosstab(self.df["senior_citizen"], self.df["churn"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = ct.values.sum()
        v = _cramers_v(chi2, n, ct.shape[1], ct.shape[0])
        effect_label = _label_effect(v, (0.1, 0.3, 0.5))

        churn_by_group = (
            self.df.groupby("senior_citizen")["churn"]
            .mean()
            .rename(index={0: "Non-senior", 1: "Senior"})
            .round(4)
            .to_dict()
        )

        logger.info(f"[H4] Chi-square={chi2:.3f}, p={p:.4f}, V={v:.3f}")
        return HypothesisResult(
            name="senior_citizen_vs_churn",
            description="Senior citizens churn at a different rate than non-seniors",
            test_type="Chi-square",
            statistic=chi2,
            p_value=p,
            alpha=self.alpha,
            effect_size=v,
            effect_size_label=f"Cramér's V = {v:.3f} ({effect_label})",
            additional={"dof": dof, "churn_by_group": churn_by_group},
        )

    # ------------------------------------------------------------------ #
    # H5: Service count → churn (Mann-Whitney U)
    # ------------------------------------------------------------------ #
    def test_service_count_vs_churn(self) -> HypothesisResult:
        churned = self.df.loc[self.df["churn"] == 1, "service_count"].dropna()
        retained = self.df.loc[self.df["churn"] == 0, "service_count"].dropna()
        u, p = stats.mannwhitneyu(churned, retained, alternative="two-sided")
        r = _rank_biserial_r(u, len(churned), len(retained))
        effect_label = _label_effect(r, (0.1, 0.3, 0.5))

        logger.info(f"[H5] U={u:.1f}, p={p:.4f}, r={r:.3f}")
        return HypothesisResult(
            name="service_count_vs_churn",
            description="Number of subscribed services differs between churners and retained customers",
            test_type="Mann-Whitney U",
            statistic=u,
            p_value=p,
            alpha=self.alpha,
            effect_size=r,
            effect_size_label=f"Rank-biserial r = {r:.3f} ({effect_label})",
            additional={
                "churned_median": float(churned.median()),
                "retained_median": float(retained.median()),
            },
        )

    def run_all(self) -> list[HypothesisResult]:
        tests = [
            self.test_contract_vs_churn,
            self.test_monthly_charges_vs_churn,
            self.test_tenure_vs_churn,
            self.test_senior_citizen_vs_churn,
            self.test_service_count_vs_churn,
        ]
        results = []
        for fn in tests:
            try:
                results.append(fn())
            except Exception as exc:
                logger.error(f"Test {fn.__name__} failed: {exc}")
        return results
