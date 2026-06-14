"""Tests for the hypothesis testing module."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.generate_data import generate_customers
from src.analysis.hypothesis_testing import HypothesisResult, HypothesisTester


@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    return generate_customers(n=1000, seed=99)


@pytest.fixture(scope="module")
def tester(sample_df) -> HypothesisTester:
    return HypothesisTester(sample_df)


class TestHypothesisTester:
    def test_contract_vs_churn_returns_result(self, tester):
        r = tester.test_contract_vs_churn()
        assert isinstance(r, HypothesisResult)
        assert r.p_value >= 0
        assert r.statistic >= 0

    def test_monthly_charges_vs_churn(self, tester):
        r = tester.test_monthly_charges_vs_churn()
        assert isinstance(r, HypothesisResult)
        assert "churned_mean" in r.additional
        assert "retained_mean" in r.additional

    def test_tenure_vs_churn(self, tester):
        r = tester.test_tenure_vs_churn()
        assert isinstance(r, HypothesisResult)
        assert r.test_type == "Mann-Whitney U"

    def test_senior_citizen_vs_churn(self, tester):
        r = tester.test_senior_citizen_vs_churn()
        assert isinstance(r, HypothesisResult)
        assert "churn_by_group" in r.additional

    def test_service_count_vs_churn(self, tester):
        r = tester.test_service_count_vs_churn()
        assert isinstance(r, HypothesisResult)

    def test_run_all_returns_five_results(self, tester):
        results = tester.run_all()
        assert len(results) == 5
        assert all(isinstance(r, HypothesisResult) for r in results)

    def test_result_to_dict(self, tester):
        r = tester.test_contract_vs_churn()
        d = r.to_dict()
        assert "p_value" in d
        assert "rejected" in d
        assert "conclusion" in d
        assert isinstance(d["rejected"], bool)

    def test_p_value_range(self, tester):
        for r in tester.run_all():
            assert 0.0 <= r.p_value <= 1.0, f"{r.name} has invalid p-value: {r.p_value}"

    def test_conclusion_text(self, tester):
        for r in tester.run_all():
            assert len(r.conclusion) > 10
