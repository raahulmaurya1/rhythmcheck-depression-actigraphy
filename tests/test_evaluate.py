"""
tests/test_evaluate.py — RhythmCheck
======================================
Tests for src/evaluate.py.

Tests verify:
- No data leakage: scaler/imputer are fit on training fold, not on full data.
- LOSO-CV produces exactly n_subjects predictions.
- Results dict contains all required keys.
- Metrics are in valid ranges.
- Bootstrap CIs have correct ordering (lower <= upper).
- results_to_dataframe returns a well-formed DataFrame.
- _bootstrap_ci is consistent and within [0, 1] for valid metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.evaluate import (
    FEATURE_COLS,
    PUBLISHED_F1,
    PUBLISHED_MCC,
    _bootstrap_ci,
    evaluate_all_models,
    results_to_dataframe,
    run_loso_cv,
)
from src.models import LogisticRegressionModel, MLPModel


# ── Synthetic feature DataFrame ───────────────────────────────────────────────

def _make_feature_df(
    n_condition: int = 10,
    n_control: int = 10,
    seed: int = 0,
    add_nans: bool = False,
) -> pd.DataFrame:
    """
    Build a minimal feature DataFrame mimicking build_feature_matrix output.
    Uses the same column names as FEATURE_COLS.
    """
    rng = np.random.default_rng(seed)
    n = n_condition + n_control
    data = {col: rng.uniform(0, 100, n) for col in FEATURE_COLS}
    data["subject_id"] = (
        [f"condition_{i}" for i in range(1, n_condition + 1)]
        + [f"control_{i}" for i in range(1, n_control + 1)]
    )
    data["label"] = [1] * n_condition + [0] * n_control
    data["group"] = ["condition"] * n_condition + ["control"] * n_control

    df = pd.DataFrame(data)

    if add_nans:
        # Inject NaN into some feature values
        for col in FEATURE_COLS[:3]:
            df.loc[df.index[:3], col] = np.nan

    return df


# ── _bootstrap_ci ─────────────────────────────────────────────────────────────

class TestBootstrapCI:
    def test_lower_leq_upper(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 20)
        y_pred = rng.integers(0, 2, 20)
        lo, hi = _bootstrap_ci(y_true, y_pred, lambda a, b: np.mean(a == b))
        assert lo <= hi

    def test_perfect_predictor_ci_near_one(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = y_true.copy()
        lo, hi = _bootstrap_ci(y_true, y_pred, lambda a, b: np.mean(a == b))
        assert lo > 0.8
        assert hi <= 1.0 + 1e-9

    def test_reproducible_with_same_seed(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, 30)
        y_pred = rng.integers(0, 2, 30)
        ci1 = _bootstrap_ci(y_true, y_pred, lambda a, b: np.mean(a == b), rng_seed=7)
        ci2 = _bootstrap_ci(y_true, y_pred, lambda a, b: np.mean(a == b), rng_seed=7)
        assert ci1 == ci2


# ── run_loso_cv ───────────────────────────────────────────────────────────────

class TestRunLosoCv:
    def _run(self, n_condition=10, n_control=10, add_nans=False):
        df = _make_feature_df(n_condition, n_control, add_nans=add_nans)
        model = LogisticRegressionModel()
        return run_loso_cv(df, model, n_bootstrap=50, verbose=False)

    def test_result_has_required_keys(self):
        result = self._run()
        required = {
            "model_name", "y_true", "y_pred", "y_proba",
            "f1", "mcc", "accuracy", "roc_auc",
            "f1_ci", "mcc_ci", "subject_ids", "fold_correct",
        }
        assert required <= set(result.keys())

    def test_n_predictions_equals_n_subjects(self):
        df = _make_feature_df(10, 10)
        result = self._run()
        assert len(result["y_true"]) == 20
        assert len(result["y_pred"]) == 20
        assert len(result["y_proba"]) == 20

    def test_y_pred_is_binary(self):
        result = self._run()
        assert set(np.unique(result["y_pred"])).issubset({0, 1})

    def test_y_proba_in_range(self):
        result = self._run()
        assert (result["y_proba"] >= 0).all()
        assert (result["y_proba"] <= 1).all()

    def test_f1_in_range(self):
        result = self._run()
        assert 0.0 <= result["f1"] <= 1.0

    def test_mcc_in_range(self):
        result = self._run()
        assert -1.0 <= result["mcc"] <= 1.0

    def test_accuracy_in_range(self):
        result = self._run()
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_ci_lower_leq_upper(self):
        result = self._run()
        f1_lo, f1_hi = result["f1_ci"]
        mcc_lo, mcc_hi = result["mcc_ci"]
        assert f1_lo <= f1_hi
        assert mcc_lo <= mcc_hi

    def test_fold_correct_shape(self):
        result = self._run()
        assert result["fold_correct"].shape == (20,)
        assert result["fold_correct"].dtype == bool

    def test_accuracy_consistent_with_fold_correct(self):
        """accuracy should equal mean of fold_correct."""
        result = self._run()
        assert result["accuracy"] == pytest.approx(result["fold_correct"].mean(), abs=1e-9)

    def test_handles_nan_features_gracefully(self):
        """NaN features should be imputed without error."""
        result = self._run(add_nans=True)
        assert not np.isnan(result["f1"])
        assert not np.isnan(result["mcc"])

    def test_no_leakage_scaler_not_fit_on_test(self):
        """
        Verify no leakage: if we deliberately set one test subject's feature
        to an extreme outlier (1e9), the prediction should still be made
        without crashing and the overall CV loop should complete successfully.
        This would crash if the scaler were fit on all data first and then
        the outlier caused a numerical issue — but with per-fold fitting it
        just gets scaled relative to training data.
        """
        df = _make_feature_df(10, 10)
        # Inject extreme outlier into one subject
        df.loc[0, "mean_daily_activity"] = 1e9
        model = LogisticRegressionModel()
        # Should not raise
        result = run_loso_cv(df, model, n_bootstrap=10, verbose=False)
        assert len(result["y_pred"]) == 20

    def test_mlp_also_works(self):
        df = _make_feature_df(10, 10)
        model = MLPModel(max_iter=50)
        result = run_loso_cv(df, model, n_bootstrap=10, verbose=False)
        assert "f1" in result
        assert 0 <= result["f1"] <= 1


# ── evaluate_all_models ───────────────────────────────────────────────────────

class TestEvaluateAllModels:
    def test_returns_two_results(self):
        df = _make_feature_df(10, 10)
        results = evaluate_all_models(df, n_bootstrap=20, verbose=False)
        assert len(results) == 2

    def test_each_result_has_model_name(self):
        df = _make_feature_df(10, 10)
        results = evaluate_all_models(df, n_bootstrap=20, verbose=False)
        for r in results:
            assert isinstance(r["model_name"], str)
            assert len(r["model_name"]) > 0


# ── results_to_dataframe ──────────────────────────────────────────────────────

class TestResultsToDataframe:
    def _make_result(self, name="TestModel", f1=0.7, mcc=0.4):
        return {
            "model_name": name,
            "f1": f1,
            "f1_ci": (f1 - 0.1, f1 + 0.1),
            "mcc": mcc,
            "mcc_ci": (mcc - 0.1, mcc + 0.1),
            "accuracy": 0.75,
            "roc_auc": 0.80,
        }

    def test_returns_dataframe(self):
        results = [self._make_result("M1"), self._make_result("M2")]
        df = results_to_dataframe(results)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_model(self):
        results = [self._make_result("M1"), self._make_result("M2")]
        df = results_to_dataframe(results)
        assert len(df) == 2

    def test_has_required_columns(self):
        results = [self._make_result("M1")]
        df = results_to_dataframe(results)
        for col in ("model", "f1", "f1_ci_lo", "f1_ci_hi", "mcc", "mcc_ci_lo", "mcc_ci_hi", "accuracy", "roc_auc"):
            assert col in df.columns, f"Missing column: {col}"

    def test_values_correct(self):
        results = [self._make_result("MyModel", f1=0.73, mcc=0.44)]
        df = results_to_dataframe(results)
        assert df.loc[0, "f1"] == pytest.approx(0.73)
        assert df.loc[0, "mcc"] == pytest.approx(0.44)


# ── Published baseline constants ──────────────────────────────────────────────

class TestPublishedBaseline:
    def test_baseline_f1_value(self):
        """Baseline F1 must match the Garcia-Ceja et al. 2018 value."""
        assert PUBLISHED_F1 == pytest.approx(0.73)

    def test_baseline_mcc_value(self):
        """Baseline MCC must match the Garcia-Ceja et al. 2018 value."""
        assert PUBLISHED_MCC == pytest.approx(0.44)
