"""
tests/test_explainability.py — RhythmCheck
===========================================
Tests for src/explainability.py.

Tests verify:
- run_shap returns required keys.
- Plots are saved to the expected paths.
- SHAP values have the correct shape.
- mean_abs_shap is a Series sorted descending.
- feature_ranking contains all feature names.
- select_best_model_for_shap picks the model with the highest F1.
- Fresh model instance is returned (not the same object).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.explainability import run_shap, select_best_model_for_shap, FEATURE_LABELS
from src.evaluate import FEATURE_COLS
from src.models import LogisticRegressionModel, MLPModel, get_models


# ── Synthetic feature DataFrame ───────────────────────────────────────────────

def _make_feature_df(n_condition: int = 8, n_control: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_condition + n_control
    data = {col: rng.uniform(0, 100, n) for col in FEATURE_COLS}
    data["subject_id"] = (
        [f"condition_{i}" for i in range(1, n_condition + 1)]
        + [f"control_{i}" for i in range(1, n_control + 1)]
    )
    data["label"] = [1] * n_condition + [0] * n_control
    data["group"] = ["condition"] * n_condition + ["control"] * n_control
    return pd.DataFrame(data)


# ── run_shap ──────────────────────────────────────────────────────────────────

class TestRunShap:
    def _run(self, tmp_path, model=None):
        df = _make_feature_df()
        m = model or LogisticRegressionModel()
        return run_shap(df, m, output_dir=tmp_path, show_plots=False), df

    def test_returns_required_keys(self, tmp_path):
        result, _ = self._run(tmp_path)
        required = {"shap_values", "mean_abs_shap", "feature_ranking",
                    "bar_plot_path", "beeswarm_path"}
        assert required <= set(result.keys())

    def test_shap_values_shape(self, tmp_path):
        df = _make_feature_df()
        result = run_shap(df, LogisticRegressionModel(), output_dir=tmp_path)
        assert result["shap_values"].shape == (len(df), len(FEATURE_COLS))

    def test_bar_plot_saved(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert Path(result["bar_plot_path"]).exists()

    def test_beeswarm_saved(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert Path(result["beeswarm_path"]).exists()

    def test_mean_abs_shap_sorted_descending(self, tmp_path):
        result, _ = self._run(tmp_path)
        vals = result["mean_abs_shap"].values
        assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))

    def test_feature_ranking_contains_all_features(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert set(result["feature_ranking"]) == set(FEATURE_COLS)

    def test_mean_abs_shap_non_negative(self, tmp_path):
        result, _ = self._run(tmp_path)
        assert (result["mean_abs_shap"].values >= 0).all()

    def test_works_with_mlp(self, tmp_path):
        """MLP uses KernelExplainer — should also work without error."""
        df = _make_feature_df()
        model = MLPModel(max_iter=50)
        result = run_shap(df, model, output_dir=tmp_path, show_plots=False)
        assert "shap_values" in result
        assert result["shap_values"].shape == (len(df), len(FEATURE_COLS))

    def test_output_dir_created_if_missing(self, tmp_path):
        new_dir = tmp_path / "nested" / "figures"
        df = _make_feature_df()
        run_shap(df, LogisticRegressionModel(), output_dir=new_dir)
        assert new_dir.exists()


# ── select_best_model_for_shap ────────────────────────────────────────────────

class TestSelectBestModel:
    def _make_results(self, f1_lr=0.70, f1_mlp=0.75):
        return [
            {"model_name": "LogisticRegression(C=1.0)", "f1": f1_lr},
            {"model_name": "MLP[32]",                   "f1": f1_mlp},
        ]

    def test_picks_higher_f1_model(self):
        results = self._make_results(f1_lr=0.60, f1_mlp=0.75)
        models = get_models()
        best = select_best_model_for_shap(results, models)
        assert isinstance(best, MLPModel)

    def test_picks_lr_when_lr_wins(self):
        results = self._make_results(f1_lr=0.80, f1_mlp=0.70)
        models = get_models()
        best = select_best_model_for_shap(results, models)
        assert isinstance(best, LogisticRegressionModel)

    def test_returns_fresh_instance_not_same_object(self):
        results = self._make_results(f1_lr=0.80, f1_mlp=0.70)
        models = get_models()
        best = select_best_model_for_shap(results, models)
        assert best is not models[0]

    def test_returns_base_model(self):
        from src.models import BaseModel
        results = self._make_results()
        models = get_models()
        best = select_best_model_for_shap(results, models)
        assert isinstance(best, BaseModel)


# ── FEATURE_LABELS ────────────────────────────────────────────────────────────

class TestFeatureLabels:
    def test_all_feature_cols_have_labels(self):
        """Every column in FEATURE_COLS must have a human-readable label."""
        for col in FEATURE_COLS:
            assert col in FEATURE_LABELS, f"No label for feature: {col}"

    def test_labels_are_non_empty_strings(self):
        for col, label in FEATURE_LABELS.items():
            assert isinstance(label, str) and len(label) > 0
