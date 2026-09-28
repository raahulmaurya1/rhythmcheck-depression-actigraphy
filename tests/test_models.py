"""
tests/test_models.py — RhythmCheck
=====================================
Tests for src/models.py.

Tests verify:
- Shared interface: both models expose fit / predict / predict_proba / name.
- Output shapes and value ranges are correct.
- Models can fit and predict on simple synthetic data.
- Class probability rows sum to 1.
- get_models() returns both model types.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import (
    BaseModel,
    LogisticRegressionModel,
    MLPModel,
    get_models,
)


# ── Synthetic data fixtures ───────────────────────────────────────────────────

@pytest.fixture
def simple_data():
    """
    Linearly separable binary classification problem.
    Class 0: X < 0, Class 1: X > 0.
    Small sample (n=20) to mirror the spirit of the real dataset.
    """
    rng = np.random.default_rng(0)
    X0 = rng.normal(-2, 0.5, (10, 3))
    X1 = rng.normal(+2, 0.5, (10, 3))
    X = np.vstack([X0, X1])
    y = np.array([0] * 10 + [1] * 10)
    return X, y


# ── Interface compliance ──────────────────────────────────────────────────────

class TestInterface:
    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_is_base_model(self, model):
        assert isinstance(model, BaseModel)

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_has_name(self, model):
        assert isinstance(model.name, str)
        assert len(model.name) > 0

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_fit_returns_self(self, model, simple_data):
        X, y = simple_data
        result = model.fit(X, y)
        assert result is model

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_predict_shape(self, model, simple_data):
        X, y = simple_data
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_predict_values_binary(self, model, simple_data):
        X, y = simple_data
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_predict_proba_shape(self, model, simple_data):
        X, y = simple_data
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_predict_proba_sums_to_one(self, model, simple_data):
        X, y = simple_data
        model.fit(X, y)
        proba = model.predict_proba(X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    @pytest.mark.parametrize("model", [LogisticRegressionModel(), MLPModel()])
    def test_predict_proba_in_range(self, model, simple_data):
        X, y = simple_data
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert (proba >= 0).all()
        assert (proba <= 1).all()


# ── Logistic Regression specifics ────────────────────────────────────────────

class TestLogisticRegressionModel:
    def test_name_contains_c(self):
        model = LogisticRegressionModel(C=0.5)
        assert "0.5" in model.name

    def test_custom_c(self, simple_data):
        X, y = simple_data
        model = LogisticRegressionModel(C=0.1)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_sklearn_model_attribute(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegressionModel()
        assert isinstance(model.sklearn_model, LogisticRegression)

    def test_separable_data_high_accuracy(self, simple_data):
        """On linearly separable data, LR should get near-perfect accuracy."""
        X, y = simple_data
        model = LogisticRegressionModel()
        model.fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.9


# ── MLP specifics ─────────────────────────────────────────────────────────────

class TestMLPModel:
    def test_name_contains_layer_size(self):
        model = MLPModel(hidden_layer_sizes=(64,))
        assert "64" in model.name

    def test_custom_hidden_layers(self, simple_data):
        X, y = simple_data
        model = MLPModel(hidden_layer_sizes=(16, 8))
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(y)

    def test_sklearn_model_attribute(self):
        from sklearn.neural_network import MLPClassifier
        model = MLPModel()
        assert isinstance(model.sklearn_model, MLPClassifier)

    def test_separable_data_high_accuracy(self, simple_data):
        """On linearly separable data, MLP should also get good accuracy."""
        X, y = simple_data
        model = MLPModel()
        model.fit(X, y)
        acc = (model.predict(X) == y).mean()
        assert acc >= 0.8  # slightly looser than LR given stochasticity


# ── get_models factory ────────────────────────────────────────────────────────

class TestGetModels:
    def test_returns_two_models(self):
        models = get_models()
        assert len(models) == 2

    def test_first_is_logistic_regression(self):
        models = get_models()
        assert isinstance(models[0], LogisticRegressionModel)

    def test_second_is_mlp(self):
        models = get_models()
        assert isinstance(models[1], MLPModel)

    def test_all_are_base_models(self):
        for m in get_models():
            assert isinstance(m, BaseModel)

    def test_each_call_returns_fresh_instances(self):
        """get_models() should return new instances each call (no shared state)."""
        m1 = get_models()
        m2 = get_models()
        assert m1[0] is not m2[0]
        assert m1[1] is not m2[1]
