"""
models.py — RhythmCheck
========================
Logistic Regression and small MLP models with a shared fit/predict interface.

Dataset citation:
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

Design notes
------------
- Both models implement the same interface: fit / predict / predict_proba.
- Logistic Regression uses L2 regularisation (primary model per spec).
- MLP uses 1 hidden layer of 32 units with dropout (comparison model).
- Hyperparameters are deliberately conservative given the small sample (n=55).
  No aggressive grid search; defaults are documented explicitly below.
- Scaling is NOT done inside the models — callers (evaluate.py) must pass
  already-scaled X, having fit the scaler on training folds only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


# -- Shared interface ----------------------------------------------------------

class BaseModel(ABC):
    """
    Minimal interface shared by all RhythmCheck classifiers.
    Callers receive a unified API regardless of the underlying algorithm.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class predictions (0 or 1)."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates, shape (n_samples, 2)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for logging/reporting."""


# -- Logistic Regression -------------------------------------------------------

class LogisticRegressionModel(BaseModel):
    """
    L2-regularised Logistic Regression — the primary model.

    Hyperparameters (conservative, documented explicitly per spec):
    - penalty : 'l2'        — shrinks coefficients, reduces overfitting on n=55.
    - C       : 1.0         — inverse regularisation strength; default sklearn value.
                              Not tuned via grid search given the sample size.
    - solver  : 'lbfgs'    — efficient for small datasets, supports L2.
    - max_iter: 1000        — generous; avoids convergence warnings.
    - class_weight: 'balanced' — compensates for 23/32 imbalance.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.C = C
        self.max_iter = max_iter
        self._model = LogisticRegression(
            C=self.C,
            solver="lbfgs",
            max_iter=self.max_iter,
            class_weight="balanced",
            random_state=42,
        )

    @property
    def name(self) -> str:
        return f"LogisticRegression(C={self.C})"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def coef_(self) -> np.ndarray:
        """Expose coefficients for SHAP compatibility."""
        return self._model.coef_

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_

    # Allow SHAP to use the underlying sklearn estimator directly
    @property
    def sklearn_model(self) -> LogisticRegression:
        return self._model


# -- MLP -----------------------------------------------------------------------

class MLPModel(BaseModel):
    """
    Small Multi-Layer Perceptron — the comparison model.

    Hyperparameters (deliberately conservative for n=55 dataset):
    - hidden_layer_sizes: (32,)   — single hidden layer, 32 units.
                                    Kept small to limit overfitting risk.
    - activation        : 'relu'  — standard for MLPs.
    - solver            : 'adam'  — adaptive learning rate, works well on small data.
    - alpha             : 0.01    — L2 regularisation (higher than sklearn default 1e-4)
                                    to mitigate overfitting on the small sample.
    - max_iter          : 500     — generous iteration budget.
    - early_stopping    : True    — holds out 10% of training data for validation;
                                    stops if validation score does not improve for
                                    n_iter_no_change=20 iterations. Reduces overfitting.
    - random_state      : 42      — reproducibility.

    Note: With n=55, leave-one-out leaves 54 training samples per fold.
    The MLP is intentionally not tuned aggressively — results are reported
    honestly and compared against LR without favouring either model.
    """

    def __init__(
        self,
        hidden_layer_sizes: tuple = (32,),
        alpha: float = 0.01,
        max_iter: int = 500,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.max_iter = max_iter
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            max_iter=self.max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=42,
        )

    @property
    def name(self) -> str:
        return f"MLP{list(self.hidden_layer_sizes)}"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLPModel":
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._model.classes_

    @property
    def sklearn_model(self) -> MLPClassifier:
        return self._model


# -- Factory helper ------------------------------------------------------------

def get_models() -> list[BaseModel]:
    """
    Return the two models used in the RhythmCheck pipeline:
    [LogisticRegressionModel, MLPModel].

    Called by evaluate.py and pipeline.py so they share the same model
    definitions without duplicating hyperparameter choices.
    """
    return [
        LogisticRegressionModel(C=1.0),
        MLPModel(hidden_layer_sizes=(32,), alpha=0.01),
    ]
