"""
evaluate.py — RhythmCheck
==========================
Leave-one-subject-out (LOSO) cross-validation evaluation for both models.

Dataset citation:
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

Published baseline to compare against: F1 = 0.73, MCC = 0.44
(Garcia-Ceja et al., 2018 — same dataset, same LOSO methodology)

Implementation notes
--------------------
- 55 folds total: one per subject.
- In each fold: StandardScaler + SimpleImputer are fit on the 54 training
  subjects ONLY, then applied to the single held-out subject.
  This is the strict no-leakage requirement from the spec.
- Primary metrics: F1 (macro) and MCC — chosen because the dataset is
  class-imbalanced (23 depressed vs 32 healthy) and raw accuracy is
  misleading. These match the dataset's own suggested metrics.
- Spread / uncertainty: 95% bootstrap confidence intervals (1000 resamples)
  on the aggregated predictions vector. With LOSO, each fold yields a single
  prediction — fold-level F1/MCC per fold is not meaningful (it would be
  0 or 1 for each single sample), so bootstrap CI is the standard approach
  for reporting uncertainty on aggregated LOSO results.
- ROC-AUC and accuracy reported as secondary metrics.
- Model comparison is reported honestly: whichever model wins wins.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.models import BaseModel, get_models

# -- Published baseline --------------------------------------------------------
PUBLISHED_F1 = 0.73
PUBLISHED_MCC = 0.44

# -- Feature columns used in modelling ----------------------------------------
FEATURE_COLS = [
    "mean_daily_activity",
    "std_daily_activity",
    "mean_minute_activity",
    "iv",
    "is_",
    "m10",
    "l5",
    "ra",
    "m10_onset",
    "l5_onset",
    "day_night_ratio",
    "sleep_efficiency",
]


# -- Bootstrap CI helper -------------------------------------------------------

def _bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_resamples: int = 1000,
    ci: float = 0.95,
    rng_seed: int = 42,
) -> tuple[float, float]:
    """
    Compute bootstrap confidence interval for a scalar metric.

    Parameters
    ----------
    y_true      : True labels.
    y_pred      : Predicted labels.
    metric_fn   : Callable(y_true, y_pred) -> float.
    n_resamples : Number of bootstrap samples.
    ci          : Confidence level (default 0.95 → 95% CI).
    rng_seed    : Random seed for reproducibility.

    Returns
    -------
    (lower, upper) confidence interval bounds.
    """
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    scores = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        scores.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    return float(np.quantile(scores, alpha)), float(np.quantile(scores, 1 - alpha))


# -- Core LOSO-CV -------------------------------------------------------------

def run_loso_cv(
    feature_df: pd.DataFrame,
    model: BaseModel,
    feature_cols: list[str] = FEATURE_COLS,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> dict:
    """
    Run leave-one-subject-out cross-validation for a single model.

    For each of the 55 subjects:
    1. Hold that subject out as the test set (1 sample).
    2. Fit StandardScaler + SimpleImputer on the remaining 54 subjects ONLY.
    3. Transform both train and held-out sets with those fitted transformers.
    4. Fit the model on the transformed training set.
    5. Predict label and probability for the held-out subject.

    Parameters
    ----------
    feature_df   : Output of feature_extraction.build_feature_matrix().
                   Must contain 'subject_id', 'label', and all feature_cols.
    model        : A BaseModel instance (LogisticRegressionModel or MLPModel).
    feature_cols : Feature columns to use. Defaults to FEATURE_COLS.
    n_bootstrap  : Number of bootstrap resamples for CI estimation.
    verbose      : Print per-fold progress if True.

    Returns
    -------
    dict with keys:
        model_name   : str
        y_true       : np.ndarray shape (55,)
        y_pred       : np.ndarray shape (55,)
        y_proba      : np.ndarray shape (55,) — P(depressed)
        f1           : float — macro F1 on aggregated predictions
        mcc          : float — MCC on aggregated predictions
        accuracy     : float
        roc_auc      : float
        f1_ci        : (float, float) — 95% bootstrap CI
        mcc_ci       : (float, float) — 95% bootstrap CI
        subject_ids  : list[str]
        fold_correct : np.ndarray[bool] — whether each fold was correct
    """
    subjects = feature_df["subject_id"].tolist()
    n = len(subjects)

    y_true_all = np.empty(n, dtype=int)
    y_pred_all = np.empty(n, dtype=int)
    y_proba_all = np.empty(n, dtype=float)
    fold_correct = np.empty(n, dtype=bool)

    for i, test_subject in enumerate(subjects):
        # -- Split -------------------------------------------------------------
        train_mask = feature_df["subject_id"] != test_subject
        test_mask = feature_df["subject_id"] == test_subject

        X_train = feature_df.loc[train_mask, feature_cols].values.astype(float)
        y_train = feature_df.loc[train_mask, "label"].values.astype(int)

        X_test = feature_df.loc[test_mask, feature_cols].values.astype(float)
        y_test = feature_df.loc[test_mask, "label"].values.astype(int)

        # -- Impute then scale — fit ONLY on training fold ------------------
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # -- Fit and predict -----------------------------------------------
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        proba = model.predict_proba(X_test)[0, 1]  # P(depressed)

        y_true_all[i] = y_test[0]
        y_pred_all[i] = pred
        y_proba_all[i] = proba
        fold_correct[i] = pred == y_test[0]

        if verbose and (i + 1) % 10 == 0:
            print(f"  [{model.name}] fold {i+1}/{n} done")

    # -- Aggregate metrics --------------------------------------------------
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_all, y_pred_all)
    acc = accuracy_score(y_true_all, y_pred_all)
    try:
        auc = roc_auc_score(y_true_all, y_proba_all)
    except ValueError:
        auc = float("nan")

    # -- Bootstrap CIs -----------------------------------------------------
    f1_ci = _bootstrap_ci(
        y_true_all, y_pred_all,
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        n_resamples=n_bootstrap,
    )
    mcc_ci = _bootstrap_ci(
        y_true_all, y_pred_all,
        matthews_corrcoef,
        n_resamples=n_bootstrap,
    )

    return {
        "model_name": model.name,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
        "y_proba": y_proba_all,
        "f1": f1,
        "mcc": mcc,
        "accuracy": acc,
        "roc_auc": auc,
        "f1_ci": f1_ci,
        "mcc_ci": mcc_ci,
        "subject_ids": subjects,
        "fold_correct": fold_correct,
    }


# -- Evaluation runner ---------------------------------------------------------

def evaluate_all_models(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
    n_bootstrap: int = 1000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run LOSO-CV for all models returned by get_models().

    Returns
    -------
    List of result dicts (one per model), ordered as returned by get_models().
    """
    models = get_models()
    results = []

    for model in models:
        print(f"\n[evaluate] Running LOSO-CV for {model.name} ...")
        result = run_loso_cv(
            feature_df,
            model,
            feature_cols=feature_cols,
            n_bootstrap=n_bootstrap,
            verbose=verbose,
        )
        results.append(result)
        _print_result(result)

    print("\n" + "=" * 60)
    _print_comparison(results)
    _print_baseline_comparison(results)

    return results


# -- Reporting helpers ---------------------------------------------------------

def _print_result(result: dict) -> None:
    """Print a clean summary for one model's LOSO-CV results."""
    f1_lo, f1_hi = result["f1_ci"]
    mcc_lo, mcc_hi = result["mcc_ci"]
    n_correct = result["fold_correct"].sum()
    n_total = len(result["fold_correct"])

    print(f"\n-- {result['model_name']} --")
    print(f"  F1  (macro):  {result['f1']:.3f}  [95% CI: {f1_lo:.3f} – {f1_hi:.3f}]")
    print(f"  MCC:          {result['mcc']:.3f}  [95% CI: {mcc_lo:.3f} – {mcc_hi:.3f}]")
    print(f"  Accuracy:     {result['accuracy']:.3f}  ({n_correct}/{n_total} correct)")
    print(f"  ROC-AUC:      {result['roc_auc']:.3f}")


def _print_comparison(results: list[dict]) -> None:
    """Print a side-by-side comparison of both models."""
    print("\nModel comparison (LOSO-CV on Depresjon dataset, n=55):")
    print(f"  {'Model':<30} {'F1':>6}  {'MCC':>6}  {'Acc':>6}  {'AUC':>6}")
    print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
    for r in results:
        print(
            f"  {r['model_name']:<30} "
            f"{r['f1']:>6.3f}  "
            f"{r['mcc']:>6.3f}  "
            f"{r['accuracy']:>6.3f}  "
            f"{r['roc_auc']:>6.3f}"
        )

    # Determine winner honestly
    if len(results) == 2:
        r0, r1 = results[0], results[1]
        if r0["f1"] > r1["f1"]:
            winner = r0["model_name"]
        elif r1["f1"] > r0["f1"]:
            winner = r1["model_name"]
        else:
            winner = "tied"
        print(f"\n  Better model by F1: {winner}")
        if winner != "tied" and abs(r0["f1"] - r1["f1"]) < 0.02:
            print("  (Difference < 0.02 — effectively a tie given sample size)")


def _print_baseline_comparison(results: list[dict]) -> None:
    """Explicitly compare best model against the published baseline."""
    best = max(results, key=lambda r: r["f1"])
    f1_diff = best["f1"] - PUBLISHED_F1
    mcc_diff = best["mcc"] - PUBLISHED_MCC

    print(f"\n-- Comparison to published baseline (Garcia-Ceja et al., 2018) --")
    print(f"  Published:  F1 = {PUBLISHED_F1:.2f},  MCC = {PUBLISHED_MCC:.2f}")
    print(
        f"  Best model ({best['model_name']}):  "
        f"F1 = {best['f1']:.3f} ({f1_diff:+.3f}),  "
        f"MCC = {best['mcc']:.3f} ({mcc_diff:+.3f})"
    )
    if f1_diff >= 0:
        print(f"  OK Meets or exceeds the published F1 baseline.")
    else:
        print(f"  FAIL Does not meet the published F1 baseline.")
        print(
            "    This is an honest result. Possible reasons: different feature set, "
            "slightly different preprocessing, or sampling variability on n=55."
        )


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Convert list of result dicts into a tidy DataFrame for reporting/saving.

    Returns one row per model with summary statistics.
    """
    rows = []
    for r in results:
        f1_lo, f1_hi = r["f1_ci"]
        mcc_lo, mcc_hi = r["mcc_ci"]
        rows.append({
            "model": r["model_name"],
            "f1": r["f1"],
            "f1_ci_lo": f1_lo,
            "f1_ci_hi": f1_hi,
            "mcc": r["mcc"],
            "mcc_ci_lo": mcc_lo,
            "mcc_ci_hi": mcc_hi,
            "accuracy": r["accuracy"],
            "roc_auc": r["roc_auc"],
        })
    return pd.DataFrame(rows)
