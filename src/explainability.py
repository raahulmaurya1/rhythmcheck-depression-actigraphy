"""
explainability.py — RhythmCheck
=================================
SHAP-based feature importance for the best-performing model.

Dataset citation:
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

Design notes (per spec Section 8, Step 6)
------------------------------------------
The model is FIT ON THE FULL DATASET for this step only — this is correct
and intentional. SHAP explains what the model learned from all available
data; it is completely separate from the LOSO-CV evaluation loop (which
measures generalisation performance). Conflating the two would be a mistake.

SHAP explainer selection:
- LogisticRegression → shap.LinearExplainer (exact, fast)
- MLP               → shap.KernelExplainer  (model-agnostic, slower but
                       reliable; uses a background summary of training data)

Outputs saved to outputs/figures/:
- shap_summary_bar.png  : Mean |SHAP| per feature (bar chart)
- shap_beeswarm.png     : Beeswarm / dot plot showing direction of impact

Ethical framing (spec requirement):
    SHAP values are used to ask a genuine ethical question: are the features
    driving the model's predictions behaviourally meaningful and consistent
    with established clinical knowledge (e.g. IS/IV as markers of circadian
    disruption in depression), or are they spurious? This framing is preserved
    in outputs/report.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.models import BaseModel, LogisticRegressionModel, MLPModel
from src.evaluate import FEATURE_COLS


# -- Friendly feature labels for plots ----------------------------------------
FEATURE_LABELS = {
    "mean_daily_activity": "Mean daily activity",
    "std_daily_activity":  "Std daily activity",
    "mean_minute_activity":"Mean minute activity",
    "iv":                  "Intradaily Variability (IV)",
    "is_":                 "Interdaily Stability (IS)",
    "m10":                 "M10 (most active 10h)",
    "l5":                  "L5 (least active 5h)",
    "ra":                  "Relative Amplitude (RA)",
    "m10_onset":           "M10 onset (hour)",
    "l5_onset":            "L5 onset (hour)",
    "day_night_ratio":     "Day/night activity ratio",
    "sleep_efficiency":    "Sleep efficiency",
}


def _prepare_data(
    feature_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[np.ndarray, np.ndarray, SimpleImputer, StandardScaler]:
    """
    Impute and scale the full feature matrix.

    Returns
    -------
    X_scaled   : Transformed feature array, shape (n_subjects, n_features).
    y          : Label array, shape (n_subjects,).
    imputer    : Fitted SimpleImputer (median strategy).
    scaler     : Fitted StandardScaler.
    """
    X_raw = feature_df[feature_cols].values.astype(float)
    y = feature_df["label"].values.astype(int)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return X_scaled, y, imputer, scaler


def _build_explainer(
    model: BaseModel,
    X_train: np.ndarray,
) -> shap.Explainer:
    """
    Return the appropriate SHAP explainer for the given model type.

    - LogisticRegressionModel → shap.LinearExplainer (exact, fast)
    - MLPModel                → shap.KernelExplainer  (model-agnostic)
    """
    if isinstance(model, LogisticRegressionModel):
        return shap.LinearExplainer(
            model.sklearn_model,
            X_train,
        )
    elif isinstance(model, MLPModel):
        # KernelExplainer needs a summarised background dataset
        # Use k-means summary to keep it tractable on n=55
        background = shap.kmeans(X_train, min(10, len(X_train)))
        return shap.KernelExplainer(
            model.sklearn_model.predict_proba,
            background,
        )
    else:
        raise TypeError(
            f"[explainability] No SHAP explainer configured for model type: "
            f"{type(model).__name__}"
        )


def run_shap(
    feature_df: pd.DataFrame,
    model: BaseModel,
    feature_cols: list[str] = FEATURE_COLS,
    output_dir: str | Path = "outputs/figures",
    show_plots: bool = False,
) -> dict:
    """
    Fit *model* on the full dataset, compute SHAP values, save plots.

    Per spec Step 6: fitting on the full dataset here is intentional and
    correct — this is for interpretation, not evaluation.

    Parameters
    ----------
    feature_df  : Output of feature_extraction.build_feature_matrix().
    model       : A fitted or unfitted BaseModel instance.
    feature_cols: Feature column names.
    output_dir  : Directory to save plot PNGs.
    show_plots  : If True, call plt.show() (use only in interactive sessions).

    Returns
    -------
    dict with keys:
        shap_values     : np.ndarray shape (n_subjects, n_features)
        mean_abs_shap   : pd.Series — mean |SHAP| per feature, sorted desc.
        feature_ranking : list[str] — feature names sorted by importance.
        bar_plot_path   : Path to saved bar chart.
        beeswarm_path   : Path to saved beeswarm plot.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[explainability] Fitting {model.name} on full dataset (n={len(feature_df)}) ...")
    X_scaled, y, imputer, scaler = _prepare_data(feature_df, feature_cols)
    model.fit(X_scaled, y)

    print(f"[explainability] Building SHAP explainer for {model.name} ...")
    explainer = _build_explainer(model, X_scaled)

    print("[explainability] Computing SHAP values ...")
    shap_output = explainer(X_scaled) if isinstance(model, LogisticRegressionModel) \
                  else explainer.shap_values(X_scaled)

    # Normalise: extract SHAP values for the positive class (depressed=1)
    if isinstance(shap_output, list):
        # KernelExplainer returns list: [shap_class0, shap_class1]
        shap_values = np.array(shap_output[1])  # shape (n, features)
    elif isinstance(shap_output, np.ndarray) and shap_output.ndim == 3:
        # KernelExplainer with predict_proba can return (n, features, 2)
        shap_values = shap_output[:, :, 1]
    elif hasattr(shap_output, "values"):
        # Explanation object from LinearExplainer
        sv = shap_output.values
        if sv.ndim == 3:
            shap_values = sv[:, :, 1]  # positive class
        else:
            shap_values = sv
    else:
        shap_values = np.array(shap_output)

    # Human-readable labels
    labels = [FEATURE_LABELS.get(c, c) for c in feature_cols]

    # -- Mean |SHAP| ranking --------------------------------------------------
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=labels).sort_values(ascending=False)
    feature_ranking = [
        feature_cols[labels.index(lbl)] for lbl in importance.index
    ]

    print("[explainability] Feature importance ranking (mean |SHAP|):")
    for rank, (feat, val) in enumerate(importance.items(), 1):
        print(f"  {rank:>2}. {feat:<35} {val:.4f}")

    # -- Bar chart -------------------------------------------------------------
    bar_path = output_dir / "shap_summary_bar.png"
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e05c5c" if v > 0 else "#5c8ae0" for v in importance.values]
    ax.barh(importance.index[::-1], importance.values[::-1], color=colors[::-1])
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(
        f"Feature Importance  {model.name}\n"
        f"(SHAP, fit on full dataset, n={len(feature_df)})",
        fontsize=11,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(bar_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"[explainability] Saved bar chart → {bar_path}")

    # -- Beeswarm / summary plot -----------------------------------------------
    beeswarm_path = output_dir / "shap_beeswarm.png"
    fig, ax = plt.subplots(figsize=(8, 6))

    # Manual beeswarm using scatter (compatible without shap.plots.beeswarm
    # depending on shap version)
    for i, (feat_label, col) in enumerate(zip(labels, feature_cols)):
        sv_col = shap_values[:, feature_cols.index(col)]
        feat_vals = X_scaled[:, feature_cols.index(col)]
        # Normalise feature values to [0, 1] for colouring
        fv_min, fv_max = feat_vals.min(), feat_vals.max()
        if fv_max > fv_min:
            fv_norm = (feat_vals - fv_min) / (fv_max - fv_min)
        else:
            fv_norm = np.zeros_like(feat_vals)
        # Jitter y positions
        rng = np.random.default_rng(i)
        y_jitter = i + rng.uniform(-0.25, 0.25, size=len(sv_col))
        ax.scatter(sv_col, y_jitter, c=fv_norm, cmap="RdBu_r", alpha=0.7,
                   s=20, vmin=0, vmax=1)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP value (impact on P(depressed))", fontsize=10)
    ax.set_title(
        f"SHAP Beeswarm  {model.name}\n"
        f"Colour: feature value (red=high, blue=low)",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(beeswarm_path, dpi=150)
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"[explainability] Saved beeswarm → {beeswarm_path}")

    return {
        "shap_values": shap_values,
        "mean_abs_shap": importance,
        "feature_ranking": feature_ranking,
        "bar_plot_path": bar_path,
        "beeswarm_path": beeswarm_path,
    }


def select_best_model_for_shap(
    results: list[dict],
    models: list[BaseModel],
) -> BaseModel:
    """
    Pick the model with the highest LOSO-CV F1 score for SHAP explanation.
    Returns a fresh instance of that model type (unfitted — run_shap fits it).

    Parameters
    ----------
    results : List of result dicts from evaluate.evaluate_all_models().
    models  : Corresponding list of BaseModel instances (same order as results).

    Returns
    -------
    Fresh unfitted BaseModel instance of the winning type.
    """
    best_idx = int(np.argmax([r["f1"] for r in results]))
    best_result = results[best_idx]
    best_model = models[best_idx]
    print(
        f"[explainability] Best model by F1: {best_result['model_name']} "
        f"(F1={best_result['f1']:.3f}) — using this for SHAP."
    )
    # Return a fresh instance so SHAP fits on full data cleanly
    if isinstance(best_model, LogisticRegressionModel):
        return LogisticRegressionModel(C=best_model.C)
    elif isinstance(best_model, MLPModel):
        return MLPModel(
            hidden_layer_sizes=best_model.hidden_layer_sizes,
            alpha=best_model.alpha,
        )
    else:
        raise TypeError(f"Unknown model type: {type(best_model)}")
