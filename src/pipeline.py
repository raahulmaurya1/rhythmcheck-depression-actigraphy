"""
pipeline.py — RhythmCheck
==========================
Single CLI entry point. Runs the full pipeline end-to-end:

    python -m src.pipeline

Steps executed (in order):
    1. Load dataset from data/raw/
    2. Extract features → data/processed/features.csv
    3. Evaluate both models with LOSO-CV → print F1/MCC/CI/baseline comparison
    4. Run SHAP on the best model → outputs/figures/*.png
    5. Write outputs/report.md

Dataset citation (MANDATORY):
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

License: Research/educational use only. Commercial use forbidden.

Ethical guardrails (enforced here):
    - No diagnostic or clinical-deployment claims are made anywhere.
    - Results are printed and saved as-is, without cherry-picking.
    - Limitations section is always written to the report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# -- Paths ---------------------------------------------------------------------
RAW_DIR         = Path("data/raw")
PROCESSED_DIR   = Path("data/processed")
FIGURES_DIR     = Path("outputs/figures")
REPORT_PATH     = Path("outputs/report.md")


def _banner(msg: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {msg}")
    print(f"{'=' * 60}")


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    figures_dir: Path = FIGURES_DIR,
    report_path: Path = REPORT_PATH,
    n_bootstrap: int = 1000,
    skip_features: bool = False,
) -> None:
    """
    Execute the full RhythmCheck pipeline.

    Parameters
    ----------
    raw_dir       : Location of condition/, control/, scores.csv.
    processed_dir : Where to write features.csv and feature_dictionary.md.
    figures_dir   : Where to write SHAP plots.
    report_path   : Where to write outputs/report.md.
    n_bootstrap   : Bootstrap resamples for CI estimation (default 1000).
    skip_features : If True, load features.csv from processed_dir instead of
                    re-extracting (speeds up re-runs when data hasn't changed).
    """
    t0 = time.time()

    # -- Step 1: Load ---------------------------------------------------------
    _banner("Step 1 — Loading dataset")
    from src.data_loader import load_dataset
    activity_df, subject_df = load_dataset(raw_dir)

    # -- Step 2: Feature extraction --------------------------------------------
    _banner("Step 2 — Feature extraction")
    import pandas as pd
    from src.feature_extraction import build_feature_matrix
    from src.evaluate import FEATURE_COLS

    features_csv = processed_dir / "features.csv"
    if skip_features and features_csv.exists():
        print(f"[pipeline] Loading cached features from {features_csv}")
        feature_df = pd.read_csv(features_csv)
    else:
        feature_df = build_feature_matrix(
            activity_df,
            output_dir=processed_dir,
            save=True,
        )

    print(f"\n[pipeline] Feature matrix: {feature_df.shape[0]} subjects × "
          f"{len(FEATURE_COLS)} features")

    # -- Step 3: Evaluate ------------------------------------------------------
    _banner("Step 3 — LOSO-CV Evaluation (both models)")
    from src.evaluate import evaluate_all_models, results_to_dataframe
    from src.models import get_models

    results = evaluate_all_models(
        feature_df,
        n_bootstrap=n_bootstrap,
        verbose=True,
    )
    results_df = results_to_dataframe(results)

    # -- Step 4: SHAP explainability -------------------------------------------
    _banner("Step 4 — SHAP Explainability")
    from src.explainability import run_shap, select_best_model_for_shap

    models = get_models()
    best_model = select_best_model_for_shap(results, models)
    shap_result = run_shap(
        feature_df,
        best_model,
        output_dir=figures_dir,
        show_plots=False,
    )

    # -- Step 5: Write report --------------------------------------------------
    _banner("Step 5 — Writing report")
    _write_report(
        report_path=report_path,
        subject_df=subject_df,
        feature_df=feature_df,
        results=results,
        results_df=results_df,
        shap_result=shap_result,
        best_model_name=best_model.name,
    )

    elapsed = time.time() - t0
    _banner(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  features.csv       → {features_csv}")
    print(f"  SHAP bar chart     → {shap_result['bar_plot_path']}")
    print(f"  SHAP beeswarm      → {shap_result['beeswarm_path']}")
    print(f"  Report             → {report_path}")


# -- Report writer -------------------------------------------------------------

def _write_report(
    report_path: Path,
    subject_df,
    feature_df,
    results: list[dict],
    results_df,
    shap_result: dict,
    best_model_name: str,
) -> None:
    """Write outputs/report.md (Step 9 of spec)."""
    import numpy as np
    from src.evaluate import PUBLISHED_F1, PUBLISHED_MCC

    report_path.parent.mkdir(parents=True, exist_ok=True)

    best = max(results, key=lambda r: r["f1"])
    f1_diff  = best["f1"]  - PUBLISHED_F1
    mcc_diff = best["mcc"] - PUBLISHED_MCC
    f1_lo, f1_hi   = best["f1_ci"]
    mcc_lo, mcc_hi = best["mcc_ci"]

    # SHAP top features
    top_features = list(shap_result["mean_abs_shap"].head(5).items())

    # Afftype breakdown
    cond = subject_df[subject_df["group"] == "condition"]
    afftype_map = {"1": "Bipolar II", "2": "Unipolar", "3": "Bipolar I"}
    afftype_lines = []
    for code, label in afftype_map.items():
        n = (cond["afftype"].astype(str).str.strip() == code).sum()
        afftype_lines.append(f"  - {label} (afftype={code}): {n} subjects")

    md = f"""\
# RhythmCheck — Depression-Risk Detection from Wearable Actigraphy
## Results Report

> **Dataset citation (MANDATORY):**
> Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
> Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
> Depression Episodes in Unipolar and Bipolar Patients.
> *Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).*
>
> **License:** Research and educational use only. Commercial use forbidden
> without written permission from the authors.

---

## 1. Methodology

### Dataset
- **Source:** Depresjon dataset (Garcia-Ceja et al., 2018)
  — https://datasets.simula.no/depresjon/
- **Subjects:** {len(subject_df)} total —
  {int(subject_df['label'].sum())} depressed (condition group),
  {int((subject_df['label']==0).sum())} healthy controls
- **Recording:** Wrist-worn Actiwatch at 1-minute intervals, 5–20 days per subject

### Features
Per-subject actigraphy features extracted from raw minute-level data:

| Feature | Description |
|---|---|
| `iv` | Intradaily Variability — rhythm fragmentation |
| `is_` | Interdaily Stability — day-to-day consistency |
| `m10` | Mean activity in the 10 most active hours |
| `l5` | Mean activity in the 5 least active hours |
| `ra` | Relative Amplitude — contrast between active/rest periods |
| `m10_onset` | Clock hour of peak activity window |
| `l5_onset` | Clock hour of least active window |
| `mean_daily_activity` | Mean total activity per calendar day |
| `std_daily_activity` | Std of daily activity across days |
| `mean_minute_activity` | Mean per-minute activity count |
| `day_night_ratio` | Daytime (08:00–20:00) / nighttime activity |
| `sleep_efficiency` | Fraction of nighttime minutes with activity < 10 |

See `data/processed/feature_dictionary.md` for full definitions.

### Evaluation
- **Cross-validation:** Leave-one-subject-out (LOSO-CV), 55 folds
  — matches the dataset's own published methodology
- **No-leakage guarantee:** StandardScaler + median imputer are fitted
  on the 54 training subjects in each fold and applied to the held-out
  subject. No global scaling before splitting.
- **Primary metrics:** F1 (macro) and MCC — chosen because the dataset is
  class-imbalanced (23 depressed vs 32 healthy) and raw accuracy is misleading
- **Uncertainty:** 95% bootstrap confidence intervals (1000 resamples) on
  aggregated LOSO predictions
- **Models compared:**
  - Regularised Logistic Regression (L2, C=1.0) — primary
  - Small MLP (1 hidden layer, 32 units, L2 α=0.01, early stopping) — comparison

---

## 2. LOSO-CV Results

### Model comparison

| Model | F1 (macro) | 95% CI | MCC | 95% CI | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
"""

    for r in results:
        f1_l, f1_h = r["f1_ci"]
        m_l, m_h   = r["mcc_ci"]
        md += (
            f"| {r['model_name']} "
            f"| {r['f1']:.3f} "
            f"| [{f1_l:.3f} – {f1_h:.3f}] "
            f"| {r['mcc']:.3f} "
            f"| [{m_l:.3f} – {m_h:.3f}] "
            f"| {r['accuracy']:.3f} "
            f"| {r['roc_auc']:.3f} |\n"
        )

    md += f"""
**Best model by F1:** {best_model_name}

### Comparison to published baseline (Garcia-Ceja et al., 2018)

| Metric | Published baseline | This work (best model) | Difference |
|---|---|---|---|
| F1 (macro) | {PUBLISHED_F1:.2f} | {best['f1']:.3f} | {f1_diff:+.3f} |
| MCC | {PUBLISHED_MCC:.2f} | {best['mcc']:.3f} | {mcc_diff:+.3f} |

"""
    if f1_diff >= 0:
        md += "> OK This work meets or exceeds the published F1 baseline.\n\n"
    else:
        md += (
            "> FAIL This work does not meet the published F1 baseline. "
            "This is reported honestly. Possible reasons: different feature set, "
            "slightly different preprocessing, or sampling variability on n=55.\n\n"
        )

    md += f"""\
---

## 3. SHAP Feature Importance

SHAP values were computed on **{best_model_name}** fitted on the full dataset
(n={len(feature_df)}). This is separate from the LOSO-CV evaluation — the
purpose here is to understand *what the model learned*, not to estimate
generalisation.

### Top 5 features by mean |SHAP value|

| Rank | Feature | Mean |SHAP| |
|---|---|---|
"""
    for i, (feat, val) in enumerate(top_features, 1):
        md += f"| {i} | {feat} | {val:.4f} |\n"

    md += f"""
**Ethical interpretation:** The features driving predictions should be
scrutinised against clinical knowledge. IS and IV are established markers of
circadian disruption in depression (Van Someren et al., 1997); if these rank
highly in SHAP, the model's reasoning is behaviourally plausible. If
unexpected features dominate (e.g. raw activity means), this warrants caution
before drawing any clinical conclusions.

See SHAP plots in `outputs/figures/`:
- `shap_summary_bar.png` — feature importance bar chart
- `shap_beeswarm.png` — direction of feature impact on predictions

---

## 4. Limitations

1. **Small sample size (n=55).** Results have high variance. Confidence
   intervals are wide and should be interpreted accordingly. No single metric
   should be treated as a stable estimate.

2. **Condition-group heterogeneity.** The "condition" group is not homogeneous —
   it includes patients with three distinct diagnoses:
{chr(10).join(afftype_lines)}
   A model trained on this mixed group learns to separate "any depressive
   condition" from healthy controls, not to distinguish between subtypes.
   Differences in circadian rhythms between bipolar I, bipolar II, and
   unipolar depression may be clinically relevant but are not modelled here.

3. **Single-population generalisation.** This dataset was collected from a
   specific clinical population in Norway. Features and model coefficients
   derived here may not transfer to other populations, age groups, or
   recording devices.

4. **Wrist actigraphy only.** Motor activity is a behavioural proxy for sleep
   and circadian rhythm, not a direct neural or physiological measure.
   It does not capture mood, cognition, medication effects, or social context.

5. **No diagnostic claims.** This pipeline is a research demonstration only.
   It must not be used, directly or indirectly, for clinical diagnosis, triage,
   or treatment decisions.

6. **Feature engineering choices.** The 12 features implemented here are a
   subset of possible circadian measures. Alternative feature sets (e.g.
   cosinor parameters, wavelet features) could yield different results.

---

## 5. Next Steps

This project deliberately avoids deep learning to complement (not duplicate)
the author's existing EEG/neural-network work. Possible extensions that would
add clinical grounding:

- **Multimodal data**: Combining actigraphy with self-report scales (PHQ-9,
  MADRS) or ecological momentary assessment could improve both signal and
  interpretability.
- **Longitudinal modelling**: Tracking changes in IS/IV over time (rather than
  a single recording window) could detect prodromal changes before a depressive
  episode.
- **HBB-background clinical grounding (CYBER team model)**: A partner with a
  Health/Biology/Behaviour background could contribute clinical validation,
  patient cohort design, and ethics oversight — dimensions this purely
  computational project lacks. This connects directly to the CYBER-T team
  model's interdisciplinary structure.
- **Subtype-stratified analysis**: Separating bipolar I/II from unipolar
  depression would require a larger sample but would produce more clinically
  interpretable results.

---

## 6. References

Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
Depression Episodes in Unipolar and Bipolar Patients. *Proceedings of the 9th
ACM Multimedia Systems Conference (MMSys'18).*

Van Someren, E.J.W., Lijzenga, C., Mirmiran, M., Swaab, D.F. (1997). Long-
term fitness training improves the circadian rest-activity rhythm in healthy
elderly males. *Journal of Biological Rhythms, 12*(2), 146–156.

Lundberg, S.M., Lee, S.I. (2017). A unified approach to interpreting model
predictions. *Advances in Neural Information Processing Systems, 30.*

---

*This report was generated automatically by `src/pipeline.py`.*
*AI-assisted development: code written with Antigravity AI coding assistant.*
"""

    report_path.write_text(md, encoding="utf-8")
    print(f"[pipeline] Report written → {report_path}")


# -- CLI -----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RhythmCheck: depression-risk detection from actigraphy data."
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR,
        help=f"Path to raw data directory (default: {RAW_DIR})"
    )
    parser.add_argument(
        "--skip-features", action="store_true",
        help="Load features.csv from data/processed/ instead of re-extracting."
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=1000,
        help="Bootstrap resamples for CI estimation (default: 1000)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(
        raw_dir=args.raw_dir,
        n_bootstrap=args.n_bootstrap,
        skip_features=args.skip_features,
    )
