# RhythmCheck — Depression-Risk Detection from Wearable Actigraphy

Automated circadian-rhythm analysis pipeline that detects depression-risk signals
from wrist-worn actigraphy recordings.

Built on the **Depresjon** dataset (Garcia-Ceja et al., 2018) — 55 subjects,
1.57 million activity records, 5–20 days per subject.

> **Research and educational use only.**
> This project must not be used for clinical diagnosis or treatment decisions.

---

## Dataset

**Depresjon** — A Motor Activity Database of Depression Episodes in Unipolar and Bipolar Patients
Garcia-Ceja, E., Riegler, M., Jakobsen, P., Torresen, J., Nordgreen, T., Oedegaard, K., Fasmer, O. (2018).
*Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).*

- 23 depressed subjects (condition group: bipolar I/II + unipolar depression)
- 32 healthy controls
- Wrist actigraphy at 1-minute intervals, collected in a Norwegian clinical setting

Download: https://datasets.simula.no/depresjon/

Place the downloaded data under `data/raw/`:
```
data/raw/
    condition/       (23 CSV files, one per patient)
    control/         (32 CSV files)
    scores.csv       (clinical metadata)
```

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/raahulmaurya1/rhythmcheck-depression-actigraphy.git
cd rhythmcheck-depression-actigraphy

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python -m src.pipeline
```

This runs everything end-to-end (~3-5 minutes):

1. **Load** — validates and joins the 55 subject files
2. **Extract** — computes 12 circadian features per subject
3. **Evaluate** — LOSO-CV with F1/MCC + 95% bootstrap CIs for both models
4. **Explain** — SHAP feature importance for the best model
5. **Report** — writes `outputs/report.md`

**Output files produced:**

| File | What it contains |
|---|---|
| `data/processed/features.csv` | 55 × 12 feature matrix |
| `data/processed/feature_dictionary.md` | Definition of every feature |
| `outputs/figures/shap_summary_bar.png` | Feature importance bar chart |
| `outputs/figures/shap_beeswarm.png` | SHAP beeswarm plot |
| `outputs/report.md` | Full methodology + results + limitations |

### Skip feature extraction on re-runs

```bash
python -m src.pipeline --skip-features
```

Loads cached `features.csv` instead of re-extracting — much faster for
re-running just the evaluation or SHAP step.

### Run the automated test suite

```bash
python -m pytest tests/ -v
```

Expected: **115 tests, all passing.**

---

## Results (LOSO-CV, n=55)

Results from running `python -m src.pipeline` on the real dataset:

| Model | F1 (macro) | 95% CI | MCC | 95% CI | Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression (L2, C=1.0) | **0.667** | [0.542 – 0.799] | **0.336** | [0.085 – 0.606] | 0.673 | **0.822** |
| MLP (32 units, early stopping) | 0.427 | [0.309 – 0.542] | -0.076 | [-0.324 – 0.175] | 0.527 | 0.437 |

**Best model:** Logistic Regression (by F1 and all other metrics).

**Published baseline (Garcia-Ceja et al., 2018):** F1 = 0.73, MCC = 0.44

This work achieves F1 = 0.667 (−0.063 vs baseline). The gap is reported honestly.
The wide CIs reflect the small sample (n=55); results should not be over-interpreted.

> **Note on ROC-AUC:** AUC = 0.822 indicates that the model has strong
> *ranking* ability (it tends to assign higher probabilities to truly depressed
> subjects), even when F1 is moderate. The F1/MCC gap vs. the baseline
> is likely due to the feature set used rather than model failure.

---

## Top Features (SHAP, full-dataset fit)

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Sleep efficiency | 0.988 |
| 2 | Mean daily activity | 0.618 |
| 3 | Std daily activity | 0.572 |
| 4 | Day/night activity ratio | 0.551 |
| 5 | L5 onset (hour) | 0.510 |

Sleep efficiency and overall activity levels dominate. Classical circadian
markers (IS, IV) contributed less than expected — a finding worth investigating
with a larger dataset or richer feature engineering.

---

## Project Structure

```
rhythmcheck-depression-actigraphy/
    src/
        data_loader.py          # Load + validate Depresjon dataset
        feature_extraction.py   # 12 circadian features per subject
        models.py               # Logistic Regression + MLP (shared interface)
        evaluate.py             # LOSO-CV, F1/MCC, bootstrap CIs
        explainability.py       # SHAP feature importance
        pipeline.py             # Main entry point (python -m src.pipeline)
    tests/
        test_data_loader.py         # 16 tests
        test_feature_extraction.py  # 31 tests
        test_models.py              # 29 tests
        test_evaluate.py            # 24 tests
        test_explainability.py      # 15 tests
    data/
        raw/                    # Raw dataset (not tracked in git)
        processed/              # Generated: features.csv, feature_dictionary.md
    outputs/
        figures/                # Generated: SHAP plots
        report.md               # Generated: full results report
    requirements.txt
    README.md
```

---

## Methodology

- **Evaluation:** Leave-one-subject-out cross-validation (LOSO-CV, 55 folds)
  — matches the dataset's own published methodology
- **No-leakage:** `StandardScaler` and `SimpleImputer` are fitted on the 54
  training subjects in each fold and applied to the held-out subject only
- **Primary metrics:** F1 (macro) and MCC — chosen because the 23/32
  class imbalance makes accuracy misleading
- **Uncertainty:** 95% bootstrap confidence intervals (500 resamples) on
  aggregated LOSO predictions
- **Explainability:** SHAP `LinearExplainer` (Logistic Regression) or
  `KernelExplainer` (MLP), fitted on full dataset for interpretation only

---

## Limitations

1. **Small sample (n=55)** — wide CIs, results not stable across subsets
2. **Heterogeneous condition group** — bipolar I/II + unipolar depression mixed
3. **Single population** — Norwegian clinical cohort, may not generalise
4. **Wrist actigraphy only** — behavioural proxy, not a direct clinical measure
5. **No diagnostic claims** — research demonstration only

See `outputs/report.md` for a full limitations discussion.

---

## Ethical Statement

This pipeline is a **research tool** built to explore circadian rhythm analysis
as a signal for depression risk. It must not be used:

- As a diagnostic or screening tool for patients
- In any clinical or therapeutic decision-making context
- In any commercial application without written permission from the dataset authors

The dataset is used under its academic research licence (Garcia-Ceja et al., 2018).

---

## AI-Assisted Development

Code was written with the assistance of **Antigravity AI** (Google DeepMind).
All logic, design decisions, and scientific choices were made and verified by the author.

---

## Author

Rahul Maurya
GitHub: [@raahulmaurya1](https://github.com/raahulmaurya1)

---

## License

MIT License — see `LICENSE` for details.
Dataset licence: research/educational use only (Garcia-Ceja et al., 2018).
Commercial use of the dataset requires written permission from the dataset authors.
