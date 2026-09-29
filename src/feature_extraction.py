"""
feature_extraction.py — RhythmCheck
=====================================
Extract circadian and rest-activity rhythm features from minute-level
actigraphy data, one row per subject.

Dataset citation:
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

Features implemented
--------------------
Non-parametric circadian rhythm analysis (Van Someren et al., 1997):
  - IS  : Interdaily Stability  — how consistent the 24-h rhythm is day-to-day
  - IV  : Intradaily Variability — fragmentation / number of rest-activity transitions
  - RA  : Relative Amplitude    — (M10 - L5) / (M10 + L5)
  - M10 : Mean activity in the 10 most active consecutive hours
  - L5  : Mean activity in the 5  least active consecutive hours
  - M10_onset: Clock hour at which M10 window starts
  - L5_onset : Clock hour at which L5 window starts

Summary activity features:
  - mean_daily_activity   : Mean total activity per 24-h day
  - std_daily_activity    : Std of total daily activity across days
  - mean_minute_activity  : Mean per-minute activity count
  - day_night_ratio       : Mean daytime (08:00-20:00) / nighttime activity
  - sleep_efficiency      : Fraction of nighttime minutes with near-zero activity (<10)

All features are defined in data/processed/feature_dictionary.md.

No data leakage note:
    Features are derived purely from the activity time series of each subject.
    No cross-subject information is used. Scaling/normalisation is intentionally
    left to evaluate.py, which fits scalers per CV fold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# -- Constants -----------------------------------------------------------------
NIGHT_START_HOUR = 20   # 20:00 — start of nighttime window
NIGHT_END_HOUR = 8      # 08:00 — end   of nighttime window (exclusive)
SLEEP_THRESHOLD = 10    # activity counts below this treated as "sleep/rest"


# -- Core feature functions ----------------------------------------------------

def _to_hourly(minute_series: pd.Series, timestamps: pd.Series) -> pd.Series:
    """
    Resample minute-level activity to hourly means.

    Parameters
    ----------
    minute_series : Activity counts, one value per minute.
    timestamps    : Corresponding datetime index.

    Returns
    -------
    pd.Series indexed by datetime (hourly), values = mean activity per hour.
    """
    df = pd.DataFrame({"activity": minute_series.values}, index=pd.to_datetime(timestamps.values))
    hourly = df["activity"].resample("h").mean().dropna()
    return hourly


def compute_iv(hourly: pd.Series) -> float:
    """
    Intradaily Variability (IV).

    Measures rhythm fragmentation — how often and how abruptly the
    rest-activity cycle switches direction. Higher IV = more fragmented.

    Formula (Van Someren et al., 1997):
        IV = n * Σ(x_{i+1} - x_i)^2 / ((n-1) * Σ(x_i - x̄)^2)

    Returns NaN if variance is zero or fewer than 3 hourly epochs.
    """
    x = hourly.values.astype(float)
    n = len(x)
    if n < 3:
        return np.nan
    numerator = n * np.sum(np.diff(x) ** 2)
    denominator = (n - 1) * np.sum((x - x.mean()) ** 2)
    if denominator == 0:
        return np.nan
    return float(numerator / denominator)


def compute_is(hourly: pd.Series) -> float:
    """
    Interdaily Stability (IS).

    Measures how similar the 24-h activity pattern is from day to day.
    Higher IS = more regular/stable circadian rhythm.

    Formula (Van Someren et al., 1997):
        IS = (n/p) * Σ_p(x̄_h - x̄)^2 / Σ_n(x_i - x̄)^2

    where p=24 (hours per day), x̄_h = mean activity for hour h across days.

    Returns NaN if fewer than 2 full days of data or zero overall variance.
    """
    x = hourly.copy()
    if len(x) < 2 * 24:
        return np.nan

    n = len(x)
    p = 24  # epochs per day (hourly)
    x_bar = x.mean()

    # Mean activity for each hour-of-day across all days
    hourly_profile = x.groupby(x.index.hour).mean()
    x_bar_h = hourly_profile.reindex(x.index.hour.values).values

    numerator = (n / p) * np.sum((x_bar_h - x_bar) ** 2)
    denominator = np.sum((x.values - x_bar) ** 2)
    if denominator == 0:
        return np.nan
    return float(np.clip(numerator / denominator, 0, 1))


def compute_m10_l5(hourly: pd.Series) -> dict:
    """
    M10 and L5 — most/least active windows.

    M10 : Mean activity in the 10 consecutive hours with the highest mean.
    L5  : Mean activity in the 5  consecutive hours with the lowest mean.
    RA  : Relative Amplitude = (M10 - L5) / (M10 + L5)

    Uses a circular sliding window so the window can wrap across midnight.

    Returns
    -------
    dict with keys: m10, l5, ra, m10_onset, l5_onset
    All NaN if fewer than 24 hours of data.
    """
    null = {"m10": np.nan, "l5": np.nan, "ra": np.nan, "m10_onset": np.nan, "l5_onset": np.nan}
    if len(hourly) < 24:
        return null

    # Build 24-value average hourly profile (wraps midnight cleanly)
    profile = hourly.groupby(hourly.index.hour).mean()
    profile = profile.reindex(range(24)).interpolate(method="linear", fill_value="extrapolate")
    vals = profile.values  # shape (24,)

    # Rolling window sums (circular: tile to handle wrap-around)
    tiled = np.concatenate([vals, vals])

    # M10 — 10-hour window
    m10_sums = np.array([tiled[i : i + 10].mean() for i in range(24)])
    m10_idx = int(np.argmax(m10_sums))
    m10_val = float(m10_sums[m10_idx])

    # L5 — 5-hour window
    l5_sums = np.array([tiled[i : i + 5].mean() for i in range(24)])
    l5_idx = int(np.argmin(l5_sums))
    l5_val = float(l5_sums[l5_idx])

    denom = m10_val + l5_val
    ra = float((m10_val - l5_val) / denom) if denom > 0 else np.nan

    return {
        "m10": m10_val,
        "l5": l5_val,
        "ra": ra,
        "m10_onset": float(m10_idx),
        "l5_onset": float(l5_idx),
    }


def compute_day_night_ratio(minute_series: pd.Series, timestamps: pd.Series) -> float:
    """
    Day-vs-night activity ratio.

    Daytime  = 08:00 – 20:00 (12 hours).
    Nighttime = 20:00 – 08:00 (12 hours).

    Returns mean daytime activity / mean nighttime activity.
    Returns NaN if nighttime mean is zero.
    """
    df = pd.DataFrame({"activity": minute_series.values}, index=pd.to_datetime(timestamps.values))
    hour = df.index.hour
    daytime = df["activity"][(hour >= 8) & (hour < 20)].mean()
    nighttime = df["activity"][(hour >= 20) | (hour < 8)].mean()
    if nighttime == 0 or np.isnan(nighttime):
        return np.nan
    return float(daytime / nighttime)


def compute_sleep_efficiency(minute_series: pd.Series, timestamps: pd.Series) -> float:
    """
    Sleep efficiency proxy.

    Fraction of nighttime minutes (20:00-08:00) where activity < SLEEP_THRESHOLD.
    Higher value means more consolidated sleep.
    """
    df = pd.DataFrame({"activity": minute_series.values}, index=pd.to_datetime(timestamps.values))
    hour = df.index.hour
    night = df["activity"][(hour >= 20) | (hour < 8)]
    if len(night) == 0:
        return np.nan
    return float((night < SLEEP_THRESHOLD).mean())


# -- Per-subject feature extraction --------------------------------------------

def extract_subject_features(
    subject_id: str,
    activity_series: pd.Series,
    timestamps: pd.Series,
) -> dict:
    """
    Extract all features for a single subject.

    Parameters
    ----------
    subject_id     : e.g. 'condition_7'
    activity_series: Per-minute activity counts.
    timestamps     : Corresponding datetime values.

    Returns
    -------
    dict with subject_id + all feature columns.
    """
    hourly = _to_hourly(activity_series, timestamps)

    # Daily totals for summary stats
    df = pd.DataFrame({"activity": activity_series.values}, index=pd.to_datetime(timestamps.values))
    daily_totals = df["activity"].resample("D").sum()

    m10_l5 = compute_m10_l5(hourly)

    features = {
        "subject_id": subject_id,
        # Summary activity
        "mean_daily_activity": float(daily_totals.mean()),
        "std_daily_activity": float(daily_totals.std(ddof=1)) if len(daily_totals) > 1 else np.nan,
        "mean_minute_activity": float(activity_series.mean()),
        # Circadian rhythm (non-parametric)
        "iv": compute_iv(hourly),
        "is_": compute_is(hourly),
        **m10_l5,
        # Day/night behaviour
        "day_night_ratio": compute_day_night_ratio(activity_series, timestamps),
        "sleep_efficiency": compute_sleep_efficiency(activity_series, timestamps),
    }
    return features


# -- Build full feature matrix -------------------------------------------------

def build_feature_matrix(
    activity_df: pd.DataFrame,
    output_dir: str | Path = "data/processed",
    save: bool = True,
) -> pd.DataFrame:
    """
    Compute features for every subject in *activity_df* and return a
    feature matrix with one row per subject.

    Optionally writes:
    - data/processed/features.csv
    - data/processed/feature_dictionary.md

    Parameters
    ----------
    activity_df : Long-format DataFrame from data_loader.load_dataset().
                  Must have columns: subject_id, timestamp, activity, label.
    output_dir  : Directory to write features.csv and feature_dictionary.md.
    save        : If True, write files to output_dir.

    Returns
    -------
    DataFrame with one row per subject. Columns: subject_id, label, <features>.
    """
    output_dir = Path(output_dir)

    subject_ids = activity_df["subject_id"].unique()
    rows = []

    print(f"[feature_extraction] Extracting features for {len(subject_ids)} subjects...")

    for sid in sorted(subject_ids):
        sub = activity_df[activity_df["subject_id"] == sid]
        feats = extract_subject_features(
            subject_id=sid,
            activity_series=sub["activity"],
            timestamps=sub["timestamp"],
        )
        # Carry through label (0/1) and group
        feats["label"] = int(sub["label"].iloc[0])
        feats["group"] = sub["group"].iloc[0]
        rows.append(feats)

    feature_df = pd.DataFrame(rows)

    # Reorder columns: identifiers first, then features
    id_cols = ["subject_id", "group", "label"]
    feat_cols = [c for c in feature_df.columns if c not in id_cols]
    feature_df = feature_df[id_cols + feat_cols].reset_index(drop=True)

    print(f"[feature_extraction] Feature matrix shape: {feature_df.shape}")
    nan_summary = feature_df[feat_cols].isna().sum()
    if nan_summary.any():
        print(f"[feature_extraction] NaN counts per feature:\n{nan_summary[nan_summary > 0]}")

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "features.csv"
        feature_df.to_csv(csv_path, index=False)
        print(f"[feature_extraction] Saved features to {csv_path}")
        _write_feature_dictionary(output_dir)

    return feature_df


# -- Feature dictionary --------------------------------------------------------

def _write_feature_dictionary(output_dir: Path) -> None:
    """Write feature_dictionary.md explaining each column in plain language."""
    md = """\
# Feature Dictionary — RhythmCheck

Generated by `src/feature_extraction.py`. One row per subject.

## Identifier columns

| Column | Description |
|---|---|
| `subject_id` | Unique subject identifier, e.g. `condition_7` or `control_3`. |
| `group` | `condition` (depressed) or `control` (healthy). |
| `label` | Binary label: `1` = depressed (condition), `0` = healthy (control). |

## Summary activity features

| Column | Description |
|---|---|
| `mean_daily_activity` | Mean total Actiwatch activity count per calendar day. Higher values indicate more overall movement. |
| `std_daily_activity` | Standard deviation of daily total activity across days. Measures day-to-day variability in overall activity level. |
| `mean_minute_activity` | Mean activity count per minute across the full recording. Equivalent to the overall activity rate. |

## Non-parametric circadian rhythm features

Based on the methods of Van Someren et al. (1997), computed from hourly means of the minute-level data.

| Column | Description |
|---|---|
| `iv` | **Intradaily Variability (IV)**: Measures how fragmented the rest-activity rhythm is. Higher IV = more frequent, abrupt transitions between rest and activity (e.g. disrupted sleep, irregular schedule). Typical range: 0–3. |
| `is_` | **Interdaily Stability (IS)**: Measures how consistent the 24-hour activity pattern is from day to day. Higher IS = more regular rhythm. Range: 0–1. |
| `m10` | **M10**: Mean activity during the 10 consecutive hours with the highest mean activity (the most active period of the day). Reflects daytime activity level. |
| `l5` | **L5**: Mean activity during the 5 consecutive hours with the lowest mean activity (the least active / sleep period). Reflects night-time activity level. |
| `ra` | **Relative Amplitude (RA)**: `(M10 - L5) / (M10 + L5)`. Captures the contrast between the most and least active periods. Higher RA = clearer day/night difference. Range: 0–1. |
| `m10_onset` | Clock hour (0–23) at which the M10 window starts. Reflects the timing of peak activity. |
| `l5_onset` | Clock hour (0–23) at which the L5 window starts. Reflects the timing of least activity (approximate sleep onset). |

## Day/night behaviour features

| Column | Description |
|---|---|
| `day_night_ratio` | Mean daytime activity (08:00–20:00) divided by mean nighttime activity (20:00–08:00). Higher values indicate more activity concentrated in the daytime. |
| `sleep_efficiency` | Fraction of nighttime minutes (20:00–08:00) where activity count < 10 (proxy for "asleep"). Higher values indicate more consolidated, less fragmented sleep. |

## Notes

- Features are computed per-subject from the raw minute-level activity time series.
- No cross-subject information is used in feature computation.
- Scaling/normalisation is applied per CV fold in `src/evaluate.py`, not here.
- The condition group is **heterogeneous**: it includes bipolar I, bipolar II, and
  unipolar depression patients (see `afftype` column in scores.csv). This limits
  the interpretability of any feature-level comparison between groups.
- This dataset is from a specific clinical population in Norway (Garcia-Ceja et al.,
  2018). Features may not generalise to other populations.
"""
    dict_path = output_dir / "feature_dictionary.md"
    dict_path.write_text(md, encoding="utf-8")
    print(f"[feature_extraction] Saved feature dictionary to {dict_path}")
