"""
data_loader.py — RhythmCheck
============================
Load and validate the Depresjon actigraphy dataset.

Dataset citation:
    Garcia-Ceja, E., Riegler, M., Jakobsen, P., Tørresen, J., Nordgreen, T.,
    Oedegaard, K., Fasmer, O. (2018). Depresjon: A Motor Activity Database of
    Depression Episodes in Unipolar and Bipolar Patients.
    Proceedings of the 9th ACM Multimedia Systems Conference (MMSys'18).

License: Research/educational use only. Commercial use forbidden without
         written permission from the authors.

Notes:
- condition/ and control/ numbering are INDEPENDENT sequences.
  condition_7 and control_7 are different, unrelated subjects.
- The join key from filename (e.g. "condition_7") matches the 'number'
  column in scores.csv exactly (e.g. "condition_7").
- afftype column is preserved so downstream steps can flag condition-group
  heterogeneity (1=bipolar II, 2=unipolar, 3=bipolar I).
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────
EXPECTED_CONDITION_COUNT = 23
EXPECTED_CONTROL_COUNT = 32

REQUIRED_ACTIVITY_COLS = {"timestamp", "date", "activity"}
REQUIRED_SCORES_COLS = {
    "number", "days", "gender", "age", "afftype",
    "melanch", "inpatient", "edu", "marriage", "work", "madrs1", "madrs2",
}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _extract_subject_key(filepath: Path) -> str:
    """
    Extract the full key (e.g. 'condition_7') from a filename like
    'condition_7.csv'. The key is the stem without the .csv extension.
    """
    return filepath.stem  # e.g. "condition_7"


def _load_activity_csv(filepath: Path, subject_id: str, group: str) -> pd.DataFrame:
    """
    Load a single per-patient activity CSV and attach metadata columns.

    Parameters
    ----------
    filepath   : Path to the CSV file.
    subject_id : Full identifier string, e.g. 'condition_7'.
    group      : 'condition' or 'control'.

    Returns
    -------
    DataFrame with columns: timestamp, date, activity, subject_id, group, label
    """
    df = pd.read_csv(filepath, parse_dates=["timestamp", "date"])

    missing = REQUIRED_ACTIVITY_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[data_loader] {filepath.name} is missing required columns: {missing}"
        )

    df["subject_id"] = subject_id
    df["group"] = group
    df["label"] = 1 if group == "condition" else 0  # 1=depressed, 0=control
    return df


def _load_scores(raw_dir: Path) -> pd.DataFrame:
    """
    Load scores.csv and validate required columns are present.

    Parameters
    ----------
    raw_dir : Path to data/raw/

    Returns
    -------
    scores DataFrame indexed by 'number' column.
    """
    scores_path = raw_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"[data_loader] scores.csv not found at: {scores_path}\n"
            "Please ensure the Depresjon dataset is extracted to data/raw/."
        )

    scores = pd.read_csv(scores_path)

    # Strip whitespace from all string columns (scores.csv has some ' ' entries)
    for col in scores.select_dtypes(include="str").columns:
        scores[col] = scores[col].str.strip()

    missing = REQUIRED_SCORES_COLS - set(scores.columns)
    if missing:
        raise ValueError(
            f"[data_loader] scores.csv is missing required columns: {missing}"
        )

    return scores


# ── Public API ────────────────────────────────────────────────────────────────

def load_dataset(raw_dir: str | Path = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the full Depresjon dataset from *raw_dir*.

    Validates:
    - Exactly 23 condition files and 32 control files exist.
    - All required columns are present in each activity CSV and scores.csv.
    - Each subject file can be joined to scores.csv on its 'number' key
      within its group.

    Parameters
    ----------
    raw_dir : Path to the directory containing condition/, control/, scores.csv.
              Defaults to 'data/raw'.

    Returns
    -------
    activity_df : Long-format DataFrame with one row per minute per subject.
                  Columns: timestamp, date, activity, subject_id, group, label,
                           + all scores.csv columns joined in.
    subject_df  : Wide-format DataFrame with one row per subject (scores only +
                  group + label). Useful for stratified CV splits.
    """
    raw_dir = Path(raw_dir)

    condition_dir = raw_dir / "condition"
    control_dir = raw_dir / "control"

    # ── Validate directories exist ───────────────────────────────────────────
    for d in (condition_dir, control_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"[data_loader] Directory not found: {d}\n"
                "Please extract the Depresjon dataset into data/raw/ before running."
            )

    # ── Collect file lists ───────────────────────────────────────────────────
    condition_files = sorted(condition_dir.glob("condition_*.csv"))
    control_files = sorted(control_dir.glob("control_*.csv"))

    n_cond = len(condition_files)
    n_ctrl = len(control_files)

    if n_cond != EXPECTED_CONDITION_COUNT:
        raise ValueError(
            f"[data_loader] Expected {EXPECTED_CONDITION_COUNT} condition files, "
            f"found {n_cond}. Check data/raw/condition/."
        )
    if n_ctrl != EXPECTED_CONTROL_COUNT:
        raise ValueError(
            f"[data_loader] Expected {EXPECTED_CONTROL_COUNT} control files, "
            f"found {n_ctrl}. Check data/raw/control/."
        )

    print(
        f"[data_loader] Found {n_cond} condition files and {n_ctrl} control files. ✓"
    )

    # ── Load scores ──────────────────────────────────────────────────────────
    scores = _load_scores(raw_dir)
    scores_index = scores.set_index("number")

    # ── Load activity CSVs ────────────────────────────────────────────────────
    frames: list[pd.DataFrame] = []
    subject_records: list[dict] = []

    for group, files in [("condition", condition_files), ("control", control_files)]:
        for filepath in files:
            subject_id = _extract_subject_key(filepath)  # e.g. "condition_7"

            # Join to scores.csv: key is the full "condition_7" / "control_14" string
            if subject_id not in scores_index.index:
                raise KeyError(
                    f"[data_loader] '{subject_id}' not found in scores.csv 'number' column. "
                    f"Cannot join clinical data for this subject."
                )

            score_row = scores_index.loc[subject_id].to_dict()

            # Load minute-level activity
            df = _load_activity_csv(filepath, subject_id=subject_id, group=group)

            # Attach scores columns to every row
            for col, val in score_row.items():
                df[col] = val

            frames.append(df)

            # Build subject-level record
            subject_records.append(
                {
                    "subject_id": subject_id,
                    "group": group,
                    "label": df["label"].iloc[0],
                    **score_row,
                }
            )

    activity_df = pd.concat(frames, ignore_index=True)
    subject_df = pd.DataFrame(subject_records)

    n_subjects = len(subject_df)
    n_depressed = int(subject_df["label"].sum())
    n_control = n_subjects - n_depressed

    print(
        f"[data_loader] Loaded {n_subjects} subjects total: "
        f"{n_depressed} depressed (condition), {n_control} healthy (control). ✓"
    )
    print(
        f"[data_loader] Activity records: {len(activity_df):,} rows "
        f"across {n_subjects} subjects."
    )

    # ── Note condition-group heterogeneity (spec requirement) ────────────────
    afftype_counts = (
        subject_df[subject_df["group"] == "condition"]["afftype"]
        .value_counts()
        .to_dict()
    )
    print(
        f"[data_loader] Condition group afftype breakdown "
        f"(1=bipolar II, 2=unipolar, 3=bipolar I): {afftype_counts}"
    )
    print(
        "[data_loader] WARNING: The condition group is heterogeneous (bipolar I/II + "
        "unipolar). See Limitations in outputs/report.md."
    )

    return activity_df, subject_df
