"""
tests/test_data_loader.py — RhythmCheck
========================================
Tests for src/data_loader.py.

Tests cover:
- Happy path: correct file counts, correct join, correct labels, afftype present.
- Error path: wrong file counts, missing scores key, missing required columns.

Uses a minimal synthetic dataset written to a temporary directory —
never touches data/raw/.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    EXPECTED_CONDITION_COUNT,
    EXPECTED_CONTROL_COUNT,
    load_dataset,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_activity_csv(path: Path, rows: int = 10) -> None:
    """Write a minimal valid activity CSV."""
    lines = ["timestamp,date,activity"]
    for i in range(rows):
        lines.append(f"2003-05-07 12:{i:02d}:00,2003-05-07,{i * 10}")
    path.write_text("\n".join(lines))


def _make_scores_csv(path: Path, condition_ids: list[int], control_ids: list[int]) -> None:
    """Write a minimal scores.csv matching the given subject numbers."""
    lines = ["number,days,gender,age,afftype,melanch,inpatient,edu,marriage,work,madrs1,madrs2"]
    for n in condition_ids:
        lines.append(f"condition_{n},13,2,35-39,2,2,2,6-10,1,2,20,18")
    for n in control_ids:
        lines.append(f"control_{n},13,1,25-29,NA,NA,NA,,NA,NA,NA,NA")
    path.write_text("\n".join(lines))


def _build_fake_dataset(
    tmp_path: Path,
    n_condition: int = EXPECTED_CONDITION_COUNT,
    n_control: int = EXPECTED_CONTROL_COUNT,
    condition_ids: list[int] | None = None,
    control_ids: list[int] | None = None,
) -> Path:
    """
    Build a minimal fake Depresjon-style dataset in tmp_path/raw/.

    Returns the raw_dir path.
    """
    raw_dir = tmp_path / "raw"
    cond_dir = raw_dir / "condition"
    ctrl_dir = raw_dir / "control"
    cond_dir.mkdir(parents=True)
    ctrl_dir.mkdir(parents=True)

    cond_ids = condition_ids if condition_ids is not None else list(range(1, n_condition + 1))
    ctrl_ids = control_ids if control_ids is not None else list(range(1, n_control + 1))

    for n in cond_ids:
        _make_activity_csv(cond_dir / f"condition_{n}.csv")
    for n in ctrl_ids:
        _make_activity_csv(ctrl_dir / f"control_{n}.csv")

    _make_scores_csv(raw_dir / "scores.csv", cond_ids, ctrl_ids)

    return raw_dir


# ── Happy-path tests ──────────────────────────────────────────────────────────

class TestLoadDatasetHappyPath:
    def test_returns_two_dataframes(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        activity_df, subject_df = load_dataset(raw_dir)
        assert isinstance(activity_df, pd.DataFrame)
        assert isinstance(subject_df, pd.DataFrame)

    def test_correct_subject_count(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        _, subject_df = load_dataset(raw_dir)
        assert len(subject_df) == EXPECTED_CONDITION_COUNT + EXPECTED_CONTROL_COUNT

    def test_labels_correct(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        _, subject_df = load_dataset(raw_dir)
        assert int(subject_df["label"].sum()) == EXPECTED_CONDITION_COUNT
        assert int((subject_df["label"] == 0).sum()) == EXPECTED_CONTROL_COUNT

    def test_group_column_values(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        _, subject_df = load_dataset(raw_dir)
        assert set(subject_df["group"].unique()) == {"condition", "control"}

    def test_activity_df_has_required_columns(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        activity_df, _ = load_dataset(raw_dir)
        for col in ("timestamp", "date", "activity", "subject_id", "group", "label"):
            assert col in activity_df.columns, f"Missing column: {col}"

    def test_subject_df_has_subject_id(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        _, subject_df = load_dataset(raw_dir)
        assert "subject_id" in subject_df.columns

    def test_join_does_not_mix_groups(self, tmp_path):
        """
        condition_1 and control_1 are different subjects —
        their score rows must not be swapped.
        Condition rows must have label=1, control rows label=0.
        """
        raw_dir = _build_fake_dataset(tmp_path)
        activity_df, _ = load_dataset(raw_dir)

        cond_labels = activity_df[activity_df["group"] == "condition"]["label"].unique()
        ctrl_labels = activity_df[activity_df["group"] == "control"]["label"].unique()

        assert list(cond_labels) == [1], f"Condition rows should all have label=1, got {cond_labels}"
        assert list(ctrl_labels) == [0], f"Control rows should all have label=0, got {ctrl_labels}"

    def test_afftype_column_present(self, tmp_path):
        """afftype must be carried through so heterogeneity can be flagged downstream."""
        raw_dir = _build_fake_dataset(tmp_path)
        _, subject_df = load_dataset(raw_dir)
        assert "afftype" in subject_df.columns

    def test_scores_columns_joined_to_activity(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        activity_df, _ = load_dataset(raw_dir)
        assert "days" in activity_df.columns
        assert "madrs1" in activity_df.columns

    def test_activity_row_count(self, tmp_path):
        """Each fake subject has 10 activity rows; total should be 55 * 10."""
        raw_dir = _build_fake_dataset(tmp_path)
        activity_df, _ = load_dataset(raw_dir)
        total_subjects = EXPECTED_CONDITION_COUNT + EXPECTED_CONTROL_COUNT
        assert len(activity_df) == total_subjects * 10


# ── Error-path tests ──────────────────────────────────────────────────────────

class TestLoadDatasetErrors:
    def test_wrong_condition_count_raises(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path, n_condition=5, n_control=EXPECTED_CONTROL_COUNT)
        with pytest.raises(ValueError, match="Expected 23 condition files"):
            load_dataset(raw_dir)

    def test_wrong_control_count_raises(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path, n_condition=EXPECTED_CONDITION_COUNT, n_control=10)
        with pytest.raises(ValueError, match="Expected 32 control files"):
            load_dataset(raw_dir)

    def test_missing_scores_csv_raises(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        (raw_dir / "scores.csv").unlink()
        with pytest.raises(FileNotFoundError, match="scores.csv not found"):
            load_dataset(raw_dir)

    def test_missing_condition_dir_raises(self, tmp_path):
        raw_dir = _build_fake_dataset(tmp_path)
        import shutil
        shutil.rmtree(raw_dir / "condition")
        with pytest.raises(FileNotFoundError, match="Directory not found"):
            load_dataset(raw_dir)

    def test_subject_missing_from_scores_raises(self, tmp_path):
        """If a file exists in condition/ but is not in scores.csv, fail loudly."""
        raw_dir = _build_fake_dataset(tmp_path)
        # Add an extra condition file not listed in scores.csv
        extra = raw_dir / "condition" / f"condition_{EXPECTED_CONDITION_COUNT + 1}.csv"
        _make_activity_csv(extra)
        # Now there are 24 condition files — wrong count raises first
        with pytest.raises(ValueError, match="Expected 23 condition files"):
            load_dataset(raw_dir)

    def test_missing_activity_columns_raises(self, tmp_path):
        """Activity CSV missing required columns should fail loudly."""
        raw_dir = _build_fake_dataset(tmp_path)
        # Overwrite condition_1.csv with a broken file (no 'activity' column)
        broken = raw_dir / "condition" / "condition_1.csv"
        broken.write_text("timestamp,date\n2003-05-07 12:00:00,2003-05-07\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_dataset(raw_dir)
