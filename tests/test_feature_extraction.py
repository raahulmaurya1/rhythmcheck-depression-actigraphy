"""
tests/test_feature_extraction.py — RhythmCheck
================================================
Unit tests for src/feature_extraction.py.

Per the spec (Step 8): tests use a small synthetic activity series with
known, hand-computed rhythm patterns so results can be verified analytically.

Synthetic signal design
-----------------------
We construct a 2-day, minute-resolution sinusoidal rhythm:
  - Active period (08:00-20:00): activity = 100
  - Rest period  (20:00-08:00): activity = 5

This gives deterministic, hand-checkable values for every feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_extraction import (
    SLEEP_THRESHOLD,
    _to_hourly,
    compute_day_night_ratio,
    compute_is,
    compute_iv,
    compute_m10_l5,
    compute_sleep_efficiency,
    extract_subject_features,
    build_feature_matrix,
)


# ── Synthetic signal helpers ──────────────────────────────────────────────────

def _make_perfect_rhythm(n_days: int = 3, day_val: float = 100.0, night_val: float = 5.0):
    """
    Build a perfectly regular day/night activity series.

    Active (08:00-20:00): day_val counts per minute.
    Rest   (20:00-08:00): night_val counts per minute.

    Returns (timestamps, activity) as pd.Series.
    """
    start = pd.Timestamp("2023-01-01 00:00:00")
    n_minutes = n_days * 24 * 60
    timestamps = pd.date_range(start, periods=n_minutes, freq="min")

    activity = np.where(
        (timestamps.hour >= 8) & (timestamps.hour < 20), day_val, night_val
    ).astype(float)

    return pd.Series(timestamps), pd.Series(activity)


def _make_flat_rhythm(n_days: int = 3, val: float = 50.0):
    """Completely flat activity — no circadian variation at all."""
    start = pd.Timestamp("2023-01-01 00:00:00")
    n_minutes = n_days * 24 * 60
    timestamps = pd.date_range(start, periods=n_minutes, freq="min")
    activity = np.full(n_minutes, val, dtype=float)
    return pd.Series(timestamps), pd.Series(activity)


# ── _to_hourly ────────────────────────────────────────────────────────────────

class TestToHourly:
    def test_shape(self):
        ts, act = _make_perfect_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        # 3 days * 24 hours = 72 hourly bins
        assert len(hourly) == 3 * 24

    def test_daytime_value(self):
        """Daytime hours (08-19) should average to day_val=100."""
        ts, act = _make_perfect_rhythm(n_days=3, day_val=100.0, night_val=0.0)
        hourly = _to_hourly(act, ts)
        daytime = hourly[hourly.index.hour == 10]  # 10am — solidly in active window
        assert all(abs(v - 100.0) < 1e-6 for v in daytime.values)

    def test_nighttime_value(self):
        """Nighttime hours should average to night_val=5."""
        ts, act = _make_perfect_rhythm(n_days=3, day_val=100.0, night_val=5.0)
        hourly = _to_hourly(act, ts)
        nighttime = hourly[hourly.index.hour == 2]  # 2am — solidly in rest window
        assert all(abs(v - 5.0) < 1e-6 for v in nighttime.values)


# ── compute_iv ────────────────────────────────────────────────────────────────

class TestIV:
    def test_flat_signal_is_zero_or_nan(self):
        """
        Flat signal: no transitions → numerator = 0, denominator = 0.
        compute_iv guards against div-by-zero and returns NaN.
        """
        ts, act = _make_flat_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        iv = compute_iv(hourly)
        # Flat denominator is 0 → NaN by design (no variance → undefined IV)
        assert np.isnan(iv)

    def test_perfect_rhythm_iv_is_finite_positive(self):
        """Perfect regular rhythm has finite, positive IV."""
        ts, act = _make_perfect_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        iv = compute_iv(hourly)
        assert np.isfinite(iv)
        assert iv > 0

    def test_high_frequency_noise_increases_iv(self):
        """
        A high-frequency alternating signal (extreme fragmentation) should have
        much higher IV than a smooth regular rhythm.
        """
        # Clean regular rhythm
        ts, act_clean = _make_perfect_rhythm(n_days=5)
        hourly_clean = _to_hourly(act_clean, ts)
        iv_clean = compute_iv(hourly_clean)

        # Highly fragmented: rapidly alternating 0 and 200 every minute
        rng = np.random.default_rng(0)
        act_random = pd.Series(rng.choice([0.0, 200.0], size=len(act_clean)))
        hourly_random = _to_hourly(act_random, ts)
        iv_random = compute_iv(hourly_random)

        # A totally random signal should have higher IV than a clean rhythm
        assert iv_random > iv_clean

    def test_too_short_returns_nan(self):
        """Fewer than 3 hourly points should return NaN."""
        short = pd.Series([10.0, 20.0], index=pd.date_range("2023-01-01", periods=2, freq="h"))
        assert np.isnan(compute_iv(short))


# ── compute_is ────────────────────────────────────────────────────────────────

class TestIS:
    def test_perfect_rhythm_is_near_one(self):
        """
        Perfectly regular rhythm repeated identically each day → IS ≈ 1.
        """
        ts, act = _make_perfect_rhythm(n_days=5)
        hourly = _to_hourly(act, ts)
        is_val = compute_is(hourly)
        # For a perfectly repeating signal, IS should be very close to 1
        assert is_val == pytest.approx(1.0, abs=0.01)

    def test_flat_signal_is_nan(self):
        """
        Flat signal has zero variance → IS is NaN (division by zero guard).
        """
        ts, act = _make_flat_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        is_val = compute_is(hourly)
        assert np.isnan(is_val)

    def test_is_range(self):
        """IS should be clipped to [0, 1]."""
        ts, act = _make_perfect_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        is_val = compute_is(hourly)
        assert 0 <= is_val <= 1

    def test_too_short_returns_nan(self):
        """Fewer than 2 full days returns NaN."""
        ts, act = _make_perfect_rhythm(n_days=1)
        hourly = _to_hourly(act, ts)
        is_val = compute_is(hourly)
        assert np.isnan(is_val)


# ── compute_m10_l5 ────────────────────────────────────────────────────────────

class TestM10L5:
    def test_m10_greater_than_l5(self):
        """M10 must be >= L5 by definition."""
        ts, act = _make_perfect_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        assert res["m10"] >= res["l5"]

    def test_perfect_rhythm_m10_approx_day_val(self):
        """
        With a perfect day=100/night=5 rhythm, M10 should ≈ 100.
        (The 10-hour peak window lands squarely in daytime.)
        """
        ts, act = _make_perfect_rhythm(n_days=5, day_val=100.0, night_val=5.0)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        assert res["m10"] == pytest.approx(100.0, abs=1.0)

    def test_perfect_rhythm_l5_approx_night_val(self):
        """L5 should ≈ 5 (nighttime activity level)."""
        ts, act = _make_perfect_rhythm(n_days=5, day_val=100.0, night_val=5.0)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        assert res["l5"] == pytest.approx(5.0, abs=1.0)

    def test_ra_range(self):
        """RA must be in [0, 1]."""
        ts, act = _make_perfect_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        assert 0 <= res["ra"] <= 1

    def test_flat_rhythm_ra_is_zero(self):
        """Flat signal: M10 == L5 → RA = 0."""
        ts, act = _make_flat_rhythm(n_days=3)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        assert res["ra"] == pytest.approx(0.0, abs=1e-10)

    def test_m10_onset_in_daytime(self):
        """M10 onset should be during daytime hours for a day=high rhythm."""
        ts, act = _make_perfect_rhythm(n_days=5, day_val=100.0, night_val=0.0)
        hourly = _to_hourly(act, ts)
        res = compute_m10_l5(hourly)
        onset = res["m10_onset"]
        # The 10-hour active window (8am-8pm) should start around hour 8
        assert 6 <= onset <= 12, f"M10 onset unexpectedly at hour {onset}"

    def test_too_short_returns_nan(self):
        """Fewer than 24 hours returns NaN for all M10/L5 values."""
        ts, act = _make_perfect_rhythm(n_days=1)
        # Only give 12 hours
        hourly = _to_hourly(act[:12], ts[:12])
        res = compute_m10_l5(hourly)
        assert np.isnan(res["m10"])
        assert np.isnan(res["l5"])


# ── compute_day_night_ratio ───────────────────────────────────────────────────

class TestDayNightRatio:
    def test_known_ratio(self):
        """
        day_val=100, night_val=5 → ratio should be exactly 100/5 = 20.
        """
        ts, act = _make_perfect_rhythm(n_days=3, day_val=100.0, night_val=5.0)
        ratio = compute_day_night_ratio(act, ts)
        assert ratio == pytest.approx(20.0, rel=0.01)

    def test_flat_rhythm_ratio_is_one(self):
        """Flat signal → day and night activity equal → ratio = 1."""
        ts, act = _make_flat_rhythm(n_days=3, val=50.0)
        ratio = compute_day_night_ratio(act, ts)
        assert ratio == pytest.approx(1.0, abs=0.01)

    def test_zero_night_returns_nan(self):
        """Zero nighttime activity → ratio is NaN (avoid division by zero)."""
        ts, act = _make_perfect_rhythm(n_days=3, day_val=100.0, night_val=0.0)
        ratio = compute_day_night_ratio(act, ts)
        assert np.isnan(ratio)


# ── compute_sleep_efficiency ──────────────────────────────────────────────────

class TestSleepEfficiency:
    def test_perfect_sleep(self):
        """
        Nighttime activity = 0 (< SLEEP_THRESHOLD=10) → sleep_efficiency = 1.0.
        """
        ts, act = _make_perfect_rhythm(n_days=3, day_val=100.0, night_val=0.0)
        se = compute_sleep_efficiency(act, ts)
        assert se == pytest.approx(1.0, abs=0.01)

    def test_no_sleep(self):
        """
        Nighttime activity = 100 (> SLEEP_THRESHOLD) → sleep_efficiency ≈ 0.
        """
        ts, act = _make_flat_rhythm(n_days=3, val=100.0)
        se = compute_sleep_efficiency(act, ts)
        assert se == pytest.approx(0.0, abs=0.01)

    def test_range(self):
        """Sleep efficiency must be in [0, 1]."""
        ts, act = _make_perfect_rhythm(n_days=3)
        se = compute_sleep_efficiency(act, ts)
        assert 0 <= se <= 1


# ── extract_subject_features ──────────────────────────────────────────────────

class TestExtractSubjectFeatures:
    def test_returns_all_expected_keys(self):
        expected_keys = {
            "subject_id", "mean_daily_activity", "std_daily_activity",
            "mean_minute_activity", "iv", "is_", "m10", "l5", "ra",
            "m10_onset", "l5_onset", "day_night_ratio", "sleep_efficiency",
        }
        ts, act = _make_perfect_rhythm(n_days=3)
        result = extract_subject_features("test_1", act, ts)
        assert expected_keys <= set(result.keys())

    def test_mean_minute_activity_correct(self):
        """
        For day=100, night=5: 12 day-hours + 12 night-hours per day.
        mean = (12*100 + 12*5) / 24 = (1200 + 60) / 24 = 52.5
        """
        ts, act = _make_perfect_rhythm(n_days=5, day_val=100.0, night_val=5.0)
        result = extract_subject_features("test_1", act, ts)
        assert result["mean_minute_activity"] == pytest.approx(52.5, rel=0.01)


# ── build_feature_matrix ──────────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def _make_activity_df(self, n_days: int = 3) -> pd.DataFrame:
        """Build a minimal activity_df mimicking data_loader output."""
        records = []
        for sid, group, label, day_val, night_val in [
            ("condition_1", "condition", 1, 40.0, 10.0),
            ("condition_2", "condition", 1, 35.0, 8.0),
            ("control_1",   "control",   0, 80.0, 3.0),
        ]:
            ts_series, act_series = _make_perfect_rhythm(n_days, day_val, night_val)
            df = pd.DataFrame({
                "subject_id": sid,
                "group": group,
                "label": label,
                "timestamp": ts_series,
                "activity": act_series,
            })
            records.append(df)
        return pd.concat(records, ignore_index=True)

    def test_one_row_per_subject(self, tmp_path):
        activity_df = self._make_activity_df()
        feat_df = build_feature_matrix(activity_df, output_dir=tmp_path, save=False)
        assert len(feat_df) == 3

    def test_label_preserved(self, tmp_path):
        activity_df = self._make_activity_df()
        feat_df = build_feature_matrix(activity_df, output_dir=tmp_path, save=False)
        assert set(feat_df["label"].unique()) == {0, 1}

    def test_saves_csv(self, tmp_path):
        activity_df = self._make_activity_df()
        build_feature_matrix(activity_df, output_dir=tmp_path, save=True)
        assert (tmp_path / "features.csv").exists()

    def test_saves_feature_dictionary(self, tmp_path):
        activity_df = self._make_activity_df()
        build_feature_matrix(activity_df, output_dir=tmp_path, save=True)
        assert (tmp_path / "feature_dictionary.md").exists()

    def test_no_duplicate_subjects(self, tmp_path):
        activity_df = self._make_activity_df()
        feat_df = build_feature_matrix(activity_df, output_dir=tmp_path, save=False)
        assert feat_df["subject_id"].nunique() == len(feat_df)
