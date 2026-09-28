"""
quick_test.py — Manual smoke test for RhythmCheck pipeline (Steps 2-5)

Run from project root with .venv activated:
    python quick_test.py

This is NOT a substitute for the automated tests — it runs the real dataset
end-to-end so you can see actual output and check everything works.
"""

import pandas as pd

print("=" * 60)
print("RhythmCheck — Manual Smoke Test")
print("=" * 60)

# ── Step 2: Data Loader ───────────────────────────────────────────────────────
print("\n[STEP 2] Loading dataset from data/raw/ ...")
from src.data_loader import load_dataset

activity_df, subject_df = load_dataset("data/raw")

print(f"\nSubject summary:")
print(subject_df[["subject_id", "group", "label", "afftype", "days"]].to_string(index=False))

# ── Step 3: Feature Extraction ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[STEP 3] Extracting features ...")
from src.feature_extraction import build_feature_matrix

feat_df = build_feature_matrix(activity_df, output_dir="data/processed", save=True)

print(f"\nFeature matrix ({feat_df.shape[0]} subjects × {feat_df.shape[1]} columns):")
display_cols = ["subject_id", "label", "mean_daily_activity", "iv", "is_", "ra", "day_night_ratio", "sleep_efficiency"]
print(feat_df[display_cols].round(3).to_string(index=False))

# ── Step 4: Models (quick sanity check — NOT the real evaluation) ─────────────
print("\n" + "=" * 60)
print("[STEP 4] Quick model sanity check (train=test, not real evaluation) ...")

from src.models import get_models
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np

from src.evaluate import FEATURE_COLS

X_raw = feat_df[FEATURE_COLS].values.astype(float)
y = feat_df["label"].values

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()
X = scaler.fit_transform(imputer.fit_transform(X_raw))

print(f"Feature matrix shape: {X.shape}, Labels: {y.sum()} depressed / {(y==0).sum()} healthy")

for model in get_models():
    model.fit(X, y)
    preds = model.predict(X)
    train_acc = (preds == y).mean()
    print(f"  {model.name}: train accuracy = {train_acc:.2f}  (not the real metric — see Step 5)")

# ── Step 5: LOSO-CV (real evaluation — takes ~1-2 min) ───────────────────────
print("\n" + "=" * 60)
print("[STEP 5] Running LOSO-CV on real data (this takes ~1-2 minutes) ...")
from src.evaluate import evaluate_all_models

results = evaluate_all_models(feat_df, n_bootstrap=500, verbose=True)

print("\n" + "=" * 60)
print("Smoke test complete.")
print(f"  features.csv       → data/processed/features.csv")
print(f"  feature_dictionary → data/processed/feature_dictionary.md")
