"""
ml/health_model.py
==================
Trains a LightGBM health scoring model on cleaned OFF data.

Run after food_pipeline_v4.py:
  python ml/health_model.py

Saves:
  models/health_model.pkl   <- loaded by app.py at startup
  data/model_meta.json      <- feature list + model stats for UI display
"""

import json
import pickle
import time
import os
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------
CSV_PATH      = 'data/cleaned_food_data_v4.csv'
FEATURES_PATH = 'data/feature_cols.txt'
TFIDF_NPZ     = 'data/tfidf_matrix.npz'
MODEL_PATH    = 'models/health_model.pkl'
META_PATH     = 'data/model_meta.json'
RANDOM_STATE  = 42
TOP_FEATURES  = 150

OVERFIT_THRESHOLD = 0.15

os.makedirs('models', exist_ok=True)

# -----------------------------------------------------------------------
# STEP 1 — LOAD DATA
# -----------------------------------------------------------------------
print("=" * 60)
print("SHELFSCAN — HEALTH MODEL TRAINING")
print("=" * 60)
print("\n[1/6] Loading cleaned data...")

df = pd.read_csv(CSV_PATH, low_memory=False)
print(f"  Rows: {len(df):,}")

# Load feature columns saved by pipeline
with open(FEATURES_PATH) as f:
    feature_cols = [
        line.strip().split(',', 1)[1]
        for line in f if line.strip()
    ]

# Only keep features that exist in this CSV
feature_cols = [c for c in feature_cols if c in df.columns]
print(f"  Tabular features available: {len(feature_cols)}")

# Load TF-IDF matrix — already aligned with CSV rows
tfidf_matrix = load_npz(TFIDF_NPZ)
print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

assert tfidf_matrix.shape[0] == len(df), (
    f"Shape mismatch: tfidf={tfidf_matrix.shape[0]} "
    f"csv={len(df)} — re-run food_pipeline_v4.py"
)

# Combine tabular + TF-IDF (keep sparse)
X_tabular = csr_matrix(df[feature_cols].fillna(0).values)
X_full    = hstack([X_tabular, tfidf_matrix])

# Build full feature name list
tfidf_feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
all_feature_names   = feature_cols + tfidf_feature_names

print(f"  Total features (tabular + TF-IDF): {X_full.shape[1]}")

y = df['health_score'].values

# -----------------------------------------------------------------------
# STEP 2 — TRAIN / TEST SPLIT
# -----------------------------------------------------------------------
print("\n[2/6] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=RANDOM_STATE
)
print(f"  Train: {len(y_train):,}   Test: {len(y_test):,}")

# -----------------------------------------------------------------------
# SAMPLE WEIGHTS
# Correct for health tier skew in training data.
# Rarer tiers get higher weight so model doesn't over-optimise
# for the Moderate majority at the expense of extremes.
# -----------------------------------------------------------------------
print("\n  Computing sample weights...")

tier        = df['health_tier'].values
tier_counts = np.bincount(tier)

print(f"  Tier distribution:")
for cls, name in [(0, 'Unhealthy'), (1, 'Moderate'), (2, 'Healthy')]:
    pct = tier_counts[cls] / len(tier) * 100
    print(f"    {cls} ({name}): {tier_counts[cls]:,}  ({pct:.1f}%)")

weights      = 1.0 / tier_counts[tier]
weights     /= weights.mean()
train_weights = weights[:len(y_train)]

print(f"  Weight range: {weights.min():.3f} – {weights.max():.3f}  "
      f"mean={weights.mean():.3f}")

# -----------------------------------------------------------------------
# STEP 3 — ISOLATION FOREST OUTLIER REMOVAL
# -----------------------------------------------------------------------
print("\n[3/6] Removing outliers with Isolation Forest...")

iso = IsolationForest(
    contamination=0.02,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
# .toarray() converts sparse -> dense (~350MB RAM spike on 120k x 367)
# Only the tabular portion is used for outlier detection —
# TF-IDF columns are too sparse to be meaningful for IsolationForest
X_train_tabular = X_train[:, :len(feature_cols)].toarray()
outlier_labels  = iso.fit_predict(X_train_tabular)
mask            = outlier_labels == 1
removed         = (~mask).sum()

X_train_clean  = X_train[mask]
y_train_clean  = y_train[mask]
train_weights  = train_weights[mask]

print(f"  Removed {removed:,} outliers "
      f"({removed/len(outlier_labels)*100:.1f}%)")
print(f"  Training set after cleaning: {len(y_train_clean):,}")

# -----------------------------------------------------------------------
# STEP 4 — LIGHTGBM FEATURE SELECTION
# -----------------------------------------------------------------------
print("\n[4/6] Feature selection with LightGBM importance...")

temp_model = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)
temp_model.fit(X_train_clean, y_train_clean)

importances = temp_model.feature_importances_
top_idx     = np.argsort(importances)[-TOP_FEATURES:]

X_train_sel = X_train_clean[:, top_idx]
X_test_sel  = X_test[:, top_idx]

selected_feature_names = [all_feature_names[i] for i in top_idx]

print(f"  Features before selection: {X_full.shape[1]}")
print(f"  Features after selection : {TOP_FEATURES}")

# -----------------------------------------------------------------------
# STEP 5 — TRAIN FINAL LIGHTGBM MODEL
# -----------------------------------------------------------------------
print("\n[5/6] Training LightGBM regressor...")
print("  (this takes 1-3 minutes — not frozen)")

model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=127,
    max_depth=10,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

t_start = time.time()
model.fit(
    X_train_sel, y_train_clean,
    sample_weight=train_weights,
    eval_set=[(X_test_sel, y_test)],
    callbacks=[
        lgb.early_stopping(50, verbose=False),
        lgb.log_evaluation(False),
    ]
)
t_elapsed = time.time() - t_start
print(f"  Training complete in {t_elapsed:.1f}s")

# -----------------------------------------------------------------------
# EVALUATE
# -----------------------------------------------------------------------
print("\n  Evaluating...")

y_pred_train = model.predict(X_train_sel)
y_pred_test  = model.predict(X_test_sel)

train_rmse = np.sqrt(mean_squared_error(y_train_clean, y_pred_train))
test_rmse  = np.sqrt(mean_squared_error(y_test,        y_pred_test))
train_r2   = r2_score(y_train_clean, y_pred_train)
test_r2    = r2_score(y_test,        y_pred_test)

print(f"  Train RMSE : {train_rmse:.2f}   R2 : {train_r2:.3f}")
print(f"  Test  RMSE : {test_rmse:.2f}   R2 : {test_r2:.3f}")

r2_gap = train_r2 - test_r2
if r2_gap > OVERFIT_THRESHOLD:
    print(f"\n  ⚠️  OVERFIT WARNING: train R2 - test R2 = {r2_gap:.3f} "
          f"(threshold: {OVERFIT_THRESHOLD})")
    print(f"  Consider: reduce n_estimators, reduce num_leaves, "
          f"increase min_child_samples")
else:
    print(f"  ✅ Overfit check passed (gap={r2_gap:.3f})")

# -----------------------------------------------------------------------
# SANITY CHECK
# Nutella should always score lower than water.
# -----------------------------------------------------------------------
print("\n  Sanity check...")

nutella_vec = np.zeros((1, TOP_FEATURES))
water_vec   = np.zeros((1, TOP_FEATURES))
col_idx     = {c: i for i, c in enumerate(selected_feature_names)}

for col, val in [
    ('sugars',                    57),
    ('fat',                       31),
    ('saturated_fat',             11),
    ('energy_kcal',              539),
    ('fiber',                      2),
    ('proteins',                   6),
    ('additive_risk_score',        3),
    ('nova_score',                 4),
    ('ultra_processed_indicator',  1),
    ('sugar_AND_no_fiber',        57 / (2 + 0.1)),
    ('sugar_AND_ultra_processed', 57),
]:
    if col in col_idx:
        nutella_vec[0, col_idx[col]] = val

nutella_score = float(np.clip(model.predict(nutella_vec)[0], 0, 100))
water_score   = float(np.clip(model.predict(water_vec)[0],   0, 100))

print(f"    Nutella profile → {nutella_score:.1f}/100")
print(f"    Water profile   → {water_score:.1f}/100")

assert nutella_score < water_score, (
    f"Sanity check FAILED: Nutella ({nutella_score:.1f}) >= "
    f"Water ({water_score:.1f})\n"
    f"Model is not scoring correctly — check feature engineering."
)
print("    ✅ Nutella scores lower than water")

# -----------------------------------------------------------------------
# FEATURE IMPORTANCE (top 20)
# -----------------------------------------------------------------------
feat_importances = model.feature_importances_
top20_idx        = np.argsort(feat_importances)[::-1][:20]

print("\n  Top 20 feature importances:")
for rank, idx in enumerate(top20_idx, 1):
    print(f"    {rank:2d}. {selected_feature_names[idx]:<40} "
          f"{feat_importances[idx]:.4f}")

# -----------------------------------------------------------------------
# STEP 6 — SAVE
# -----------------------------------------------------------------------
print("\n[6/6] Saving model...")

# Save model + metadata needed for inference
model_bundle = {
    'model':                model,
    'selected_feature_names': selected_feature_names,
    'top_idx':              top_idx,
    'all_feature_names':    all_feature_names,
    'feature_cols':         feature_cols,
}

with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model_bundle, f)

model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
print(f"  Saved → {MODEL_PATH}  ({model_size_mb:.1f} MB)")

meta = {
    'feature_cols':         feature_cols,
    'selected_features':    selected_feature_names,
    'n_features_total':     X_full.shape[1],
    'n_features_selected':  TOP_FEATURES,
    'train_rmse':           round(train_rmse, 3),
    'test_rmse':            round(test_rmse, 3),
    'train_r2':             round(train_r2, 3),
    'test_r2':              round(test_r2, 3),
    'r2_gap':               round(r2_gap, 3),
    'overfit_warning':      r2_gap > OVERFIT_THRESHOLD,
    'n_train':              len(y_train_clean),
    'n_test':               len(y_test),
    'outliers_removed':     int(removed),
    'training_time_s':      round(t_elapsed, 1),
    'model_size_mb':        round(model_size_mb, 1),
    'top_features':         selected_feature_names[:20],
    'tier_counts': {
        'unhealthy': int(tier_counts[0]),
        'moderate':  int(tier_counts[1]),
        'healthy':   int(tier_counts[2]),
    },
    'sanity_check': {
        'nutella_score': round(nutella_score, 1),
        'water_score':   round(water_score, 1),
        'passed':        nutella_score < water_score,
    },
}
with open(META_PATH, 'w') as f:
    json.dump(meta, f, indent=2)
print(f"  Saved → {META_PATH}")

# -----------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------
print(f"\n{'=' * 60}")
print("TRAINING COMPLETE")
print(f"{'=' * 60}")
print(f"  Model        : LightGBM Regressor (early stopping)")
print(f"  Features     : {TOP_FEATURES} selected from {X_full.shape[1]}")
print(f"  Outliers out : {removed:,}")
print(f"  Train time   : {t_elapsed:.1f}s")
print(f"  Test RMSE    : {test_rmse:.2f}")
print(f"  Test R2      : {test_r2:.3f}")
print(f"  Model size   : {model_size_mb:.1f} MB")
if r2_gap > OVERFIT_THRESHOLD:
    print(f"  ⚠️  Overfit gap : {r2_gap:.3f} — consider tuning")
else:
    print(f"  ✅ Overfit gap : {r2_gap:.3f} — healthy generalisation")
print(f"\nNext: streamlit run app.py")