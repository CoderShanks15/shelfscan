"""
ml/predict.py
=============
Inference-only module. Loaded by app.py at startup.
Provides predict_health() which takes a product dict from utils/api.py
and returns a full scoring result dict.

Updated for LightGBM model_bundle format saved by health_model.py:
  bundle = {
      'model':                  LGBMRegressor,
      'selected_feature_names': list of 150 feature names,
      'top_idx':                indices into all_feature_names,
      'all_feature_names':      tabular + tfidf feature names,
      'feature_cols':           tabular-only feature names,
  }

Key differences from sklearn Pipeline version:
  - MODEL_PATH is now 'models/health_model.pkl' (not 'data/')
  - Feature vector built against selected_feature_names (150)
    not feature_cols (117)
  - model.predict() called directly, not pipeline.predict()
  - TF-IDF slots stay zero at inference (no vectorizer at runtime)
"""

import re
import json
import pickle
import numpy as np

MODEL_PATH = 'models/health_model.pkl'
META_PATH  = 'data/model_meta.json'

# Module-level cache — loaded once, reused on every scan
_model                  = None
_selected_feature_names = None
_col_idx                = None


def _load():
    """Lazy-load model bundle once per process."""
    global _model, _selected_feature_names, _col_idx

    if _model is not None:
        return

    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    _model                  = bundle['model']
    _selected_feature_names = bundle['selected_feature_names']
    _col_idx                = {c: i for i, c in enumerate(_selected_feature_names)}


def predict_health(product: dict) -> dict:
    """
    Takes a cleaned product dict from utils/api.py and returns:
    {
        score         : float 0-100
        verdict       : str   'Avoid' | 'Caution' | 'OK' | 'Great'
        verdict_emoji : str
        adjustments   : list of str  (human-readable score trail)
        breakdown     : dict of key nutrient labels for UI display
    }
    """
    _load()

    vec       = _build_feature_vector(product)
    raw_score = float(np.clip(_model.predict(vec)[0], 0, 100))

    score, adjustments = _apply_adjustments(raw_score, product)

    return {
        'score':         round(score, 1),
        'verdict':       _verdict(score),
        'verdict_emoji': _verdict_emoji(score),
        'adjustments':   adjustments,
        'breakdown':     _breakdown(product),
    }


# -----------------------------------------------------------------------
# FEATURE VECTOR BUILDER
# Builds a 150-element vector aligned to selected_feature_names.
# TF-IDF slots (tfidf_N names) stay zero — no vectorizer at inference.
# This is valid because tabular features carry the primary signal and
# the model was trained on sparse TF-IDF for most rows anyway.
# -----------------------------------------------------------------------

CONCERNING_RE = re.compile(
    r'hydrogenated|corn.syrup|aspartame|sucralose|msg|'
    r'nitrite|nitrate|carrageenan|polysorbate',
    re.IGNORECASE
)

HIGH_RISK_ADDITIVES = {
    'en:e250', 'en:e251', 'en:e252',
    'en:e102', 'en:e110', 'en:e124', 'en:e129',
    'en:e621', 'en:e951', 'en:e954',
}


def _build_feature_vector(product: dict) -> np.ndarray:
    """Map product dict fields to the 150-feature inference vector."""
    vec = np.zeros((1, len(_selected_feature_names)))

    def _set(col, val):
        if col in _col_idx:
            vec[0, _col_idx[col]] = val or 0

    # ── Raw nutrients ──
    fat    = product.get('fat')          or 0
    sugar  = product.get('sugars')       or 0
    fiber  = product.get('fiber')        or 0
    prot   = product.get('proteins')     or 0
    salt   = product.get('salt')         or 0
    sat    = product.get('saturated_fat') or 0
    kcal   = product.get('energy_kcal')  or 0
    carbs  = product.get('carbohydrates') or 0
    mono   = product.get('mono_fat')     or 0
    poly   = product.get('poly_fat')     or 0
    omega3 = product.get('omega3')       or 0
    omega6 = product.get('omega6')       or 0
    sodium = salt / 2.5   # approximate

    _set('energy_kcal',   kcal)
    _set('fat',           fat)
    _set('saturated_fat', sat)
    _set('carbohydrates', carbs)
    _set('sugars',        sugar)
    _set('fiber',         fiber)
    _set('proteins',      prot)
    _set('salt',          salt)
    _set('sodium',        sodium)
    _set('trans_fat',     product.get('trans_fat') or 0)
    _set('mono_fat',      mono)
    _set('poly_fat',      poly)
    _set('omega3',        omega3)
    _set('omega6',        omega6)

    # ── Processing signals ──
    adds       = len(product.get('additives', []))
    nova       = product.get('nova_group') or 2
    ultra      = 1 if nova >= 4 else 0
    ingr_text  = (product.get('ingredients', '') or '').lower()
    ingr_count = ingr_text.count(',') + 1 if ingr_text else 0
    concerning = len(CONCERNING_RE.findall(ingr_text))
    high_risk  = sum(1 for a in product.get('additives', [])
                     if a in HIGH_RISK_ADDITIVES)
    proc_score = adds * 2 + ultra * 3 + concerning * 2 + ingr_count / 10

    _set('additives_count',             adds)
    _set('additive_risk_score',         adds * 2)
    _set('additives_high_risk',         high_risk)
    _set('nova_score',                  nova)
    _set('ultra_processed_indicator',   ultra)
    _set('processing_score',            proc_score)
    _set('ingredient_count',            ingr_count)
    _set('concerning_ingredient_count', concerning)
    _set('allergens_count',             len(product.get('allergens', [])))
    _set('traces_count',                0)
    _set('palm_oil_risk',               1 if 'palm oil' in ingr_text else 0)

    # ── Ratio features ──
    fat_quality = (mono + poly) / (sat + 0.1)
    _set('fiber_sugar_ratio',    fiber / (sugar + 0.1))
    _set('protein_fat_ratio',    prot  / (fat + 0.1))
    _set('protein_energy_ratio', prot  / (kcal + 1))
    _set('fat_quality_ratio',    fat_quality)
    _set('omega_balance',        omega3 / (omega6 + 0.001))
    _set('carb_fiber_balance',   carbs / (fiber + 0.1))
    _set('energy_density',       kcal / 100)
    _set('sodium_salt_ratio',    sodium / (salt + 0.001))
    _set('is_high_energy_density', int(kcal > 400))
    _set('omega3_sat_ratio',     omega3 / (sat + 0.1))

    # ── Interaction features ──
    _set('sugar_AND_no_fiber',        sugar * (1 / (fiber + 0.1)))
    _set('sugar_AND_ultra_processed', sugar * ultra)
    _set('additive_AND_processed',    adds * ultra)
    _set('salt_AND_fat_bomb',         int(salt > 2 and fat > 20))
    _set('protein_fiber_bonus',       prot * fiber)
    _set('high_fat_low_quality',      fat * (1 / (fat_quality + 0.1)))
    _set('macro_balance_score', (
        (prot * 4 + fiber * 2) /
        (sugar * 3 + sat * 2 + salt * 5 + 1)
    ))

    # ── Label flags ──
    labels = ' '.join(product.get('labels', []))
    _set('is_organic',     int('organic'     in labels))
    _set('is_vegan',       int('vegan'       in labels))
    _set('is_vegetarian',  int('vegetarian'  in labels))
    _set('is_gluten_free', int('gluten-free' in labels or 'no-gluten' in labels))
    _set('is_wholegrain',  int('whole-grain' in labels or 'wholegrain' in labels))
    _set('is_no_palm_oil', int('no-palm-oil' in labels))
    _set('is_high_fiber',  int('high-fiber'  in labels or 'source-of-fiber' in labels))
    _set('is_no_salt',     int('no-salt'     in labels or 'low-salt' in labels))
    _set('is_raw',         int('raw'         in labels))
    _set('is_fortified',   int('fortified'   in labels or 'enriched' in labels))

    # ── Top-3 ingredient signals ──
    top3 = ' '.join(ingr_text.split(',')[:3])
    _set('sugar_in_top3', int(any(w in top3 for w in
                                  ['sugar', 'glucose', 'fructose', 'syrup', 'honey'])))
    _set('fat_in_top3',   int(any(w in top3 for w in
                                  ['oil', 'fat', 'butter', 'cream', 'lard'])))
    _set('salt_in_top3',  int(any(w in top3 for w in ['salt', 'sodium', 'msg'])))

    # ── Whole food score ──
    is_wholegrain = int('whole-grain' in labels or 'wholegrain' in labels)
    _set('whole_food_score', fiber * is_wholegrain / (adds + ultra + 1))

    return vec


# -----------------------------------------------------------------------
# RULE-BASED ADJUSTMENTS
# Applied on top of ML score for explainability.
# Each rule appends a human-readable line shown in the UI score trail.
# -----------------------------------------------------------------------

def _apply_adjustments(score: float, product: dict) -> tuple:
    adjustments = []

    def adjust(delta, reason):
        nonlocal score
        score = max(0, min(100, score + delta))
        sign  = '+' if delta >= 0 else ''
        adjustments.append(f"{sign}{delta:.0f}  {reason}")

    n_additives = len(product.get('additives', []))
    if n_additives >= 8:
        adjust(-8, f"{n_additives} additives detected")
    elif n_additives >= 4:
        adjust(-4, f"{n_additives} additives detected")

    ingr = (product.get('ingredients', '') or '').lower()
    if 'palm oil' in ingr:
        adjust(-3, "Contains palm oil")

    nova = product.get('nova_group')
    if nova == 4:
        adjust(-5, "NOVA 4 — ultra-processed")
    elif nova == 1:
        adjust(+5, "NOVA 1 — minimally processed")

    ns = (product.get('nutriscore') or '').upper()
    for grade, delta in {'A': +5, 'B': +2, 'D': -3, 'E': -6}.items():
        if ns == grade:
            adjust(delta, f"Nutri-Score {grade}")

    sugar = product.get('sugars', 0) or 0
    if sugar > 20:
        adjust(-4, f"High sugar ({sugar:.1f}g/100g)")

    salt = product.get('salt', 0) or 0
    if salt > 2:
        adjust(-3, f"High salt ({salt:.1f}g/100g)")

    fiber = product.get('fiber', 0) or 0
    if fiber >= 6:
        adjust(+4, f"High fiber ({fiber:.1f}g/100g)")
    elif fiber >= 3:
        adjust(+2, f"Good fiber ({fiber:.1f}g/100g)")

    prot = product.get('proteins', 0) or 0
    if prot >= 15:
        adjust(+3, f"High protein ({prot:.1f}g/100g)")

    labels = ' '.join(product.get('labels', []))
    if 'organic'     in labels: adjust(+2, "Certified organic")
    if 'no-palm-oil' in labels: adjust(+2, "No palm oil certified")
    if 'wholegrain'  in labels or 'whole-grain' in labels:
        adjust(+2, "Wholegrain certified")

    return round(score, 1), adjustments


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _verdict(score: float) -> str:
    if score >= 70: return 'Great'
    if score >= 50: return 'OK'
    if score >= 30: return 'Caution'
    return 'Avoid'


def _verdict_emoji(score: float) -> str:
    if score >= 70: return '✅'
    if score >= 50: return '🟡'
    if score >= 30: return '⚠️'
    return '🚫'


def _breakdown(product: dict) -> dict:
    """Per-nutrient traffic light labels for UI display."""
    sugar = product.get('sugars', 0) or 0
    sat   = product.get('saturated_fat', 0) or 0
    salt  = product.get('salt', 0) or 0
    fiber = product.get('fiber', 0) or 0
    prot  = product.get('proteins', 0) or 0
    adds  = len(product.get('additives', []))
    nova  = product.get('nova_group') or '?'

    return {
        'sugar':     '🔴 High'   if sugar > 20 else '🟡 Medium' if sugar > 8  else '🟢 Low',
        'sat_fat':   '🔴 High'   if sat > 5    else '🟡 Medium' if sat > 2    else '🟢 Low',
        'salt':      '🔴 High'   if salt > 2   else '🟡 Medium' if salt > 1   else '🟢 Low',
        'fiber':     '🟢 High'   if fiber >= 6 else '🟡 Medium' if fiber >= 3 else '🔴 Low',
        'protein':   '🟢 High'   if prot >= 15 else '🟡 Medium' if prot >= 7  else '🔴 Low',
        'additives': '🔴 Many'   if adds >= 8  else '🟡 Some'   if adds >= 3  else '🟢 Few',
        'nova':      nova,
    }