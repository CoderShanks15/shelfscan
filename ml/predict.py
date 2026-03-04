"""
ml/predict.py
=============
Inference-only module. Loaded by app.py at startup.
Provides predict_health() which takes a product dict from utils/api.py
and returns a full scoring result dict.
"""

import json
import pickle
import numpy as np

MODEL_PATH = 'data/health_model.pkl'
META_PATH  = 'data/model_meta.json'

_pipeline     = None
_feature_cols = None


def _load():
    """Lazy-load model and feature list once."""
    global _pipeline, _feature_cols
    if _pipeline is None:
        with open(MODEL_PATH, 'rb') as f:
            _pipeline = pickle.load(f)
        with open(META_PATH) as f:
            meta = json.load(f)
        _feature_cols = meta['feature_cols']


def predict_health(product: dict) -> dict:
    """
    Takes a cleaned product dict from utils/api.py and returns:
    {
        score         : float 0-100
        verdict       : str   'Avoid' | 'Caution' | 'OK' | 'Great'
        verdict_emoji : str
        adjustments   : list of (reason_str, delta_float) tuples
        breakdown     : dict of key nutrient contributions
    }
    """
    _load()

    # Build feature vector
    vec = _build_feature_vector(product)
    raw_score = float(np.clip(_pipeline.predict(vec)[0], 0, 100))

    # Rule-based adjustments on top of ML score
    score, adjustments = _apply_adjustments(raw_score, product)

    return {
        'score':         round(score, 1),
        'verdict':       _verdict(score),
        'verdict_emoji': _verdict_emoji(score),
        'adjustments':   adjustments,
        'breakdown':     _breakdown(product),
    }


def _build_feature_vector(product: dict) -> np.ndarray:
    """Map product dict fields to model feature vector."""
    vec = np.zeros((1, len(_feature_cols)))
    col_idx = {c: i for i, c in enumerate(_feature_cols)}

    # Direct nutrient mapping
    nutrient_map = {
        'energy_kcal':   product.get('energy_kcal', 0),
        'fat':           product.get('fat', 0),
        'saturated_fat': product.get('saturated_fat', 0),
        'carbohydrates': product.get('carbohydrates', 0),
        'sugars':        product.get('sugars', 0),
        'fiber':         product.get('fiber', 0),
        'proteins':      product.get('proteins', 0),
        'salt':          product.get('salt', 0),
    }
    for col, val in nutrient_map.items():
        if col in col_idx:
            vec[0, col_idx[col]] = val or 0

    # Derived features (replicate pipeline logic)
    fat   = product.get('fat', 0) or 0
    sugar = product.get('sugars', 0) or 0
    fiber = product.get('fiber', 0) or 0
    prot  = product.get('proteins', 0) or 0
    salt  = product.get('salt', 0) or 0
    sat   = product.get('saturated_fat', 0) or 0
    kcal  = product.get('energy_kcal', 0) or 0
    carbs = product.get('carbohydrates', 0) or 0
    adds  = len(product.get('additives', []))

    nova  = product.get('nova_group') or 2
    ultra = 1 if nova >= 4 else 0
    ingr_text = (product.get('ingredients', '') or '').lower()
    ingr_count = ingr_text.count(',') + 1 if ingr_text else 0

    derived = {
        'additives_count':         adds,
        'nova_score':              nova,
        'ultra_processed_indicator': ultra,
        'fiber_sugar_ratio':       fiber / (sugar + 0.1),
        'protein_fat_ratio':       prot  / (fat + 0.1),
        'protein_energy_ratio':    prot  / (kcal + 1),
        'carb_fiber_balance':      carbs / (fiber + 0.1),
        'is_high_energy_density':  int(kcal > 400),
        'sugar_AND_no_fiber':      sugar * (1 / (fiber + 0.1)),
        'additive_AND_processed':  adds * ultra,
        'salt_AND_fat_bomb':       int(salt > 2 and fat > 20),
        'protein_fiber_bonus':     prot * fiber,
        'sugar_AND_ultra_processed': sugar * ultra,
        'ingredient_count':        ingr_count,
        'processing_score':        adds * 2 + ultra * 3 + ingr_count / 10,
        'macro_balance_score': (
            (prot * 4 + fiber * 2) /
            (sugar * 3 + sat * 2 + salt * 5 + 1)
        ),
        'allergens_count': len(product.get('allergens', [])),
    }

    # Label flags
    labels = ' '.join(product.get('labels', []))
    label_flags = {
        'is_organic':    int('organic' in labels),
        'is_vegan':      int('vegan' in labels),
        'is_gluten_free': int('gluten-free' in labels or 'no-gluten' in labels),
        'is_high_fiber': int('high-fiber' in labels or 'source-of-fiber' in labels),
        'is_wholegrain': int('whole-grain' in labels or 'wholegrain' in labels),
        'is_no_palm_oil': int('no-palm-oil' in labels),
    }

    for col, val in {**derived, **label_flags}.items():
        if col in col_idx:
            vec[0, col_idx[col]] = val

    return vec


def _apply_adjustments(score: float, product: dict) -> tuple:
    """
    Rule-based adjustments applied AFTER ML score.
    Each rule adds or subtracts points and logs a human-readable reason.
    Returns (adjusted_score, list_of_adjustment_strings).
    """
    adjustments = []

    def adjust(delta, reason):
        nonlocal score
        score = max(0, min(100, score + delta))
        sign = '+' if delta >= 0 else ''
        adjustments.append(f"{sign}{delta:.0f}  {reason}")

    # Additives
    n_additives = len(product.get('additives', []))
    if n_additives >= 8:
        adjust(-8, f"{n_additives} additives detected")
    elif n_additives >= 4:
        adjust(-4, f"{n_additives} additives detected")

    # Palm oil
    ingr = (product.get('ingredients', '') or '').lower()
    if 'palm oil' in ingr:
        adjust(-3, "Contains palm oil")

    # NOVA group
    nova = product.get('nova_group')
    if nova == 4:
        adjust(-5, "NOVA 4 — ultra-processed")
    elif nova == 1:
        adjust(+5, "NOVA 1 — minimally processed")

    # Nutriscore bonus/penalty
    ns = (product.get('nutriscore') or '').upper()
    ns_map = {'A': +5, 'B': +2, 'D': -3, 'E': -6}
    if ns in ns_map:
        adjust(ns_map[ns], f"Nutri-Score {ns}")

    # High sugar
    sugar = product.get('sugars', 0) or 0
    if sugar > 20:
        adjust(-4, f"High sugar ({sugar:.1f}g/100g)")

    # High salt
    salt = product.get('salt', 0) or 0
    if salt > 2:
        adjust(-3, f"High salt ({salt:.1f}g/100g)")

    # Good fiber
    fiber = product.get('fiber', 0) or 0
    if fiber >= 6:
        adjust(+4, f"High fiber ({fiber:.1f}g/100g)")
    elif fiber >= 3:
        adjust(+2, f"Good fiber ({fiber:.1f}g/100g)")

    # Good protein
    prot = product.get('proteins', 0) or 0
    if prot >= 15:
        adjust(+3, f"High protein ({prot:.1f}g/100g)")

    # Label bonuses
    labels = ' '.join(product.get('labels', []))
    if 'organic' in labels:
        adjust(+2, "Certified organic")
    if 'no-palm-oil' in labels:
        adjust(+2, "No palm oil")

    return round(score, 1), adjustments


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
    """Return per-nutrient contribution labels for display."""
    sugar = product.get('sugars', 0) or 0
    sat   = product.get('saturated_fat', 0) or 0
    salt  = product.get('salt', 0) or 0
    fiber = product.get('fiber', 0) or 0
    prot  = product.get('proteins', 0) or 0
    adds  = len(product.get('additives', []))
    nova  = product.get('nova_group') or '?'

    return {
        'sugar':     ('🔴 High' if sugar > 20 else '🟡 Medium' if sugar > 8 else '🟢 Low'),
        'sat_fat':   ('🔴 High' if sat > 5    else '🟡 Medium' if sat > 2   else '🟢 Low'),
        'salt':      ('🔴 High' if salt > 2   else '🟡 Medium' if salt > 1  else '🟢 Low'),
        'fiber':     ('🟢 High' if fiber >= 6 else '🟡 Medium' if fiber >= 3 else '🔴 Low'),
        'protein':   ('🟢 High' if prot >= 15 else '🟡 Medium' if prot >= 7  else '🔴 Low'),
        'additives': ('🔴 Many' if adds >= 8  else '🟡 Some'   if adds >= 3  else '🟢 Few'),
        'nova':      nova,
    }