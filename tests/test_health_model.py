"""
tests/test_health_model.py
==========================
Tests for ml/predict.py — the inference module.

All tests mock the model so no .pkl file is needed to run.

Run:
    pytest tests/test_health_model.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# -----------------------------------------------------------------------
# MOCK MODEL — patches _load so no .pkl file needed
# -----------------------------------------------------------------------

def _mock_load():
    """
    Replace _load() with a mock that sets up a fake model.
    The fake model returns a score proportional to protein and inversely
    proportional to sugar — enough to test sanity checks.
    """
    import ml.predict as mod

    if mod._model is not None:
        return

    mock_model = MagicMock()

    # Simple scoring: higher protein → higher score, higher sugar → lower
    def mock_predict(vec):
        # Use a simplistic formula based on known feature positions
        # Just return a reasonable value based on the sum of the vector
        total = float(np.sum(vec))
        # Clamp to a reasonable range
        return np.array([max(10, min(90, 50 + total * 0.01))])

    mock_model.predict = mock_predict

    # Set up module state
    # Use a small set of known feature names
    feature_names = [
        'energy_kcal', 'fat', 'saturated_fat', 'carbohydrates', 'sugars',
        'fiber', 'proteins', 'salt', 'sodium', 'trans_fat',
        'mono_fat', 'poly_fat', 'omega3', 'omega6',
        'additives_count', 'additive_risk_score', 'additives_high_risk',
        'nova_score', 'ultra_processed_indicator', 'processing_score',
        'ingredient_count', 'concerning_ingredient_count', 'allergens_count',
        'traces_count', 'palm_oil_risk',
        'fiber_sugar_ratio', 'protein_fat_ratio', 'protein_energy_ratio',
        'fat_quality_ratio', 'omega_balance', 'carb_fiber_balance',
        'energy_density', 'sodium_salt_ratio', 'is_high_energy_density',
        'omega3_sat_ratio',
        'sugar_AND_no_fiber', 'sugar_AND_ultra_processed',
        'additive_AND_processed', 'salt_AND_fat_bomb',
        'protein_fiber_bonus', 'high_fat_low_quality', 'macro_balance_score',
        'is_organic', 'is_vegan', 'is_vegetarian', 'is_gluten_free',
        'is_wholegrain', 'is_no_palm_oil', 'is_high_fiber',
        'is_no_salt', 'is_raw', 'is_fortified',
        'sugar_in_top3', 'fat_in_top3', 'salt_in_top3',
        'whole_food_score',
    ]

    mod._model = mock_model
    mod._selected_feature_names = feature_names
    mod._col_idx = {c: i for i, c in enumerate(feature_names)}


@pytest.fixture(autouse=True)
def mock_model():
    """Auto-mock the model for all tests."""
    import ml.predict as mod
    # Reset module state
    mod._model = None
    mod._selected_feature_names = None
    mod._col_idx = None

    with patch('ml.predict._load', side_effect=_mock_load):
        yield


# -----------------------------------------------------------------------
# FIXTURES — product dicts matching services/api.py output
# -----------------------------------------------------------------------

NUTELLA = {
    "barcode":       "3017620422003",
    "name":          "Nutella",
    "brand":         "Ferrero",
    "quantity":      "400g",
    "nutriscore":    "E",
    "nova_group":    4,
    "energy_kcal":   539,
    "fat":           30.9,
    "saturated_fat": 10.6,
    "carbohydrates": 57.5,
    "sugars":        56.3,
    "fiber":         0,
    "proteins":      6.3,
    "salt":          0.107,
    "allergens":     ["en:gluten", "en:milk"],
    "additives":     ["en:e322"],
    "labels":        [],
    "ingredients":   "Sugar, palm oil, hazelnuts, cocoa, skimmed milk powder",
}

WATER = {
    "barcode":       "3068320113994",
    "name":          "Evian Natural Mineral Water",
    "brand":         "Evian",
    "quantity":      "1.5L",
    "nutriscore":    "A",
    "nova_group":    1,
    "energy_kcal":   0,
    "fat":           0,
    "saturated_fat": 0,
    "carbohydrates": 0,
    "sugars":        0,
    "fiber":         0,
    "proteins":      0,
    "salt":          0,
    "allergens":     [],
    "additives":     [],
    "labels":        ["en:no-additives"],
    "ingredients":   "Natural mineral water",
}

OATS = {
    "barcode":       "5000173008065",
    "name":          "Quaker Oats",
    "brand":         "Quaker",
    "quantity":      "1kg",
    "nutriscore":    "A",
    "nova_group":    1,
    "energy_kcal":   375,
    "fat":           8.0,
    "saturated_fat": 1.5,
    "carbohydrates": 60,
    "sugars":        1.0,
    "fiber":         9.0,
    "proteins":      11.0,
    "salt":          0.01,
    "allergens":     ["en:gluten"],
    "additives":     [],
    "labels":        ["en:organic", "en:wholegrain"],
    "ingredients":   "100% wholegrain oats",
}

MAYONNAISE = {
    "barcode":       "8909106029842",
    "name":          "Smoky Tandoori Mayonnaise",
    "brand":         "HUL",
    "quantity":      "250g",
    "nutriscore":    "",
    "nova_group":    4,
    "energy_kcal":   680,
    "fat":           70.0,
    "saturated_fat": 10.0,
    "carbohydrates": 2.0,
    "sugars":        1.5,
    "fiber":         0,
    "proteins":      1.0,
    "salt":          1.5,
    "allergens":     ["en:eggs"],
    "additives":     ["en:e385", "en:e211", "en:e202", "en:e330"],
    "labels":        [],
    "ingredients":   "Soybean oil, water, egg yolk, sugar, salt, spices",
}

EMPTY_PRODUCT = {
    "barcode":       "0000000000000",
    "name":          "Unknown Product",
    "brand":         "Unknown Brand",
}


# -----------------------------------------------------------------------
# TEST: predict_health return shape
# -----------------------------------------------------------------------

class TestPredictHealthShape:

    def test_returns_dict(self):
        result = predict_health(NUTELLA)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = predict_health(NUTELLA)
        for key in ['score', 'verdict', 'verdict_emoji', 'adjustments', 'breakdown']:
            assert key in result, f"Missing key: {key}"

    def test_score_is_float(self):
        result = predict_health(NUTELLA)
        assert isinstance(result['score'], float)

    def test_verdict_is_string(self):
        result = predict_health(NUTELLA)
        assert isinstance(result['verdict'], str)

    def test_adjustments_is_list(self):
        result = predict_health(NUTELLA)
        assert isinstance(result['adjustments'], list)

    def test_breakdown_is_dict(self):
        result = predict_health(NUTELLA)
        assert isinstance(result['breakdown'], dict)


# -----------------------------------------------------------------------
# TEST: score range
# -----------------------------------------------------------------------

class TestScoreRange:

    def test_nutella_score_in_range(self):
        result = predict_health(NUTELLA)
        assert 0 <= result['score'] <= 100

    def test_water_score_in_range(self):
        result = predict_health(WATER)
        assert 0 <= result['score'] <= 100

    def test_empty_product_score_in_range(self):
        result = predict_health(EMPTY_PRODUCT)
        assert 0 <= result['score'] <= 100


# -----------------------------------------------------------------------
# TEST: verdict values
# -----------------------------------------------------------------------

class TestVerdicts:

    def test_verdict_is_valid_string(self):
        for product in [NUTELLA, WATER, OATS, EMPTY_PRODUCT]:
            result = predict_health(product)
            assert result['verdict'] in ('Avoid', 'Caution', 'OK', 'Great')

    def test_verdict_emoji_is_non_empty(self):
        result = predict_health(NUTELLA)
        assert len(result['verdict_emoji']) > 0


# -----------------------------------------------------------------------
# TEST: rule-based adjustments
# -----------------------------------------------------------------------

class TestAdjustments:

    def test_nutella_gets_high_sugar_penalty(self):
        """Nutella has 56.3g sugar — should trigger high sugar adjustment."""
        result = predict_health(NUTELLA)
        adj_text = " ".join(result['adjustments'])
        assert "sugar" in adj_text.lower() or "Sugar" in adj_text

    def test_nutella_gets_nova4_penalty(self):
        """Nutella is NOVA 4 — should trigger ultra-processed penalty."""
        result = predict_health(NUTELLA)
        adj_text = " ".join(result['adjustments'])
        assert "nova" in adj_text.lower() or "NOVA" in adj_text

    def test_nutella_gets_palm_oil_penalty(self):
        """Nutella contains palm oil."""
        result = predict_health(NUTELLA)
        adj_text = " ".join(result['adjustments'])
        assert "palm oil" in adj_text.lower()

    def test_nutella_gets_nutriscore_e_penalty(self):
        """Nutri-Score E should apply negative adjustment."""
        result = predict_health(NUTELLA)
        adj_text = " ".join(result['adjustments'])
        assert "nutri-score" in adj_text.lower() or "Nutri-Score" in adj_text

    def test_water_gets_nova1_bonus(self):
        """Water is NOVA 1 — should get a bonus."""
        result = predict_health(WATER)
        adj_text = " ".join(result['adjustments'])
        assert "nova" in adj_text.lower() or "NOVA" in adj_text

    def test_water_gets_nutriscore_a_bonus(self):
        """Water has Nutri-Score A."""
        result = predict_health(WATER)
        adj_text = " ".join(result['adjustments'])
        assert "nutri-score" in adj_text.lower() or "Nutri-Score" in adj_text

    def test_oats_gets_fiber_bonus(self):
        """Oats have 9g fiber — should trigger high fiber bonus."""
        result = predict_health(OATS)
        adj_text = " ".join(result['adjustments'])
        assert "fiber" in adj_text.lower() or "fibre" in adj_text.lower()

    def test_oats_gets_organic_bonus(self):
        """Oats are labelled organic."""
        result = predict_health(OATS)
        adj_text = " ".join(result['adjustments'])
        assert "organic" in adj_text.lower()

    def test_oats_gets_wholegrain_bonus(self):
        result = predict_health(OATS)
        adj_text = " ".join(result['adjustments'])
        assert "wholegrain" in adj_text.lower() or "whole" in adj_text.lower()

    # ---- Mayonnaise penalty tests (NEW) ----

    def test_mayonnaise_gets_high_fat_penalty(self):
        """Mayonnaise has 70g fat — should trigger very high fat adjustment."""
        result = predict_health(MAYONNAISE)
        adj_text = " ".join(result['adjustments'])
        assert "fat" in adj_text.lower()

    def test_mayonnaise_gets_sat_fat_penalty(self):
        """Mayonnaise has 10g saturated fat — should trigger sat fat penalty."""
        result = predict_health(MAYONNAISE)
        adj_text = " ".join(result['adjustments'])
        assert "saturated" in adj_text.lower()

    def test_mayonnaise_gets_energy_penalty(self):
        """Mayonnaise has 680 kcal — should trigger energy density penalty."""
        result = predict_health(MAYONNAISE)
        adj_text = " ".join(result['adjustments'])
        assert "calorie" in adj_text.lower() or "energy" in adj_text.lower()

    def test_mayonnaise_score_below_50(self):
        """After all adjustments, mayonnaise must score ≤ 50."""
        result = predict_health(MAYONNAISE)
        assert result['score'] <= 50, (
            f"Mayonnaise scored {result['score']} — should be ≤ 50"
        )


# -----------------------------------------------------------------------
# TEST: breakdown format
# -----------------------------------------------------------------------

class TestBreakdown:

    def test_breakdown_has_expected_keys(self):
        result = predict_health(NUTELLA)
        bd = result['breakdown']
        for key in ['sugar', 'fat', 'sat_fat', 'salt', 'energy', 'fiber', 'protein', 'additives', 'nova']:
            assert key in bd, f"Missing breakdown key: {key}"

    def test_breakdown_values_contain_emoji(self):
        """Each breakdown value should start with a traffic light emoji."""
        result = predict_health(NUTELLA)
        bd = result['breakdown']
        for key in ['sugar', 'sat_fat', 'salt', 'fiber', 'protein', 'additives']:
            assert bd[key][0] in ('🔴', '🟡', '🟢'), \
                f"Breakdown '{key}' missing traffic light emoji: {bd[key]}"

    def test_nutella_sugar_is_high(self):
        result = predict_health(NUTELLA)
        assert "High" in result['breakdown']['sugar']

    def test_water_sugar_is_low(self):
        result = predict_health(WATER)
        assert "Low" in result['breakdown']['sugar']


# -----------------------------------------------------------------------
# TEST: edge cases
# -----------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_product_does_not_crash(self):
        """Minimal product dict should not raise."""
        result = predict_health(EMPTY_PRODUCT)
        assert 'score' in result

    def test_none_nutrients_handled(self):
        """None values in nutrient fields should not crash."""
        product = {
            "name": "Test",
            "energy_kcal": None,
            "fat": None,
            "sugars": None,
            "fiber": None,
            "proteins": None,
            "salt": None,
            "saturated_fat": None,
        }
        result = predict_health(product)
        assert 0 <= result['score'] <= 100

    def test_negative_nutrients_clamped(self):
        """Negative nutrient values (data errors) should not crash."""
        product = {
            "name": "Bad Data",
            "energy_kcal": -100,
            "fat": -5,
            "sugars": -10,
        }
        result = predict_health(product)
        assert 0 <= result['score'] <= 100
