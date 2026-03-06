"""
tests/test_price.py
===================
Tests for ml/price_intelligence.py

Run:
    pytest tests/test_price.py -v
"""

import pytest
from ml.price_intelligence import (
    PriceIntelligence,
    _nutrition_density,
    _parse_weight_g,
    _safe,
)


# -----------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------

NUTELLA = {
    "name":          "Nutella",
    "barcode":       "3017620422003",
    "brand":         "Ferrero",
    "quantity":      "400g",
    "proteins":      6.3,
    "fiber":         0,
    "sugars":        56.3,
    "saturated_fat": 10.6,
    "salt":          0.107,
    "nova_group":    4,
    "additives":     ["en:e322"],
    "labels":        [],
    "ingredients":   "Sugar, palm oil, hazelnuts, cocoa, skimmed milk powder",
}

WATER = {
    "name":          "Evian Water",
    "barcode":       "3068320113994",
    "quantity":      "1.5L",
    "proteins":      0,
    "fiber":         0,
    "sugars":        0,
    "saturated_fat": 0,
    "salt":          0,
    "nova_group":    1,
    "additives":     [],
    "labels":        ["en:no-additives"],
    "ingredients":   "Natural mineral water",
}

OATS = {
    "name":          "Quaker Oats",
    "barcode":       "5000173008065",
    "quantity":      "1kg",
    "proteins":      11.0,
    "fiber":         9.0,
    "sugars":        1.0,
    "saturated_fat": 1.5,
    "salt":          0.01,
    "nova_group":    1,
    "additives":     [],
    "labels":        ["en:organic", "en:wholegrain"],
    "ingredients":   "100% wholegrain oats",
}


@pytest.fixture
def pi():
    """Fresh PriceIntelligence instance."""
    return PriceIntelligence()


# -----------------------------------------------------------------------
# TEST: analyze() output shape
# -----------------------------------------------------------------------

class TestOutputShape:

    def test_returns_dict(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        assert isinstance(result, dict)

    def test_has_required_keys(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        required = [
            'health_per_penny', 'nutrition_density', 'category_benchmark',
            'category', 'vs_benchmark', 'value_verdict', 'value_emoji',
            'explanation', 'estimated_price', 'estimated_pack_price',
            'price_band',
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_nutrition_density_is_float(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        assert isinstance(result['nutrition_density'], float)

    def test_value_verdict_is_string(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        assert isinstance(result['value_verdict'], str)


# -----------------------------------------------------------------------
# TEST: nutrition density
# -----------------------------------------------------------------------

class TestNutritionDensity:

    def test_density_in_range(self):
        for product in [NUTELLA, WATER, OATS]:
            d = _nutrition_density(product)
            assert 0 <= d <= 100, f"Density {d} out of range for {product['name']}"

    def test_oats_more_dense_than_nutella(self):
        """Oats (high protein, high fiber) should score higher than Nutella."""
        assert _nutrition_density(OATS) > _nutrition_density(NUTELLA)

    def test_water_higher_than_nutella(self):
        """Water has zero negatives, Nutella has high sugar+fat."""
        assert _nutrition_density(WATER) > _nutrition_density(NUTELLA)

    def test_empty_product_returns_non_negative(self):
        d = _nutrition_density({})
        assert d >= 0

    def test_none_nutrients_handled(self):
        product = {"proteins": None, "fiber": None, "sugars": None}
        d = _nutrition_density(product)
        assert d >= 0

    def test_high_protein_increases_density(self):
        low  = _nutrition_density({"proteins": 2, "fiber": 0, "sugars": 0,
                                    "saturated_fat": 0, "salt": 0})
        high = _nutrition_density({"proteins": 20, "fiber": 0, "sugars": 0,
                                    "saturated_fat": 0, "salt": 0})
        assert high > low

    def test_high_sugar_decreases_density(self):
        low_sugar  = _nutrition_density({"sugars": 2, "proteins": 5,
                                          "saturated_fat": 0, "salt": 0})
        high_sugar = _nutrition_density({"sugars": 40, "proteins": 5,
                                          "saturated_fat": 0, "salt": 0})
        assert low_sugar > high_sugar


# -----------------------------------------------------------------------
# TEST: weight parsing
# -----------------------------------------------------------------------

class TestWeightParsing:

    def test_grams(self):
        assert _parse_weight_g({"quantity": "400g"}) == 400.0

    def test_kilograms(self):
        assert _parse_weight_g({"quantity": "1.5kg"}) == 1500.0

    def test_litres(self):
        assert _parse_weight_g({"quantity": "1.5L"}) == 1500.0

    def test_millilitres(self):
        assert _parse_weight_g({"quantity": "330ml"}) == 330.0

    def test_weight_g_field_preferred(self):
        p = {"weight_g": 500, "quantity": "400g"}
        assert _parse_weight_g(p) == 500.0

    def test_fallback_to_100(self):
        assert _parse_weight_g({}) == 100.0


# -----------------------------------------------------------------------
# TEST: health_per_penny
# -----------------------------------------------------------------------

class TestHealthPerPenny:

    def test_hpp_is_non_negative(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        if result['health_per_penny'] is not None:
            assert result['health_per_penny'] >= 0

    def test_hpp_calculated_from_estimated_price(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        # HPP should exist since we always estimate a price
        assert result['health_per_penny'] is not None


# -----------------------------------------------------------------------
# TEST: category detection
# -----------------------------------------------------------------------

class TestCategoryDetection:

    def test_uses_provided_category(self, pi):
        result = pi.analyze(WATER, 88.0, category="water")
        assert result['category'] == "water"

    def test_detects_category_from_name(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        # Should detect nutella as spreads via keyword
        assert result['category'] in ('spreads', 'chocolate', 'confectionery', 'snacks', 'default')


# -----------------------------------------------------------------------
# TEST: safe helper
# -----------------------------------------------------------------------

class TestSafe:

    def test_float_passthrough(self):
        assert _safe(3.14) == 3.14

    def test_int_to_float(self):
        assert _safe(5) == 5.0

    def test_none_returns_default(self):
        assert _safe(None) == 0.0

    def test_string_returns_default(self):
        assert _safe("abc") == 0.0

    def test_custom_default(self):
        assert _safe(None, -1.0) == -1.0


# -----------------------------------------------------------------------
# TEST: explanation
# -----------------------------------------------------------------------

class TestExplanation:

    def test_explanation_contains_product_name(self, pi):
        result = pi.analyze(NUTELLA, 28.0)
        assert "Nutella" in result['explanation']

    def test_explanation_is_non_empty(self, pi):
        result = pi.analyze(OATS, 74.0)
        assert len(result['explanation']) > 10
