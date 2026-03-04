"""
tests/test_api.py
=================
Tests for utils/api.py

All network calls are mocked — tests run fully offline.
No real API calls made.

Run:
    pytest tests/test_api.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

# -----------------------------------------------------------------------
# Mock streamlit cache decorator BEFORE importing api.py
# Without this, @st.cache_data fails outside a Streamlit runtime.
# -----------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

# Stub streamlit so cache decorator is a no-op
st_mock = MagicMock()
st_mock.cache_data = lambda **kwargs: (lambda f: f)
sys.modules['streamlit'] = st_mock

from utils.api import fetch_product, _clean_product, search_products, get_product_image_url


# -----------------------------------------------------------------------
# FIXTURES — reusable mock data
# -----------------------------------------------------------------------

NUTELLA_RAW = {
    "code": "3017620422003",
    "product_name": "Nutella",
    "brands": "Ferrero",
    "quantity": "400g",
    "nutriscore_grade": "e",
    "nova_group": 4,
    "ecoscore_grade": "c",
    "nutriments": {
        "energy-kcal_100g": 539,
        "fat_100g": 30.9,
        "saturated-fat_100g": 10.6,
        "carbohydrates_100g": 57.5,
        "sugars_100g": 56.3,
        "fiber_100g": 0,
        "proteins_100g": 6.3,
        "salt_100g": 0.107,
    },
    "allergens_tags": ["en:gluten", "en:milk"],
    "additives_tags": ["en:e322"],
    "labels_tags": [],
    "ingredients_text": "Sugar, palm oil, hazelnuts, cocoa, skimmed milk powder",
    "image_front_url": "https://images.openfoodfacts.org/nutella.jpg",
    "stores": "Tesco, Sainsbury's",
}

WATER_RAW = {
    "code": "3068320113994",
    "product_name": "Evian Natural Mineral Water",
    "brands": "Evian",
    "quantity": "1.5L",
    "nutriscore_grade": "a",
    "nova_group": 1,
    "ecoscore_grade": "b",
    "nutriments": {
        "energy-kcal_100g": 0,
        "fat_100g": 0,
        "saturated-fat_100g": 0,
        "carbohydrates_100g": 0,
        "sugars_100g": 0,
        "fiber_100g": 0,
        "proteins_100g": 0,
        "salt_100g": 0,
    },
    "allergens_tags": [],
    "additives_tags": [],
    "labels_tags": ["en:no-additives"],
    "ingredients_text": "Natural mineral water",
    "image_front_url": "https://images.openfoodfacts.org/evian.jpg",
    "stores": "Tesco",
}

def _make_response(raw_product, status=1):
    """Build a mock requests.Response for a given product dict."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "status": status,
        "product": raw_product,
    }
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# -----------------------------------------------------------------------
# TEST: _clean_product
# Unit tests — no network, no mocking needed.
# -----------------------------------------------------------------------

class TestCleanProduct:

    def test_basic_fields_extracted(self):
        result = _clean_product(NUTELLA_RAW)
        assert result['name']   == "Nutella"
        assert result['brand']  == "Ferrero"
        assert result['barcode'] == "3017620422003"

    def test_nutriscore_uppercased(self):
        result = _clean_product(NUTELLA_RAW)
        assert result['nutriscore'] == "E"

    def test_nova_group_preserved(self):
        result = _clean_product(NUTELLA_RAW)
        assert result['nova_group'] == 4

    def test_nutrients_are_floats(self):
        result = _clean_product(NUTELLA_RAW)
        assert isinstance(result['fat'], float)
        assert result['fat'] == pytest.approx(30.9)
        assert result['sugars'] == pytest.approx(56.3)

    def test_missing_nutrient_returns_none(self):
        """Missing nutrient should return None, not 0."""
        raw = {**NUTELLA_RAW}
        raw['nutriments'] = {}   # empty nutriments
        result = _clean_product(raw)
        assert result['energy_kcal'] is None
        assert result['fat'] is None

    def test_allergens_list(self):
        result = _clean_product(NUTELLA_RAW)
        assert "en:gluten" in result['allergens']
        assert "en:milk"   in result['allergens']

    def test_missing_allergens_returns_empty_list(self):
        raw = {**NUTELLA_RAW, 'allergens_tags': None}
        result = _clean_product(raw)
        assert result['allergens'] == []

    def test_missing_product_name_fallback(self):
        raw = {**NUTELLA_RAW, 'product_name': None}
        result = _clean_product(raw)
        assert result['name'] == "Unknown Product"

    def test_missing_brand_fallback(self):
        raw = {**NUTELLA_RAW, 'brands': ''}
        result = _clean_product(raw)
        assert result['brand'] == "Unknown Brand"

    def test_water_zero_nutrients_are_float_zero(self):
        """Explicit 0 in nutriments should come back as 0.0, not None."""
        result = _clean_product(WATER_RAW)
        assert result['fat'] == 0.0
        assert result['sugars'] == 0.0


# -----------------------------------------------------------------------
# TEST: fetch_product
# All network calls mocked.
# -----------------------------------------------------------------------

class TestFetchProduct:

    @patch('utils.api._session')
    def test_successful_fetch_returns_ok_true(self, mock_session):
        mock_session.get.return_value = _make_response(NUTELLA_RAW, status=1)
        result = fetch_product("3017620422003")
        assert result['ok'] is True
        assert 'product' in result

    @patch('utils.api._session')
    def test_successful_fetch_product_name(self, mock_session):
        mock_session.get.return_value = _make_response(NUTELLA_RAW, status=1)
        result = fetch_product("3017620422003")
        assert result['product']['name'] == "Nutella"

    @patch('utils.api._session')
    def test_product_not_found_returns_ok_false(self, mock_session):
        """status=0 from OFF means product not in database."""
        mock_session.get.return_value = _make_response(NUTELLA_RAW, status=0)
        result = fetch_product("0000000000000")
        assert result['ok'] is False
        assert 'error' in result
        assert result['retryable'] is False

    @patch('utils.api._session')
    def test_timeout_returns_ok_false_retryable(self, mock_session):
        import requests as req
        mock_session.get.side_effect = req.exceptions.Timeout()
        result = fetch_product("3017620422003")
        assert result['ok'] is False
        assert result['retryable'] is True
        assert "timed out" in result['error'].lower()

    @patch('utils.api._session')
    def test_connection_error_returns_ok_false_retryable(self, mock_session):
        import requests as req
        mock_session.get.side_effect = req.exceptions.ConnectionError()
        result = fetch_product("3017620422003")
        assert result['ok'] is False
        assert result['retryable'] is True
        assert "internet" in result['error'].lower()

    @patch('utils.api._session')
    def test_http_error_returns_ok_false_retryable(self, mock_session):
        import requests as req
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_session.get.side_effect = req.exceptions.HTTPError(
            response=mock_resp
        )
        result = fetch_product("3017620422003")
        assert result['ok'] is False
        assert result['retryable'] is True
        assert "503" in result['error']

    @patch('utils.api._session')
    def test_barcode_stripped_of_whitespace(self, mock_session):
        """Leading/trailing whitespace in barcode should not cause issues."""
        mock_session.get.return_value = _make_response(NUTELLA_RAW, status=1)
        result = fetch_product("  3017620422003  ")
        assert result['ok'] is True

    @patch('utils.api._session')
    def test_water_fetch(self, mock_session):
        mock_session.get.return_value = _make_response(WATER_RAW, status=1)
        result = fetch_product("3068320113994")
        assert result['ok'] is True
        assert result['product']['nova_group'] == 1
        assert result['product']['nutriscore'] == "A"


# -----------------------------------------------------------------------
# TEST: search_products
# -----------------------------------------------------------------------

class TestSearchProducts:

    @patch('utils.api._session')
    def test_successful_search_returns_ok_true(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "products": [NUTELLA_RAW, WATER_RAW]
        }
        mock_session.get.return_value = mock_resp
        result = search_products("nutella")
        assert result['ok'] is True
        assert len(result['products']) == 2

    @patch('utils.api._session')
    def test_search_cleans_each_product(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"products": [NUTELLA_RAW]}
        mock_session.get.return_value = mock_resp
        result = search_products("nutella")
        assert result['products'][0]['name'] == "Nutella"

    @patch('utils.api._session')
    def test_search_timeout_returns_ok_false(self, mock_session):
        import requests as req
        mock_session.get.side_effect = req.exceptions.Timeout()
        result = search_products("nutella")
        assert result['ok'] is False
        assert "timed out" in result['error'].lower()

    @patch('utils.api._session')
    def test_empty_search_results(self, mock_session):
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"products": []}
        mock_session.get.return_value = mock_resp
        result = search_products("xyzxyzxyz_nothing")
        assert result['ok'] is True
        assert result['products'] == []


# -----------------------------------------------------------------------
# TEST: get_product_image_url
# -----------------------------------------------------------------------

class TestGetProductImageUrl:

    @patch('utils.api.fetch_product')
    def test_returns_image_url_on_success(self, mock_fetch):
        mock_fetch.return_value = {
            'ok': True,
            'product': {'image_url': 'https://images.openfoodfacts.org/nutella.jpg'}
        }
        url = get_product_image_url("3017620422003")
        assert url == 'https://images.openfoodfacts.org/nutella.jpg'

    @patch('utils.api.fetch_product')
    def test_returns_none_on_failure(self, mock_fetch):
        mock_fetch.return_value = {'ok': False, 'error': 'Not found'}
        url = get_product_image_url("0000000000000")
        assert url is None

    @patch('utils.api.fetch_product')
    def test_returns_none_when_image_url_empty(self, mock_fetch):
        mock_fetch.return_value = {
            'ok': True,
            'product': {'image_url': ''}
        }
        url = get_product_image_url("3017620422003")
        assert url is None


# -----------------------------------------------------------------------
# TEST: sanity checks — Nutella vs Water
# These mirror the demo script in the masterplan.
# -----------------------------------------------------------------------

class TestDemoSanity:

    def test_nutella_fields_match_expected(self):
        result = _clean_product(NUTELLA_RAW)
        assert result['sugars'] > 50        # should be high sugar
        assert result['nova_group'] == 4    # ultra-processed
        assert result['nutriscore'] == 'E'  # worst grade

    def test_water_fields_match_expected(self):
        result = _clean_product(WATER_RAW)
        assert result['fat'] == 0.0
        assert result['sugars'] == 0.0
        assert result['nova_group'] == 1    # minimally processed
        assert result['nutriscore'] == 'A'  # best grade

    def test_nutella_has_more_additives_than_water(self):
        nutella = _clean_product(NUTELLA_RAW)
        water   = _clean_product(WATER_RAW)
        assert len(nutella['additives']) > len(water['additives'])

    def test_nutella_has_allergens_water_does_not(self):
        nutella = _clean_product(NUTELLA_RAW)
        water   = _clean_product(WATER_RAW)
        assert len(nutella['allergens']) > 0
        assert len(water['allergens']) == 0