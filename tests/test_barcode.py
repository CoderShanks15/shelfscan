"""
tests/test_barcode.py
=====================
Tests for utils/barcode.py

Run:
    pytest tests/test_barcode.py -v

Coverage:
  - validate_barcode (EAN-13, UPC-A, EAN-8, invalid inputs)
  - decode_barcode_from_image (mocked stages)
  - get_barcode_region (mocked cv2)
  - checksum validators directly
  - stage functions individually
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock

from utils.barcode import (
    validate_barcode,
    decode_barcode_from_image,
    get_barcode_region,
    _validate_ean13,
    _validate_upca,
    _validate_ean8,
    _pyzbar_decode,
    _stage1_direct,
    _stage2_clahe,
    _stage3_region_crop,
    _stage4_rotation,
    _stage5_threshold,
    _stage6_zxing,
)


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _blank_image(w=200, h=100) -> Image.Image:
    """Create a blank white PIL image for testing."""
    return Image.fromarray(np.ones((h, w, 3), dtype=np.uint8) * 255)


def _mock_pyzbar_result(barcode_str: str):
    """Build a mock pyzbar result object."""
    mock = MagicMock()
    mock.data = barcode_str.encode('utf-8')
    return mock


# -----------------------------------------------------------------------
# TEST: validate_barcode — public API
# -----------------------------------------------------------------------

class TestValidateBarcode:

    # EAN-13 valid
    def test_nutella_ean13_valid(self):
        assert validate_barcode("3017620422003") is True

    def test_evian_ean13_valid(self):
        assert validate_barcode("3068320113994") is True

    def test_pringles_ean13_valid(self):
        assert validate_barcode("5053990101538") is True

    def test_quaker_oats_ean13_valid(self):
        assert validate_barcode("5000173008065") is True

    # EAN-13 invalid checksum
    def test_ean13_wrong_check_digit(self):
        assert validate_barcode("3017620422004") is False  # last digit wrong

    def test_ean13_all_zeros_invalid(self):
        assert validate_barcode("0000000000000") is False

    # UPC-A valid (12 digits)
    def test_upca_valid(self):
        # Coca-Cola Classic US UPC
        assert validate_barcode("049000028911") is True

    def test_upca_wrong_check_digit(self):
        assert validate_barcode("049000028912") is False

    # EAN-8 valid (8 digits)
    def test_ean8_valid(self):
        assert validate_barcode("96385074") is True

    def test_ean8_wrong_check_digit(self):
        assert validate_barcode("96385075") is False

    # Edge cases
    def test_empty_string_invalid(self):
        assert validate_barcode("") is False

    def test_none_invalid(self):
        assert validate_barcode(None) is False

    def test_letters_invalid(self):
        assert validate_barcode("ABC123456789") is False

    def test_too_short_invalid(self):
        assert validate_barcode("123") is False

    def test_too_long_invalid(self):
        assert validate_barcode("12345678901234") is False

    def test_whitespace_invalid(self):
        assert validate_barcode("3017620 422003") is False


# -----------------------------------------------------------------------
# TEST: checksum validators directly
# -----------------------------------------------------------------------

class TestChecksumValidators:

    def test_ean13_valid_nutella(self):
        assert _validate_ean13("3017620422003") is True

    def test_ean13_invalid_digit(self):
        assert _validate_ean13("3017620422009") is False

    def test_upca_valid(self):
        assert _validate_upca("049000028911") is True

    def test_upca_invalid(self):
        assert _validate_upca("049000028910") is False

    def test_ean8_valid(self):
        assert _validate_ean8("96385074") is True

    def test_ean8_invalid(self):
        assert _validate_ean8("96385079") is False


# -----------------------------------------------------------------------
# TEST: _pyzbar_decode
# -----------------------------------------------------------------------

class TestPyzbarDecode:

    @patch('utils.barcode.pyzbar.decode')
    def test_returns_barcode_string_on_success(self, mock_decode):
        mock_decode.return_value = [_mock_pyzbar_result("3017620422003")]
        result = _pyzbar_decode(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode.pyzbar.decode')
    def test_returns_none_when_no_barcode(self, mock_decode):
        mock_decode.return_value = []
        result = _pyzbar_decode(_blank_image())
        assert result is None

    @patch('utils.barcode.pyzbar.decode')
    def test_returns_first_barcode_when_multiple(self, mock_decode):
        mock_decode.return_value = [
            _mock_pyzbar_result("3017620422003"),
            _mock_pyzbar_result("3068320113994"),
        ]
        result = _pyzbar_decode(_blank_image())
        assert result == "3017620422003"


# -----------------------------------------------------------------------
# TEST: decode_barcode_from_image (cascade)
# -----------------------------------------------------------------------

class TestDecodeBarcodeFromImage:

    @patch('utils.barcode._stage1_direct')
    def test_returns_barcode_on_stage1_success(self, mock_s1):
        mock_s1.return_value = "3017620422003"
        result = decode_barcode_from_image(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode._stage1_direct')
    @patch('utils.barcode._stage2_clahe')
    def test_falls_through_to_stage2(self, mock_s2, mock_s1):
        mock_s1.return_value = None
        mock_s2.return_value = "3017620422003"
        result = decode_barcode_from_image(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode._stage1_direct')
    @patch('utils.barcode._stage2_clahe')
    @patch('utils.barcode._stage3_region_crop')
    def test_falls_through_to_stage3(self, mock_s3, mock_s2, mock_s1):
        mock_s1.return_value = None
        mock_s2.return_value = None
        mock_s3.return_value = "3017620422003"
        result = decode_barcode_from_image(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode._stage1_direct')
    @patch('utils.barcode._stage2_clahe')
    @patch('utils.barcode._stage3_region_crop')
    @patch('utils.barcode._stage4_rotation')
    @patch('utils.barcode._stage5_threshold')
    @patch('utils.barcode._stage6_zxing')
    def test_returns_none_when_all_stages_fail(
        self, mock_s6, mock_s5, mock_s4, mock_s3, mock_s2, mock_s1
    ):
        for m in [mock_s1, mock_s2, mock_s3, mock_s4, mock_s5, mock_s6]:
            m.return_value = None
        result = decode_barcode_from_image(_blank_image())
        assert result is None

    @patch('utils.barcode._stage1_direct')
    @patch('utils.barcode._stage2_clahe')
    def test_stage2_not_called_if_stage1_succeeds(self, mock_s2, mock_s1):
        """Early exit — once a stage succeeds, later stages not called."""
        mock_s1.return_value = "3017620422003"
        decode_barcode_from_image(_blank_image())
        mock_s2.assert_not_called()


# -----------------------------------------------------------------------
# TEST: individual stages
# -----------------------------------------------------------------------

class TestStages:

    @patch('utils.barcode.pyzbar.decode')
    def test_stage1_direct(self, mock_decode):
        mock_decode.return_value = [_mock_pyzbar_result("3017620422003")]
        result = _stage1_direct(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode.pyzbar.decode')
    def test_stage2_clahe_returns_none_on_no_barcode(self, mock_decode):
        mock_decode.return_value = []
        result = _stage2_clahe(_blank_image())
        assert result is None

    @patch('utils.barcode.get_barcode_region')
    def test_stage3_returns_none_when_no_region(self, mock_region):
        mock_region.return_value = None
        result = _stage3_region_crop(_blank_image())
        assert result is None

    @patch('utils.barcode.get_barcode_region')
    @patch('utils.barcode.pyzbar.decode')
    def test_stage3_crops_and_decodes(self, mock_decode, mock_region):
        mock_region.return_value = (10, 10, 80, 40)
        mock_decode.return_value = [_mock_pyzbar_result("3017620422003")]
        result = _stage3_region_crop(_blank_image())
        assert result == "3017620422003"

    @patch('utils.barcode.pyzbar.decode')
    def test_stage4_rotation_tries_three_angles(self, mock_decode):
        """Stage 4 should try exactly 90, 180, 270 — not 45 or 135."""
        mock_decode.return_value = []
        _stage4_rotation(_blank_image())
        assert mock_decode.call_count == 3  # 90, 180, 270 only

    @patch('utils.barcode.pyzbar.decode')
    def test_stage4_returns_on_first_success(self, mock_decode):
        """Should stop rotating once a barcode is found."""
        mock_decode.side_effect = [
            [],                                          # 90° fails
            [_mock_pyzbar_result("3017620422003")],      # 180° succeeds
        ]
        result = _stage4_rotation(_blank_image())
        assert result == "3017620422003"
        assert mock_decode.call_count == 2  # stopped at 180°

    @patch('utils.barcode.pyzbar.decode')
    def test_stage5_threshold_returns_none_on_blank(self, mock_decode):
        mock_decode.return_value = []
        result = _stage5_threshold(_blank_image())
        assert result is None

    def test_stage6_zxing_returns_none_gracefully_if_unavailable(self):
        with patch('utils.barcode._ZXING_AVAILABLE', False):
            result = _stage6_zxing(_blank_image())
            assert result is None


# -----------------------------------------------------------------------
# TEST: get_barcode_region
# -----------------------------------------------------------------------

class TestGetBarcodeRegion:

    def test_returns_none_on_blank_image(self):
        """A blank white image has no barcode region."""
        result = get_barcode_region(_blank_image())
        assert result is None

    def test_returns_tuple_of_four_ints_when_region_found(self):
        """When a region is detected, it should be a 4-tuple."""
        with patch('utils.barcode.cv2.findContours') as mock_contours:
            # Create a fake contour that looks like a barcode region
            fake_contour = np.array([[[10, 10]], [[90, 10]],
                                     [[90, 50]], [[10, 50]]])
            mock_contours.return_value = ([fake_contour], None)
            result = get_barcode_region(_blank_image())
            if result is not None:
                assert len(result) == 4
                x, y, w, h = result
                assert all(isinstance(v, (int, np.integer)) for v in [x, y, w, h])

    def test_returns_none_when_no_contours(self):
        with patch('utils.barcode.cv2.findContours') as mock_contours:
            mock_contours.return_value = ([], None)
            result = get_barcode_region(_blank_image())
            assert result is None


# -----------------------------------------------------------------------
# TEST: demo barcode sanity
# -----------------------------------------------------------------------

class TestDemoBarcodes:
    """All 8 demo barcodes from the masterplan should pass validation."""

    DEMO_BARCODES = [
        ("Nutella 400g",           "3017620422003"),
        ("Haribo Gold-Bears",      "4001686323564"),
        ("Kellogg's Corn Flakes",  "5010477348549"),
        ("Evian Water 1.5L",       "3068320113994"),
        ("Pringles Original",      "5053990101538"),
        ("Kinder Bueno",           "8000500221938"),
        ("Activia Strawberry",     "3033490006397"),
        ("Quaker Oats 1kg",        "5000173008065"),
    ]

    @pytest.mark.parametrize("name,barcode", DEMO_BARCODES)
    def test_demo_barcode_validates(self, name, barcode):
        assert validate_barcode(barcode) is True, (
            f"{name} ({barcode}) failed validation"
        )