"""
utils/barcode.py
================
5-stage cascade barcode decoder for ShelfScan.

Stage order (fastest/most reliable first):
  1. Direct pyzbar decode
  2. CLAHE contrast enhancement
  3. Region crop + decode (uses morphological region detection)
  4. Multi-angle rotation (90°, 180°, 270° only)
  5. Adaptive threshold
  6. ZXing C++ fallback

Design decisions:
  - Each stage is isolated — adding/removing one doesn't affect others
  - _pyzbar_decode is a shared helper to avoid duplicating decode logic
  - get_barcode_region is now used inside the pipeline (Stage 3)
  - 45°/135° rotation removed — black fill corners cause more failures than successes
  - pytesseract removed — was imported but never used
  - Stage that succeeded is returned alongside barcode for debugging
  - ZXing wrapped in try/except — optional dependency, graceful fallback
"""

import logging

import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar

logger = logging.getLogger(__name__)

# Try importing ZXing — optional dependency
try:
    import zxingcpp
    _ZXING_AVAILABLE = True
except ImportError:
    _ZXING_AVAILABLE = False
    logger.warning("zxingcpp not installed — Stage 6 (ZXing fallback) disabled")


# -----------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------

def decode_barcode_from_image(image: Image.Image) -> str | None:
    """
    5-stage cascade barcode decoder.
    Tries each stage in order, returns first successful decode.
    Returns barcode string or None if all stages fail.
    """
    stages = [
        ("Stage 1: Direct",            _stage1_direct),
        ("Stage 2: CLAHE",             _stage2_clahe),
        ("Stage 3: Region crop",       _stage3_region_crop),
        ("Stage 4: Rotation",          _stage4_rotation),
        ("Stage 5: Adaptive threshold",_stage5_threshold),
        ("Stage 6: ZXing fallback",    _stage6_zxing),
    ]

    for stage_name, stage_fn in stages:
        result = stage_fn(image)
        if result:
            logger.info("Barcode decoded via %s → %s", stage_name, result)
            return result

    logger.warning("All stages failed to decode barcode")
    return None


def validate_barcode(barcode: str) -> bool:
    """
    Validate barcode checksum.
    Supports EAN-13, EAN-8, UPC-A (with proper checksum for all three).
    Returns True if valid, False otherwise.
    """
    if not barcode or not barcode.isdigit():
        return False

    if len(barcode) == 13:
        return _validate_ean13(barcode)

    if len(barcode) == 12:
        return _validate_upca(barcode)

    if len(barcode) == 8:
        return _validate_ean8(barcode)

    return False


# -----------------------------------------------------------------------
# STAGE IMPLEMENTATIONS
# -----------------------------------------------------------------------

def _pyzbar_decode(image: Image.Image) -> str | None:
    """Shared helper — run pyzbar on a PIL image, return barcode or None."""
    results = pyzbar.decode(image)
    if results:
        return results[0].data.decode("utf-8")
    return None


def _stage1_direct(image: Image.Image) -> str | None:
    """Stage 1: Direct decode — no preprocessing. Fastest path."""
    return _pyzbar_decode(image)


def _stage2_clahe(image: Image.Image) -> str | None:
    """
    Stage 2: CLAHE histogram equalisation.
    Helps with dark or low-contrast images from bad lighting.
    """
    img_array = np.array(image.convert("L"))
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(img_array)
    return _pyzbar_decode(Image.fromarray(enhanced))


def _stage3_region_crop(image: Image.Image) -> str | None:
    """
    Stage 3: Detect barcode region, crop to it, then decode.
    Significantly improves accuracy on photos with complex backgrounds
    where the barcode is a small part of the full image.
    """
    region = get_barcode_region(image)
    if region is None:
        return None

    x, y, w, h = region

    # Add padding around detected region — improves decode reliability
    padding = 20
    img_w, img_h = image.size
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)

    cropped = image.crop((x1, y1, x2, y2))
    return _pyzbar_decode(cropped)


def _stage4_rotation(image: Image.Image) -> str | None:
    """
    Stage 4: Try 90°, 180°, 270° rotations.
    Handles barcodes photographed sideways or upside down.

    Note: 45°/135° intentionally excluded — diagonal rotation produces
    black triangular fill corners that interfere with pyzbar detection.
    """
    for angle in [90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        result  = _pyzbar_decode(rotated)
        if result:
            return result
    return None


def _stage5_threshold(image: Image.Image) -> str | None:
    """
    Stage 5: Adaptive Gaussian threshold.
    Helps with uneven lighting — e.g. one side of barcode in shadow.
    """
    img_array = np.array(image.convert("L"))
    thresh    = cv2.adaptiveThreshold(
        img_array, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return _pyzbar_decode(Image.fromarray(thresh))


def _stage6_zxing(image: Image.Image) -> str | None:
    """
    Stage 6: ZXing C++ fallback.
    Catches ~12% of barcodes pyzbar misses — different algorithm,
    different failure modes. Handles DataMatrix and Code 128 too.
    Silently skipped if zxingcpp is not installed.
    """
    if not _ZXING_AVAILABLE:
        return None
    try:
        img_array = np.array(image.convert("RGB"))
        results   = zxingcpp.read_barcodes(img_array)
        if results:
            return results[0].text
    except Exception as e:
        logger.debug("ZXing error: %s", e)
    return None


# -----------------------------------------------------------------------
# REGION DETECTION
# -----------------------------------------------------------------------

def get_barcode_region(image: Image.Image) -> tuple | None:
    """
    Detect barcode region using morphological operations.
    Returns (x, y, w, h) bounding box or None if not found.

    Pipeline:
      Scharr gradient → blur → threshold → morphological close
      → erode → dilate → find largest contour
    """
    img_array = np.array(image.convert("L"))

    # Scharr gradient highlights the dense vertical stripes of a barcode
    grad_x   = cv2.Scharr(img_array, cv2.CV_32F, 1, 0)
    grad_y   = cv2.Scharr(img_array, cv2.CV_32F, 0, 1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Blur to merge nearby stripes into a solid region
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Morphological close fills the gaps between barcode bars
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Erode then dilate to remove noise outside the barcode region
    eroded  = cv2.erode(closed, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest  = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)


# -----------------------------------------------------------------------
# CHECKSUM VALIDATORS
# -----------------------------------------------------------------------

def _validate_ean13(barcode: str) -> bool:
    """EAN-13 checksum: alternating weights 1 and 3 on first 12 digits."""
    digits = [int(d) for d in barcode]
    total  = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:-1]))
    check  = (10 - (total % 10)) % 10
    return check == digits[-1]


def _validate_upca(barcode: str) -> bool:
    """
    UPC-A checksum: same algorithm as EAN-13 but with 12 digits.
    Weights start with 3 (odd positions) then alternate.
    """
    digits = [int(d) for d in barcode]
    total  = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits[:-1]))
    check  = (10 - (total % 10)) % 10
    return check == digits[-1]


def _validate_ean8(barcode: str) -> bool:
    """EAN-8 checksum: same alternating weight algorithm over 7 digits."""
    digits = [int(d) for d in barcode]
    total  = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits[:-1]))
    check  = (10 - (total % 10)) % 10
    return check == digits[-1]