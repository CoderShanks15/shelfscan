import cv2
import numpy as np
from PIL import Image
from pyzbar import pyzbar
import pytesseract
import zxingcpp


def decode_barcode_from_image(image: Image.Image) -> str | None:
    """
    5-stage cascade barcode decoder.
    Tries each stage in order, returns first successful decode.
    """
    # Stage 1: Direct pyzbar decode
    result = _stage1_direct(image)
    if result:
        return result

    # Stage 2: CLAHE contrast enhancement
    result = _stage2_clahe(image)
    if result:
        return result

    # Stage 3: Multi-angle rotation scan
    result = _stage3_rotation(image)
    if result:
        return result

    # Stage 4: Adaptive threshold
    result = _stage4_threshold(image)
    if result:
        return result

    # Stage 5: ZXing fallback
    result = _stage5_zxing(image)
    if result:
        return result

    return None


def validate_barcode(barcode: str) -> bool:
    """
    Validate EAN-13 checksum.
    Returns True if valid EAN-13, False otherwise.
    """
    if not barcode or not barcode.isdigit():
        return False

    # UPC-A (12 digits) — also valid
    if len(barcode) == 12:
        return True

    # EAN-8
    if len(barcode) == 8:
        return True

    # EAN-13 checksum validation
    if len(barcode) == 13:
        digits = [int(d) for d in barcode]
        total = 0
        for i, digit in enumerate(digits[:-1]):
            total += digit * (1 if i % 2 == 0 else 3)
        check = (10 - (total % 10)) % 10
        return check == digits[-1]

    return False


# --- Stage implementations ---

def _pyzbar_decode(image: Image.Image) -> str | None:
    """Run pyzbar on a PIL image, return barcode string or None."""
    results = pyzbar.decode(image)
    if results:
        return results[0].data.decode("utf-8")
    return None


def _stage1_direct(image: Image.Image) -> str | None:
    """Stage 1: Direct decode — no preprocessing."""
    return _pyzbar_decode(image)


def _stage2_clahe(image: Image.Image) -> str | None:
    """Stage 2: CLAHE histogram equalisation for dark/low contrast images."""
    img_array = np.array(image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_array)
    enhanced_image = Image.fromarray(enhanced)
    return _pyzbar_decode(enhanced_image)


def _stage3_rotation(image: Image.Image) -> str | None:
    """Stage 3: Try multiple rotation angles."""
    for angle in [90, 180, 270, 45, 135]:
        rotated = image.rotate(angle, expand=True)
        result = _pyzbar_decode(rotated)
        if result:
            return result
    return None


def _stage4_threshold(image: Image.Image) -> str | None:
    """Stage 4: Adaptive threshold for uneven lighting."""
    img_array = np.array(image.convert("L"))
    thresh = cv2.adaptiveThreshold(
        img_array, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh_image = Image.fromarray(thresh)
    return _pyzbar_decode(thresh_image)


def _stage5_zxing(image: Image.Image) -> str | None:
    """Stage 5: ZXing C++ fallback decoder."""
    try:
        img_array = np.array(image.convert("RGB"))
        results = zxingcpp.read_barcodes(img_array)
        if results:
            return results[0].text
    except Exception:
        pass
    return None


def get_barcode_region(image: Image.Image) -> tuple | None:
    """
    Detect barcode region in image using morphological ops.
    Returns (x, y, w, h) bounding box or None.
    """
    img_array = np.array(image.convert("L"))

    # Scharr gradient to find barcode stripes
    grad_x = cv2.Scharr(img_array, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(img_array, cv2.CV_32F, 0, 1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    # Blur and threshold
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # Morphological close to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Erode and dilate to remove noise
    eroded = cv2.erode(closed, None, iterations=4)
    dilated = cv2.dilate(eroded, None, iterations=4)

    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, w, h)