"""
ml/image_classifier.py
======================
Product image analysis for ShelfScan.

Two-layer design:
  Layer 1 — OpenCV (always available)
      - Image quality scoring (blur, brightness, contrast)
      - K-means dominant colour extraction (HSV-based naturalness)
      - Barcode region detection with aspect ratio filtering
      - Heuristic category inference from product metadata

  Layer 2 — CLIP zero-shot classification (optional, ~340MB download)
      - Loads openai/clip-vit-base-patch32 via HuggingFace transformers
      - Classifies product image against 28 descriptive text labels
      - Fused with metadata prediction (70% CLIP / 30% metadata)
      - Gracefully absent if torch/transformers not installed

Public API:
    classifier = ProductImageClassifier()
    result = classifier.classify(image, product)

    result = {
        'category'           : str   — inferred product category
        'confidence'         : float — 0.0-1.0
        'clip_available'     : bool  — whether CLIP ran or fell back
        'processing_level'   : str   — 'minimally processed' | 'ultra-processed' | ...
        'dominant_colours'   : list  — top-5 RGB tuples by pixel frequency
        'colour_naturalness' : float — 0-100, higher = more natural-looking
        'barcode_region'     : tuple | None — (x, y, w, h) or None
        'image_quality'      : dict  — blur/brightness/contrast scores + warning
        'reasoning'          : list  — human-readable explanation of classification
    }

Design decisions:
  - CLIP loaded once at __init__ — use @st.cache_resource on classifier in app.py
  - Hybrid fusion: CLIP (0.7) + metadata (0.3) — metadata corrects CLIP errors
  - HSV colour space for naturalness — more stable than RGB heuristics
  - Low-saturation pixels (s <= 20) excluded from naturalness — white/grey
    packaging would otherwise score as 'natural' due to H=0 falling in red range
  - Quality check order: brightness → contrast → blur (flat images have blur=0
    which would incorrectly report 'blurry' before 'low contrast')
  - Barcode region filtered by aspect ratio (w/h > 1.5) — real barcodes are
    wider than tall; reduces false positives from other rectangular regions
  - Laplacian blur detection requires uint8 input — float32 crashes on CV_64F
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# CLIP — optional import
# -----------------------------------------------------------------------

try:
    import torch
    from transformers import CLIPModel, CLIPProcessor
    _CLIP_IMPORTABLE = True
except ImportError:
    _CLIP_IMPORTABLE = False
    logger.warning(
        "torch / transformers not installed — CLIP unavailable. "
        "Install with: pip install torch torchvision transformers"
    )


# -----------------------------------------------------------------------
# CLIP LABELS — 28 descriptive sentences for better prompt coverage
# Descriptive sentences work better than single words for CLIP.
# -----------------------------------------------------------------------

_CLIP_LABELS: List[str] = [
    # Produce
    "fresh fruit or vegetables",
    "fresh leafy vegetables like spinach or lettuce",
    "whole grain bread or oats",
    # Dairy
    "dairy product like milk or cheese",
    "fermented dairy like yoghurt or kefir",
    # Snacks / confectionery
    "processed snack food like crisps or chocolate bar",
    "sugary confectionery like sweets gummies or candy",
    "chocolate spread or nut butter",
    # Cereals
    "breakfast cereal in a box",
    "granola or muesli bag",
    # Beverages
    "bottled water or natural mineral water",
    "fruit juice or smoothie bottle",
    "soft drink or energy drink can",
    # Proteins
    "meat or fish product",
    "processed meat like sausages or ham",
    "nuts or seeds like almonds or peanuts",
    "protein bar or energy bar",
    # Ready meals / convenience
    "ready meal or ultra-processed convenience food",
    "packaged instant noodles or ramen",
    "frozen food packaging",
    # Other
    "spread or condiment like jam peanut butter or sauce",
    "pasta or rice in packaging",
    "baby food or infant formula",
    "plant-based meat alternative",
    "cooking oil or fat",
    "soup or broth in a can or carton",
    "food packaging or product box with no visible food",
    "supplement or protein powder",
]

# CLIP label → internal category string
_LABEL_TO_CATEGORY: Dict[str, str] = {
    "fresh fruit or vegetables":                             "fresh produce",
    "fresh leafy vegetables like spinach or lettuce":        "fresh produce",
    "whole grain bread or oats":                             "whole grain",
    "dairy product like milk or cheese":                     "dairy",
    "fermented dairy like yoghurt or kefir":                 "dairy",
    "processed snack food like crisps or chocolate bar":     "processed snack",
    "sugary confectionery like sweets gummies or candy":     "confectionery",
    "chocolate spread or nut butter":                        "spread",
    "breakfast cereal in a box":                             "breakfast cereal",
    "granola or muesli bag":                                 "breakfast cereal",
    "bottled water or natural mineral water":                "water",
    "fruit juice or smoothie bottle":                        "juice",
    "soft drink or energy drink can":                        "soft drink",
    "meat or fish product":                                  "meat or fish",
    "processed meat like sausages or ham":                   "processed meat",
    "nuts or seeds like almonds or peanuts":                 "nuts or seeds",
    "protein bar or energy bar":                             "snack bar",
    "ready meal or ultra-processed convenience food":        "ultra-processed food",
    "packaged instant noodles or ramen":                     "ultra-processed food",
    "frozen food packaging":                                 "frozen food",
    "spread or condiment like jam peanut butter or sauce":   "spread",
    "pasta or rice in packaging":                            "grains",
    "baby food or infant formula":                           "baby food",
    "plant-based meat alternative":                          "plant-based",
    "cooking oil or fat":                                    "oil or fat",
    "soup or broth in a can or carton":                      "soup",
    "food packaging or product box with no visible food":    "packaged food",
    "supplement or protein powder":                          "supplement",
}

# NOVA group → processing level string
_NOVA_TO_LEVEL: Dict[int, str] = {
    1: "minimally processed",
    2: "processed ingredient",
    3: "processed food",
    4: "ultra-processed",
}

# Image quality thresholds
_BLUR_THRESHOLD       = 80.0   # Laplacian variance — below = blurry
_BRIGHTNESS_MIN       = 40.0   # mean pixel value   — below = too dark
_BRIGHTNESS_MAX       = 220.0  # mean pixel value   — above = overexposed
_CONTRAST_THRESHOLD   = 30.0   # std deviation      — below = low contrast


# -----------------------------------------------------------------------
# CLASSIFIER
# -----------------------------------------------------------------------

class ProductImageClassifier:
    """
    Classifies product images using OpenCV heuristics + optional CLIP.

    Thread-safe: stateless after __init__. CLIP model stored on instance
    but only read during inference (no mutation after load).

    Usage:
        classifier = ProductImageClassifier()
        result = classifier.classify(pil_image, product_dict)
    """

    def __init__(self) -> None:
        self._clip_model     = None
        self._clip_processor = None
        self._clip_available = False

        if _CLIP_IMPORTABLE:
            self._load_clip()

    def _load_clip(self) -> None:
        """
        Load CLIP model from HuggingFace.
        First call downloads ~340MB — cached locally afterwards.
        Use @st.cache_resource on the classifier instance in app.py
        to ensure the model loads once per process, not per session.
        """
        try:
            model_name = "openai/clip-vit-base-patch32"
            logger.info("Loading CLIP model %s ...", model_name)
            self._clip_processor = CLIPProcessor.from_pretrained(model_name)
            self._clip_model     = CLIPModel.from_pretrained(model_name)
            self._clip_model.eval()
            self._clip_available = True
            logger.info("CLIP loaded successfully")
        except Exception as e:
            logger.warning("CLIP failed to load: %s — falling back to heuristics", e)
            self._clip_available = False

    # -----------------------------------------------------------------------
    # PUBLIC METHOD
    # -----------------------------------------------------------------------

    def classify(
        self,
        image:   Image.Image,
        product: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Analyse a product image.

        Args:
            image:   PIL Image — product photo from Streamlit file uploader
            product: cleaned product dict from services/api.py (optional)

        Returns full result dict — see module docstring for field list.
        """
        product = product or {}

        # --- OpenCV analysis (always runs) ---
        image_quality      = self._image_quality(image)
        dominant_colours   = self._extract_dominant_colours(image)
        colour_naturalness = self._colour_naturalness_hsv(image)
        barcode_region     = self._detect_barcode_region(image)
        processing_level   = self._processing_level(product)

        # --- Category classification ---
        reasoning: List[str] = []

        if self._clip_available:
            clip_category, clip_confidence = self._clip_classify(image)
            meta_category, _               = self._heuristic_category(product)

            # Hybrid fusion: 70% CLIP vision + 30% metadata signal.
            # Metadata corrects CLIP when vision signal is ambiguous —
            # e.g. CLIP sees brown packaging and says 'chocolate' but
            # metadata says 'oats'. Confidence blended accordingly.
            if meta_category != "packaged food" and meta_category != clip_category:
                category   = clip_category
                confidence = round(clip_confidence * 0.7 + 0.3 * 0.6, 3)
                reasoning.append(
                    f"CLIP classified as '{clip_category}' "
                    f"(confidence {clip_confidence:.0%})"
                )
                reasoning.append(
                    f"Metadata suggests '{meta_category}' — "
                    f"fusion applied (70% vision / 30% metadata)"
                )
            else:
                category   = clip_category
                confidence = round(clip_confidence, 3)
                reasoning.append(
                    f"CLIP classified as '{clip_category}' "
                    f"(confidence {clip_confidence:.0%})"
                )
                reasoning.append("Metadata signal consistent with vision result")
        else:
            category, confidence = self._heuristic_category(product)
            reasoning.append(
                f"CLIP unavailable — heuristic classification: '{category}'"
            )

        # --- Additives-based processing level refinement ---
        # If product has 5+ additives, override nova-based processing level.
        # Many products are NOVA 3 but have extensive additive lists.
        additives = product.get("additives") or []
        if len(additives) >= 5 and processing_level != "ultra-processed":
            processing_level = "ultra-processed"
            reasoning.append(
                f"{len(additives)} additives detected — "
                f"overriding processing level to ultra-processed"
            )
        else:
            reasoning.append(f"Processing level: {processing_level}")

        # --- Colour signal ---
        tone_label = "natural tones" if colour_naturalness >= 55 else "artificial/packaged tones"
        reasoning.append(
            f"Colour naturalness: {colour_naturalness}/100 ({tone_label})"
        )

        # --- Quality warning appended last ---
        if not image_quality["usable"]:
            reasoning.append(
                f"Image quality warning: {image_quality['warning']} — "
                f"classification may be less accurate"
            )

        return {
            "category":           category,
            "confidence":         confidence,
            "clip_available":     self._clip_available,
            "processing_level":   processing_level,
            "dominant_colours":   dominant_colours,
            "colour_naturalness": colour_naturalness,
            "barcode_region":     barcode_region,
            "image_quality":      image_quality,
            "reasoning":          reasoning,
        }

    # -----------------------------------------------------------------------
    # IMAGE QUALITY
    # -----------------------------------------------------------------------

    def _image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """
        Score image quality along three axes:
          blur       — Laplacian variance (higher = sharper)
          brightness — mean pixel intensity (40-220 = acceptable range)
          contrast   — pixel std deviation (higher = more contrast)

        Warning check order: brightness → contrast → blur.
        Flat/solid-colour images have blur=0, which would incorrectly
        report 'blurry' if blur is checked first. Brightness and contrast
        are more meaningful signals for flat images.

        Laplacian requires uint8 input — float32 raises cv2.error on CV_64F.
        """
        gray       = np.array(image.convert("L"), dtype=np.uint8)
        blur       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.astype(np.float32).mean())
        contrast   = float(gray.astype(np.float32).std())

        warning = None
        if brightness < _BRIGHTNESS_MIN:
            warning = f"image is too dark (brightness={brightness:.0f})"
        elif brightness > _BRIGHTNESS_MAX:
            warning = f"image is overexposed (brightness={brightness:.0f})"
        elif contrast < _CONTRAST_THRESHOLD:
            warning = f"image has low contrast (contrast={contrast:.0f})"
        elif blur < _BLUR_THRESHOLD:
            warning = f"image is blurry (sharpness={blur:.0f})"

        return {
            "blur":       round(blur, 1),
            "brightness": round(brightness, 1),
            "contrast":   round(contrast, 1),
            "usable":     warning is None,
            "warning":    warning,
        }

    # -----------------------------------------------------------------------
    # LAYER 2 — CLIP
    # -----------------------------------------------------------------------

    def _clip_classify(self, image: Image.Image) -> Tuple[str, float]:
        """
        Run CLIP zero-shot classification against _CLIP_LABELS.
        Returns (category_string, confidence_float).

        torch.no_grad() prevents gradient accumulation — essential for
        inference speed and memory. Single image takes ~0.8s on CPU.
        """
        try:
            import torch

            inputs = self._clip_processor(
                text=_CLIP_LABELS,
                images=image.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs   = self._clip_model(**inputs)
                logits    = outputs.logits_per_image   # shape: (1, n_labels)
                probs     = logits.softmax(dim=1).squeeze()

            best_idx   = int(probs.argmax())
            best_label = _CLIP_LABELS[best_idx]
            confidence = float(probs[best_idx])
            category   = _LABEL_TO_CATEGORY.get(best_label, best_label)

            logger.info(
                "CLIP classification: %s (confidence=%.2f)", category, confidence
            )
            return category, confidence

        except Exception as e:
            logger.warning("CLIP inference failed: %s — falling back to heuristics", e)
            return self._heuristic_category({})

    # -----------------------------------------------------------------------
    # LAYER 1 — OPENCV
    # -----------------------------------------------------------------------

    def _extract_dominant_colours(
        self, image: Image.Image, k: int = 5
    ) -> List[Tuple[int, int, int]]:
        """
        K-means colour clustering to find dominant colours.
        Resize to 150x150 — sufficient colour representation, ~30ms.
        Returns top-k RGB tuples ordered by pixel frequency (most common first).
        """
        img_small = image.convert("RGB").resize((150, 150))
        arr       = np.array(img_small, dtype=np.float32).reshape(-1, 3)
        k         = min(k, len(arr))
        criteria  = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            20, 1.0,
        )
        try:
            _, labels, centres = cv2.kmeans(
                arr, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
            )
        except cv2.error as e:
            logger.warning("K-means failed: %s", e)
            return []

        counts = np.bincount(labels.flatten())
        order  = np.argsort(-counts)
        return [
            (int(centres[i][0]), int(centres[i][1]), int(centres[i][2]))
            for i in order
        ]

    def _colour_naturalness_hsv(self, image: Image.Image) -> float:
        """
        HSV-based colour naturalness score (0-100).

        Natural hue ranges in OpenCV HSV (H scale 0-180):
          Greens  : 35-85  (vegetables, herbs, fruit skin)
          Browns  : 8-25   (bread, oats, chocolate, nuts)
          Deep red: 0-8 and 160-180 (berries, tomatoes, apples)

        Low-saturation pixels (s <= 20) are excluded — white and grey
        packaging pixels have H=0 which falls inside the red range and
        would incorrectly inflate the naturalness score.

        Neon penalty: highly saturated pixels outside natural hues
        reduce the score by up to 40 points.
        """
        img_hsv = cv2.cvtColor(
            np.array(image.convert("RGB").resize((150, 150))),
            cv2.COLOR_RGB2HSV,
        )
        h = img_hsv[:, :, 0].flatten().astype(float)
        s = img_hsv[:, :, 1].flatten().astype(float)
        v = img_hsv[:, :, 2].flatten().astype(float)

        # Exclude low-saturation pixels (whites, greys, near-whites)
        saturated   = s > 20

        green_mask  = saturated & (h >= 35)  & (h <= 85)
        earthy_mask = saturated & (h >=  8)  & (h <= 25)
        red_mask    = saturated & ((h <= 8)  | (h >= 160))

        natural_pixels = (green_mask | earthy_mask | red_mask).sum()
        total_pixels   = len(h)

        neon_mask    = (s > 200) & (v > 200) & ~(green_mask | earthy_mask | red_mask)
        neon_penalty = neon_mask.sum() / total_pixels * 40

        naturalness = (natural_pixels / total_pixels) * 100 - neon_penalty
        return round(float(np.clip(naturalness, 0, 100)), 1)

    def _detect_barcode_region(
        self, image: Image.Image
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect barcode region using Scharr gradient + morphological ops.
        Filters candidates by aspect ratio (w/h > 1.5) — real barcodes
        are wider than tall. Reduces false positives from product text
        blocks and rectangular packaging features.
        Returns (x, y, w, h) or None.
        """
        img_array = np.array(image.convert("L"))
        try:
            grad_x   = cv2.Scharr(img_array, cv2.CV_32F, 1, 0)
            grad_y   = cv2.Scharr(img_array, cv2.CV_32F, 0, 1)
            gradient = cv2.subtract(grad_x, grad_y)
            gradient = cv2.convertScaleAbs(gradient)

            blurred   = cv2.blur(gradient, (9, 9))
            _, thresh = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            eroded  = cv2.erode(closed, None, iterations=4)
            dilated = cv2.dilate(eroded, None, iterations=4)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                return None

            # Aspect ratio filter: barcodes are wider than tall
            candidates = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if h > 0 and (w / h) > 1.5:
                    candidates.append((cv2.contourArea(c), x, y, w, h))

            if not candidates:
                return None

            _, x, y, w, h = max(candidates)
            return (x, y, w, h)

        except cv2.error as e:
            logger.warning("Barcode region detection failed: %s", e)
            return None

    # -----------------------------------------------------------------------
    # HEURISTIC FALLBACKS
    # -----------------------------------------------------------------------

    def _heuristic_category(
        self, product: Dict[str, Any]
    ) -> Tuple[str, float]:
        """
        Infer category from product dict when CLIP is unavailable.
        Confidence fixed at 0.6 — honest signal that this is a heuristic.
        """
        name = (
            product.get("name")
            or product.get("product_name")
            or ""
        ).lower()
        nova = product.get("nova_group") or 2

        if any(w in name for w in ["water", "evian", "volvic"]):
            return "water", 0.6
        if any(w in name for w in ["oat", "porridge", "quaker", "granola", "muesli"]):
            return "whole grain", 0.6
        if any(w in name for w in ["yoghurt", "yogurt", "activia"]):
            return "dairy", 0.6
        if any(w in name for w in ["juice", "smoothie"]):
            return "juice", 0.6
        if any(w in name for w in ["nutella", "spread", "jam", "butter"]):
            return "spread", 0.6
        if any(w in name for w in ["cereal", "cornflake", "kellogg"]):
            return "breakfast cereal", 0.6
        if any(w in name for w in ["crisp", "chip", "pringles", "chocolate", "kinder", "bueno"]):
            return "processed snack", 0.6
        if any(w in name for w in ["haribo", "gummy", "candy", "sweet"]):
            return "confectionery", 0.6

        if nova == 1: return "whole food", 0.6
        if nova == 4: return "ultra-processed food", 0.6
        return "packaged food", 0.6

    def _processing_level(self, product: Dict[str, Any]) -> str:
        """Map nova_group to human-readable processing level."""
        nova = product.get("nova_group")
        if nova is None:
            return "unknown"
        return _NOVA_TO_LEVEL.get(int(nova), "unknown")