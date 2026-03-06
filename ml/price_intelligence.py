"""
ml/price_intelligence.py
========================
Price intelligence and value analysis for ShelfScan.

Calculates:
  - estimated_price       — category-based price per 100g (GBP)
  - estimated_pack_price  — estimated full pack price
  - health_per_penny      — health score normalised by price
  - nutrition_density     — composite nutrient quality score (0-100)
  - category_benchmark    — category median nutrition density
  - vs_benchmark          — how product compares to category average
  - value_verdict         — human-readable value judgement + emoji
  - explanation           — one-line human-readable summary

Category detection priority:
  1. OpenFoodFacts categories_tags / pnns_groups_2 (most reliable)
  2. Specific keyword matching (longer/precise terms first)
  3. Generic keyword matching (short/broad terms second)
  4. NOVA group fallback (coarsest signal)
"""

import re
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# CATEGORY PRICE LOOKUP
# Estimated price per 100g in GBP based on UK supermarket averages.
# Mid-range estimates across Tesco / Sainsbury's / ASDA.
# -----------------------------------------------------------------------

_CATEGORY_PRICE_PER_100G: Dict[str, float] = {
    # Confectionery / spreads
    "spreads":           0.45,
    "chocolate":         0.55,
    "confectionery":     0.40,
    "snacks":            0.38,
    "crisps":            0.35,

    # Cereals / grains
    "cereals":           0.22,
    "oats":              0.12,
    "bread":             0.18,
    "pasta":             0.15,
    "rice":              0.12,

    # Dairy / alternatives
    "dairy":             0.20,
    "yoghurt":           0.25,
    "cheese":            0.60,
    "milk_alternative":  0.22,

    # Beverages
    "water":             0.05,
    "juice":             0.18,
    "soft_drinks":       0.12,

    # Meat / fish
    "meat":              0.80,
    "fish":              0.90,

    # Fruit / veg
    "fruit":             0.25,
    "vegetables":        0.18,

    # Default — mid-price packaged goods
    "default":           0.30,
}


# -----------------------------------------------------------------------
# CATEGORY BENCHMARKS
# Median nutrition density scores for common supermarket categories.
# Scale: 0-100 — higher is more nutritious per 100g.
# -----------------------------------------------------------------------

_CATEGORY_BENCHMARKS: Dict[str, float] = {
    "water":                95.0,
    "fresh produce":        85.0,
    "whole grain":          72.0,
    "whole food":           80.0,
    "nuts or seeds":        70.0,
    "dairy":                55.0,
    "grains":               52.0,
    "plant-based":          50.0,
    "soup":                 48.0,
    "meat or fish":         45.0,
    "baby food":            45.0,
    "juice":                42.0,
    "breakfast cereal":     40.0,
    "cereals":              40.0,
    "frozen food":          38.0,
    "packaged food":        35.0,
    "spread":               32.0,
    "snack bar":            30.0,
    "snacks":               28.0,
    "oil or fat":           28.0,
    "supplement":           50.0,
    "soft drink":           15.0,
    "soft_drinks":          15.0,
    "processed snack":      22.0,
    "processed meat":       25.0,
    "confectionery":        12.0,
    "chocolate":            20.0,
    "ultra-processed food": 18.0,
}

_DEFAULT_BENCHMARK = 35.0


# -----------------------------------------------------------------------
# OFF CATEGORIES → INTERNAL CATEGORY
# -----------------------------------------------------------------------

_OFF_CATEGORY_MAP: Dict[str, str] = {
    # Cereals
    "en:breakfast-cereals":       "cereals",
    "en:cereals":                 "cereals",
    "en:oatmeals":                "oats",
    "en:porridges":               "oats",

    # Bread / grains
    "en:breads":                  "bread",
    "en:pastas":                  "pasta",
    "en:rices":                   "rice",

    # Dairy
    "en:cheeses":                 "cheese",
    "en:yogurts":                 "yoghurt",
    "en:milks":                   "dairy",
    "en:plant-based-milks":       "milk_alternative",
    "en:plant-milks":             "milk_alternative",

    # Confectionery
    "en:chocolates":              "chocolate",
    "en:chocolate-spreads":       "spreads",
    "en:candies":                 "confectionery",
    "en:gummies":                 "confectionery",

    # Snacks
    "en:chips-and-crisps":        "crisps",
    "en:snacks":                  "snacks",

    # Beverages
    "en:waters":                  "water",
    "en:fruit-juices":            "juice",
    "en:soft-drinks":             "soft_drinks",
    "en:energy-drinks":           "soft_drinks",

    # Meat / fish
    "en:meats":                   "meat",
    "en:fishes":                  "fish",
    "en:seafoods":                "fish",

    # Fruit / veg
    "en:fruits":                  "fruit",
    "en:vegetables":              "vegetables",

    # PNNS groups
    "cereals and potatoes":       "cereals",
    "milk and dairy products":    "dairy",
    "meat":                       "meat",
    "fish and seafood":           "fish",
    "fruits":                     "fruit",
    "vegetables":                 "vegetables",
    "sugary snacks":              "snacks",
    "salty snacks":               "crisps",
    "waters and flavored waters": "water",
    "fruit juices":               "juice",
    "sweetened beverages":        "soft_drinks",
}


# -----------------------------------------------------------------------
# KEYWORD → CATEGORY MAPPING
# Specific checked before generic — strict priority via list concatenation.
# -----------------------------------------------------------------------

_SPECIFIC_RAW: List[Tuple[str, str]] = [
    (r"plant\s+milk",    "milk_alternative"),
    (r"oat\s+milk",      "milk_alternative"),
    (r"almond\s+milk",   "milk_alternative"),
    (r"soy\s+milk",      "milk_alternative"),
    (r"protein\s+bar",   "snacks"),
    (r"energy\s+bar",    "snacks"),
    (r"cereal\s+bar",    "snacks"),
    (r"corn\s+flakes?",  "cereals"),
    (r"granola",         "cereals"),
    (r"muesli",          "cereals"),
    (r"porridge",        "oats"),
    (r"nutella",         "spreads"),
    (r"pringles",        "crisps"),
    (r"kellogg",         "cereals"),
    (r"quaker",          "oats"),
    (r"activia",         "yoghurt"),
    (r"evian",           "water"),
    (r"kinder",          "chocolate"),
    (r"bueno",           "chocolate"),
    (r"haribo",          "confectionery"),
]

_GENERIC_RAW: List[Tuple[str, str]] = [
    ("cereal",    "cereals"),
    ("chocolate", "chocolate"),
    ("gummy",     "confectionery"),
    ("candy",     "confectionery"),
    ("crisp",     "crisps"),
    ("chip",      "crisps"),
    ("oat",       "oats"),
    ("bread",     "bread"),
    ("pasta",     "pasta"),
    ("rice",      "rice"),
    ("water",     "water"),
    ("juice",     "juice"),
    ("yoghurt",   "yoghurt"),
    ("yogurt",    "yoghurt"),
    ("cheese",    "cheese"),
    ("milk",      "dairy"),
    ("protein",   "snacks"),
    ("energy",    "soft_drinks"),
]

_COMPILED_PATTERNS: List[Tuple[re.Pattern, str]] = (
    [
        (re.compile(rf"\b{pattern}\b", re.IGNORECASE), cat)
        for pattern, cat in _SPECIFIC_RAW
    ]
    + [
        (re.compile(rf"\b{re.escape(kw)}\b", re.IGNORECASE), cat)
        for kw, cat in _GENERIC_RAW
    ]
)

_PREMIUM_BRANDS: List[str] = [
    "lindt", "godiva", "waitrose", "m&s", "marks & spencer",
]


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _safe(value: Any, default: float = 0.0) -> float:
    """Safely cast a value to float, returning default on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_labels(raw: Any) -> str:
    """
    Normalise labels to lowercase string.
    Handles both list ['en:organic'] and string 'en:organic,en:fair-trade'.
    """
    if not raw:
        return ""
    if isinstance(raw, list):
        return " ".join(raw).lower()
    return str(raw).lower()


def _parse_weight_g(product: Dict[str, Any]) -> float:
    """
    Extract product weight in grams from weight_g or quantity fields.
    Handles: 500g, 1.5kg, 330ml, 1.5L. Falls back to 100g.
    kg checked before g to avoid partial match on '1.5kg'.
    """
    weight = _safe(product.get("weight_g"), 0.0)
    if weight > 0:
        return weight

    quantity = str(product.get("quantity") or "")

    m = re.search(r"([\d.]+)\s*kg", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000

    m = re.search(r"([\d.]+)\s*g\b", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"([\d.]+)\s*l\b", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000

    m = re.search(r"([\d.]+)\s*ml", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return 100.0


def _category_from_off_tags(product: Dict[str, Any]) -> Optional[str]:
    """
    Resolve category from OFF structured fields before keyword matching.
    Checks categories_tags then pnns_groups_2.
    """
    tags = product.get("categories_tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]

    for tag in tags:
        category = _OFF_CATEGORY_MAP.get(tag.strip().lower())
        if category:
            return category

    pnns = (product.get("pnns_groups_2") or "").strip().lower()
    if pnns:
        return _OFF_CATEGORY_MAP.get(pnns)

    return None


# -----------------------------------------------------------------------
# NUTRITION DENSITY
# Shared between PriceIntelligence and calculate_price_intelligence.
# -----------------------------------------------------------------------

def _nutrition_density(product: Dict[str, Any]) -> float:
    """
    Composite nutrition density score (0-100).

    Positive: protein (max 30pts), fiber (max 20pts).
    Negative: sugar (max -18), sat fat (max -15), salt (max -9),
              additives (max -8), NOVA penalty (0-12).

    Scores are capped to prevent extreme values from single nutrients
    dominating the result. NOVA penalty reflects processing level
    independent of nutrient profile.
    """
    protein = _safe(product.get("proteins"))
    fiber   = _safe(product.get("fiber"))
    sugar   = _safe(product.get("sugars"))
    sat_fat = _safe(product.get("saturated_fat"))
    salt    = _safe(product.get("salt"))
    adds    = len(product.get("additives") or [])
    nova    = product.get("nova_group") or 2

    positive = (
        min(protein * 2.0, 30)
        + min(fiber  * 3.0, 20)
    )

    negative = (
        min(sugar   * 0.6, 18)
        + min(sat_fat * 1.5, 15)
        + min(salt   * 3.0,  9)
        + min(adds   * 0.8,  8)
    )

    nova_penalty = {1: 0, 2: 2, 3: 5, 4: 12}.get(int(nova), 3)

    return max(0.0, min(100.0, 50 + positive - negative - nova_penalty))


# -----------------------------------------------------------------------
# PRICE INTELLIGENCE ENGINE
# -----------------------------------------------------------------------

class PriceIntelligence:
    """
    Estimates product price and calculates health-per-penny + benchmark.

    Usage:
        pi     = PriceIntelligence()
        result = pi.analyze(product_dict, health_score=74.0)
    """

    def analyze(
        self,
        product:      Dict[str, Any],
        health_score: float,
        category:     Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Calculate full price intelligence for a product.

        Args:
            product:      cleaned product dict from services/api.py
            health_score: 0-100 score from ml/predict.py
            category:     optional category from image classifier
                          (overrides internal detection if provided)

        Returns:
            {
                category              : str
                estimated_price       : float  (GBP per 100g)
                estimated_pack_price  : float  (GBP per pack)
                health_per_penny      : float | None
                nutrition_density     : float  (0-100)
                category_benchmark    : float  (category median density)
                vs_benchmark          : float  (density - benchmark)
                price_band            : str    ('budget' | 'mid' | 'premium')
                value_verdict         : str
                value_emoji           : str
                explanation           : str
            }
        """
        # Category — use provided or detect
        cat = (
            category.lower().strip()
            if category
            else self._detect_category(product)
        )

        estimated_price = self._adjust_price(
            _CATEGORY_PRICE_PER_100G.get(cat, _CATEGORY_PRICE_PER_100G["default"]),
            product,
        )

        density   = _nutrition_density(product)
        benchmark = _CATEGORY_BENCHMARKS.get(cat, _DEFAULT_BENCHMARK)

        # health_per_penny in pence — matches doc 15 convention
        # Uses estimated price converted to pence (× 100)
        price_pence      = estimated_price * 100
        health_per_penny = (
            round(health_score / price_pence, 2)
            if price_pence > 0
            else None
        )

        verdict, emoji = self._value_verdict(
            health_score, density, health_per_penny
        )

        return {
            "category":             cat,
            "estimated_price":      round(estimated_price, 2),
            "estimated_pack_price": round(
                self._estimate_pack_price(product, estimated_price), 2
            ),
            "health_per_penny":     health_per_penny,
            "nutrition_density":    round(density, 1),
            "category_benchmark":   benchmark,
            "vs_benchmark":         round(density - benchmark, 1),
            "price_band":           self._price_band(estimated_price),
            "value_verdict":        verdict,
            "value_emoji":          emoji,
            "explanation":          self._build_explanation(
                product, density, benchmark, cat, health_per_penny
            ),
        }

    # -----------------------------------------------------------------------
    # CATEGORY DETECTION
    # -----------------------------------------------------------------------

    def _detect_category(self, product: Dict[str, Any]) -> str:
        """
        4-stage priority cascade:
          1. OFF categories_tags / pnns_groups_2
          2. Specific keyword patterns
          3. Generic keyword patterns
          4. NOVA group fallback
        """
        category = _category_from_off_tags(product)
        if category:
            return category

        name = (
            product.get("name")
            or product.get("product_name")
            or product.get("generic_name")
            or ""
        ).lower()
        ingr = (product.get("ingredients") or "").lower()
        text = f"{name} {ingr}"

        for pattern, cat in _COMPILED_PATTERNS:
            if pattern.search(text):
                return cat

        nova = product.get("nova_group") or 2
        if nova == 1:
            return "fruit"
        if nova == 4:
            return "snacks"

        return "default"

    # -----------------------------------------------------------------------
    # PRICE ADJUSTMENT
    # -----------------------------------------------------------------------

    def _adjust_price(self, base: float, product: Dict[str, Any]) -> float:
        """
        Adjust base price using NOVA, organic, and premium brand signals.
        NOVA multiplier kept small — correlation with price is weak.
        """
        price = base

        nova = product.get("nova_group") or 2
        if nova == 4:
            price *= 0.98
        elif nova == 1:
            price *= 0.95

        labels = _parse_labels(product.get("labels"))
        if "organic" in labels:
            price *= 1.30

        brand = (product.get("brand") or "").lower()
        if any(b in brand for b in _PREMIUM_BRANDS):
            price *= 1.35

        return price

    # -----------------------------------------------------------------------
    # PACK PRICE / BAND / VERDICT / EXPLANATION
    # -----------------------------------------------------------------------

    def _estimate_pack_price(
        self, product: Dict[str, Any], price_per_100g: float
    ) -> float:
        return price_per_100g * (_parse_weight_g(product) / 100)

    def _price_band(self, price_per_100g: float) -> str:
        if price_per_100g < 0.20:
            return "budget"
        if price_per_100g < 0.50:
            return "mid"
        return "premium"

    def _value_verdict(
        self,
        health_score:    float,
        density:         float,
        health_per_penny: Optional[float],
    ) -> Tuple[str, str]:
        """
        Primary signal: health_per_penny when price available.
        Fallback: combined health_score + density when price unavailable.
        """
        if health_per_penny is not None:
            if health_per_penny >= 0.5:
                return "Excellent Value", "💎"
            if health_per_penny >= 0.3:
                return "Good Value", "👍"
            if health_per_penny >= 0.15:
                return "Fair Value", "➡️"
            return "Poor Value", "👎"

        combined = health_score * 0.6 + density * 0.4
        if combined >= 65:
            return "High Nutrition", "💎"
        if combined >= 45:
            return "Moderate Nutrition", "👍"
        if combined >= 25:
            return "Low Nutrition", "➡️"
        return "Very Low Nutrition", "👎"

    def _build_explanation(
        self,
        product:         Dict[str, Any],
        density:         float,
        benchmark:       float,
        category:        str,
        health_per_penny: Optional[float],
    ) -> str:
        """One-line human-readable summary for UI display."""
        name = product.get("name") or "This product"
        vs   = "above" if density >= benchmark else "below"
        diff = abs(round(density - benchmark, 1))

        parts = [
            f"{name} has a nutrition density of {density:.0f}/100, "
            f"{diff:.0f} points {vs} the {category} average ({benchmark:.0f})."
        ]

        if health_per_penny is not None:
            quality = "great" if health_per_penny >= 0.3 else "limited"
            parts.append(
                f" At {health_per_penny:.2f} health points per penny, "
                f"{quality} nutritional value for money."
            )

        return "".join(parts)