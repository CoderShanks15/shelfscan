"""
ml/price_intelligence.py
========================
Price intelligence and value scoring for ShelfScan.

Estimates product price from category signals and calculates
health-per-penny — how much nutrition you get per unit of cost.

No live pricing API at hackathon stage. Uses category lookup tables
built from UK supermarket price surveys. Production version would
hit Tesco Developer API or Open Grocery API for real prices.

Category detection priority:
  1. OpenFoodFacts categories_tags / pnns_groups_2 fields (most reliable)
  2. Specific keyword matching (longer/precise terms first)
  3. Generic keyword matching (short/broad terms second)
  4. NOVA group fallback (coarsest signal)

Fixes vs submitted version:
  - 'corn flake' pattern replaced with 'corn flakes?' regex — handles plural
  - 'protein bar' / 'energy bar' moved to _SPECIFIC_KEYWORDS so they take
    priority over generic 'chocolate' / 'energy' matches
  - _parse_weight_g now handles kg and L units (were silently returning 100g)
"""

import re
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------------------------------------------------
# CATEGORY PRICE LOOKUP
# Estimated price per 100g in GBP based on UK supermarket averages.
# Mid-range estimates across Tesco / Sainsbury's / ASDA.
# -----------------------------------------------------------------------

_CATEGORY_PRICE_PER_100G: Dict[str, float] = {
    # Confectionery / spreads
    "spreads":          0.45,
    "chocolate":        0.55,
    "confectionery":    0.40,
    "snacks":           0.38,
    "crisps":           0.35,

    # Cereals / grains
    "cereals":          0.22,
    "oats":             0.12,
    "bread":            0.18,
    "pasta":            0.15,
    "rice":             0.12,

    # Dairy / alternatives
    "dairy":            0.20,
    "yoghurt":          0.25,
    "cheese":           0.60,
    "milk_alternative": 0.22,

    # Beverages
    "water":            0.05,
    "juice":            0.18,
    "soft_drinks":      0.12,

    # Meat / fish
    "meat":             0.80,
    "fish":             0.90,

    # Fruit / veg
    "fruit":            0.25,
    "vegetables":       0.18,

    # Default — mid-price packaged goods
    "default":          0.30,
}


# -----------------------------------------------------------------------
# OFF CATEGORIES → INTERNAL CATEGORY
#
# OpenFoodFacts provides categories_tags and pnns_groups_2 which are
# more reliable than keyword matching on product names.
# Checked first in _detect_category before any keyword logic.
# -----------------------------------------------------------------------

_OFF_CATEGORY_MAP: Dict[str, str] = {
    # Cereals
    "en:breakfast-cereals":        "cereals",
    "en:cereals":                  "cereals",
    "en:oatmeals":                 "oats",
    "en:porridges":                "oats",

    # Bread / grains
    "en:breads":                   "bread",
    "en:pastas":                   "pasta",
    "en:rices":                    "rice",

    # Dairy
    "en:cheeses":                  "cheese",
    "en:yogurts":                  "yoghurt",
    "en:milks":                    "dairy",
    "en:plant-based-milks":        "milk_alternative",
    "en:plant-milks":              "milk_alternative",

    # Confectionery
    "en:chocolates":               "chocolate",
    "en:chocolate-spreads":        "spreads",
    "en:candies":                  "confectionery",
    "en:gummies":                  "confectionery",

    # Snacks
    "en:chips-and-crisps":         "crisps",
    "en:snacks":                   "snacks",

    # Beverages
    "en:waters":                   "water",
    "en:fruit-juices":             "juice",
    "en:soft-drinks":              "soft_drinks",
    "en:energy-drinks":            "soft_drinks",

    # Meat / fish
    "en:meats":                    "meat",
    "en:fishes":                   "fish",
    "en:seafoods":                 "fish",

    # Fruit / veg
    "en:fruits":                   "fruit",
    "en:vegetables":               "vegetables",

    # PNNS groups (pnns_groups_2 field — lowercase matched)
    "cereals and potatoes":        "cereals",
    "milk and dairy products":     "dairy",
    "meat":                        "meat",
    "fish and seafood":            "fish",
    "fruits":                      "fruit",
    "vegetables":                  "vegetables",
    "sugary snacks":               "snacks",
    "salty snacks":                "crisps",
    "waters and flavored waters":  "water",
    "fruit juices":                "juice",
    "sweetened beverages":         "soft_drinks",
}


# -----------------------------------------------------------------------
# KEYWORD → CATEGORY MAPPING
#
# Split into two tiers — specific checked before generic.
# _SPECIFIC_KEYWORDS: multi-word phrases, brand names, precise food types.
# _GENERIC_KEYWORDS:  single broad terms — matched only after specific fails.
#
# Patterns are pre-compiled at module load for performance.
# Dict merge NOT used — order must be preserved strictly.
# -----------------------------------------------------------------------

# Specific keywords use plain string patterns EXCEPT for plurals which
# use inline regex. Each entry is (raw_pattern_string, category).
_SPECIFIC_RAW: List[Tuple[str, str]] = [
    # Multi-word milk alternatives (must be before single 'milk')
    (r"plant\s+milk",          "milk_alternative"),
    (r"oat\s+milk",            "milk_alternative"),
    (r"almond\s+milk",         "milk_alternative"),
    (r"soy\s+milk",            "milk_alternative"),

    # Snack bars — before 'chocolate' / 'energy' generic terms
    (r"protein\s+bar",         "snacks"),
    (r"energy\s+bar",          "snacks"),
    (r"cereal\s+bar",          "snacks"),

    # Cereals — 'corn flakes?' handles both 'corn flake' and 'corn flakes'
    (r"corn\s+flakes?",        "cereals"),
    (r"granola",               "cereals"),
    (r"muesli",                "cereals"),
    (r"porridge",              "oats"),

    # Brand names
    (r"nutella",               "spreads"),
    (r"pringles",              "crisps"),
    (r"kellogg",               "cereals"),
    (r"quaker",                "oats"),
    (r"activia",               "yoghurt"),
    (r"evian",                 "water"),
    (r"kinder",                "chocolate"),
    (r"bueno",                 "chocolate"),
    (r"haribo",                "confectionery"),
]

_GENERIC_RAW: List[Tuple[str, str]] = [
    ("cereal",       "cereals"),
    ("chocolate",    "chocolate"),
    ("gummy",        "confectionery"),
    ("candy",        "confectionery"),
    ("crisp",        "crisps"),
    ("chip",         "crisps"),
    ("oat",          "oats"),
    ("bread",        "bread"),
    ("pasta",        "pasta"),
    ("rice",         "rice"),
    ("water",        "water"),
    ("juice",        "juice"),
    ("yoghurt",      "yoghurt"),
    ("yogurt",       "yoghurt"),
    ("cheese",       "cheese"),
    ("milk",         "dairy"),
    ("protein",      "snacks"),
    ("energy",       "soft_drinks"),
]

# Pre-compile — specific first, generic second, strict order preserved
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

# Premium brands that command a price premium
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
    Normalise labels field to a lowercase string for keyword matching.
    Handles both list ['en:organic'] and raw string 'en:organic,en:fair-trade'
    from Open Food Facts — format varies by API endpoint.
    """
    if not raw:
        return ""
    if isinstance(raw, list):
        return " ".join(raw).lower()
    return str(raw).lower()


def _parse_weight_g(product: Dict[str, Any]) -> float:
    """
    Extract product weight in grams from weight_g or quantity fields.

    Handles all common OFF quantity formats:
      '500g', '500 g', '1.5kg', '1.5 kg', '1 kg',
      '330ml', '330 ml', '1.5L', '1.5 l', '1 L'

    Falls back to 100g — safe default for per-100g pricing.
    """
    # Direct numeric field
    weight = _safe(product.get("weight_g"), 0.0)
    if weight > 0:
        return weight

    quantity = str(product.get("quantity") or "")

    # kg — must be checked before 'g' to avoid partial match on '1.5kg'
    m = re.search(r"([\d.]+)\s*kg", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000

    # g
    m = re.search(r"([\d.]+)\s*g\b", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1))

    # L — word boundary prevents matching 'ml'
    m = re.search(r"([\d.]+)\s*l\b", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 1000

    # ml
    m = re.search(r"([\d.]+)\s*ml", quantity, re.IGNORECASE)
    if m:
        return float(m.group(1))

    return 100.0


def _category_from_off_tags(product: Dict[str, Any]) -> Optional[str]:
    """
    Attempt to resolve category from OpenFoodFacts structured fields
    before falling back to keyword matching.

    Checks categories_tags (list or comma string) then pnns_groups_2.
    Returns category string or None if no match found.
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
        category = _OFF_CATEGORY_MAP.get(pnns)
        if category:
            return category

    return None


# -----------------------------------------------------------------------
# PRICE INTELLIGENCE ENGINE
# -----------------------------------------------------------------------

class PriceIntelligence:
    """
    Estimates product price and calculates health-per-penny score.

    Usage:
        pi = PriceIntelligence()
        result = pi.analyze(product_dict, health_score=74.0)
    """

    def analyze(
        self,
        product: Dict[str, Any],
        health_score: float,
    ) -> Dict[str, Any]:
        """
        Calculate price intelligence for a product.

        Args:
            product:      cleaned product dict from services/api.py
            health_score: 0-100 score from ml/predict.py

        Returns:
            {
                category              : str
                estimated_price       : float  (GBP per 100g)
                estimated_pack_price  : float  (GBP per pack)
                health_per_penny      : float  (0-100 score)
                nutrition_density     : float  (nutrient quality per kcal)
                value_verdict         : str
                price_band            : str    ('budget' | 'mid' | 'premium')
            }
        """
        category        = self._detect_category(product)
        estimated_price = self._adjust_price(
            _CATEGORY_PRICE_PER_100G.get(category, _CATEGORY_PRICE_PER_100G["default"]),
            product,
        )

        # health_per_penny: health value per penny spent per 100g.
        # Normalised 0-100: raw ratio scaled against typical range 0-3.
        # 3.0 constant is a reasonable hackathon-stage estimate —
        # production version should derive this from dataset percentiles.
        if estimated_price > 0:
            raw_hpp          = health_score / (estimated_price * 100)
            health_per_penny = float(np.clip(raw_hpp / 3.0 * 100, 0, 100))
        else:
            health_per_penny = float(health_score)

        return {
            "category":             category,
            "estimated_price":      round(estimated_price, 2),
            "estimated_pack_price": round(self._estimate_pack_price(product, estimated_price), 2),
            "health_per_penny":     round(health_per_penny, 1),
            "nutrition_density":    round(self._nutrition_density(product), 2),
            "value_verdict":        self._value_verdict(health_per_penny),
            "price_band":           self._price_band(estimated_price),
        }

    # -----------------------------------------------------------------------
    # CATEGORY DETECTION
    # -----------------------------------------------------------------------

    def _detect_category(self, product: Dict[str, Any]) -> str:
        """
        4-stage priority cascade:
          1. OFF categories_tags / pnns_groups_2 (most reliable)
          2. Specific keyword patterns (brand names, multi-word phrases)
          3. Generic keyword patterns (single broad terms)
          4. NOVA group fallback (coarsest signal)
        """
        # Stage 1 — OFF structured taxonomy
        category = _category_from_off_tags(product)
        if category:
            return category

        # Stage 2 + 3 — pre-compiled patterns, specific before generic
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

        # Stage 4 — NOVA group fallback
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
        Adjust base category price using product-level signals.

        NOVA correlation with price is weak — multiplier is small (0.98/0.95)
        to avoid injecting significant bias. Organic and premium brand
        signals are stronger and more reliable.
        """
        price = base

        nova = product.get("nova_group") or 2
        if nova == 4:
            price *= 0.98   # ultra-processed: marginal discount
        elif nova == 1:
            price *= 0.95   # whole/raw foods: slightly cheaper per 100g

        labels = _parse_labels(product.get("labels"))
        if "organic" in labels:
            price *= 1.30   # organic premium (~30% in UK supermarkets)

        brand = (product.get("brand") or "").lower()
        if any(b in brand for b in _PREMIUM_BRANDS):
            price *= 1.35

        return price

    # -----------------------------------------------------------------------
    # NUTRITION DENSITY
    # -----------------------------------------------------------------------

    def _nutrition_density(self, product: Dict[str, Any]) -> float:
        """
        Nutrition density index: positive nutrients minus negative nutrients,
        normalised per 100 kcal. Higher = more nutritious per calorie.

        Sodium: salt / 2.5 (correct conversion — salt = sodium × 2.5).
        Micronutrients default 0 — OFF data often absent, safe fallback.
        """
        kcal      = max(_safe(product.get("energy_kcal"), 100), 50)
        protein   = _safe(product.get("proteins"))
        fiber     = _safe(product.get("fiber"))
        sugar     = _safe(product.get("sugars"))
        sat       = _safe(product.get("saturated_fat"))
        sodium    = _safe(product.get("salt")) / 2.5

        potassium = _safe(product.get("potassium"))
        calcium   = _safe(product.get("calcium"))
        iron      = _safe(product.get("iron"))
        vitamin_c = _safe(product.get("vitamin_c"))

        positive = (
            protein      * 2
            + fiber      * 3
            + potassium  / 100
            + calcium    / 100
            + iron
            + vitamin_c
        )

        negative = (
            sugar   * 2
            + sat   * 1.5
            + sodium / 1000
        )

        return max(0.0, float((positive - negative * 0.5) / kcal * 100))

    # -----------------------------------------------------------------------
    # PACK PRICE / BAND / VERDICT
    # -----------------------------------------------------------------------

    def _estimate_pack_price(
        self, product: Dict[str, Any], price_per_100g: float
    ) -> float:
        """Estimate full pack price from weight and price per 100g."""
        return price_per_100g * (_parse_weight_g(product) / 100)

    def _price_band(self, price_per_100g: float) -> str:
        if price_per_100g < 0.20:
            return "budget"
        if price_per_100g < 0.50:
            return "mid"
        return "premium"

    def _value_verdict(self, health_per_penny: float) -> str:
        if health_per_penny >= 60:
            return "Excellent value"
        if health_per_penny >= 40:
            return "Good value"
        if health_per_penny >= 20:
            return "Average value"
        return "Poor value"