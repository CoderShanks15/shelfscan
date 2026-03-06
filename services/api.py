import requests
import streamlit as st

BASE_URL = "https://world.openfoodfacts.org"
HEADERS = {"User-Agent": "ShelfScan/1.0 (hackathon project)"}
TIMEOUT = 30


def fetch_product(barcode: str) -> dict | None:
    """Fetch product data from Open Food Facts API.
    Returns a cleaned product dict, an error dict, or None if not found.

    Successful results are cached in session state for the session lifetime.
    Errors are NOT cached so they can be retried immediately.
    """

    # Check session-level cache first (Bug 2 fix)
    cache = st.session_state.get("_product_cache")
    if cache is None:
        cache = {}
        st.session_state["_product_cache"] = cache

    if barcode in cache:
        return cache[barcode]

    try:
        url = f"{BASE_URL}/api/v0/product/{barcode}.json"
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != 1:
            return None

        product = data.get("product", {})
        cleaned = _clean_product(product)

        # Cache only successful results
        cache[barcode] = cleaned
        return cleaned

    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "No internet connection."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def _clean_product(raw: dict) -> dict:
    """
    Extract and clean only the fields ShelfScan needs.
    """
    nutriments = raw.get("nutriments", {})

    return {
        # Identity
        "barcode":        raw.get("code", ""),
        "name":           raw.get("product_name", "Unknown Product"),
        "brand":          raw.get("brands", "Unknown Brand"),
        "quantity":       raw.get("quantity", ""),

        # Nutrition scores
        "nutriscore":     raw.get("nutriscore_grade", "").upper(),
        "nova_group":     raw.get("nova_group", None),
        "ecoscore":       raw.get("ecoscore_grade", "").upper(),

        # Nutrients per 100g
        "energy_kcal":    nutriments.get("energy-kcal_100g", 0),
        "fat":            nutriments.get("fat_100g", 0),
        "saturated_fat":  nutriments.get("saturated-fat_100g", 0),
        "carbohydrates":  nutriments.get("carbohydrates_100g", 0),
        "sugars":         nutriments.get("sugars_100g", 0),
        "fiber":          nutriments.get("fiber_100g", 0),
        "proteins":       nutriments.get("proteins_100g", 0),
        "salt":           nutriments.get("salt_100g", 0),

        # Extra info
        "allergens":      raw.get("allergens_tags", []),
        "additives":      raw.get("additives_tags", []),
        "labels":         raw.get("labels_tags", []),
        "ingredients":    raw.get("ingredients_text", ""),
        "image_url":      raw.get("image_front_url", ""),
        "stores":         raw.get("stores", ""),
    }