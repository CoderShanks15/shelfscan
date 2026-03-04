import requests         #this is a library that makes HTTP calls like a browser
import streamlit as st  #caching feature

BASE_URL = "https://world.openfoodfacts.org"                    #web address of the database that stores info on most packages of food throughout the world.
HEADERS = {"User-Agent": "ShelfScan/1.0 (hackathon project)"}   #sort of introducing myself to the server for using it's database.
TIMEOUT = 30                                                     #well this prevents hainging my app or site if the server doesnot repond for more than 8 secs.


@st.cache_data(ttl=3600)                             #Streamlit Caching DECORARTOR = store the data for 1hr . lets say if a barcode is read twice then the second time it will show the saved result instead op calling the api again
def fetch_product(barcode: str) -> dict | None:      
    """Fetch product data from Open Food Facts API. Returns a cleaned product dict or None if not found."""

    try:
        url = f"{BASE_URL}/api/v0/product/{barcode}.json"              #builds the url eg->'https://world.openfoodfacts.org/api/v0/product/1234567.json'
        response = requests.get(url, headers=HEADERS, timeout=TIMEOUT) #send the get request to use the database or retrieve a specific data by giving appropriate header and timeout .
        response.raise_for_status()                                    #if the server returns an error then it will give an exception
        data = response.json()

        if data.get("status") != 1:
            return None

        product = data.get("product", {})
        return _clean_product(product)

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


def search_products(query: str, page: int = 1) -> list[dict]:
    """
    Search products by name. Returns list of cleaned product dicts.
    """
    try:
        url = f"{BASE_URL}/cgi/search.pl"
        params = {
            "search_terms": query,
            "json": 1,
            "page": page,
            "page_size": 10,
        }
        response = requests.get(url, headers=HEADERS,
                                params=params, timeout=TIMEOUT)
        response.raise_for_status()
        data = response.json()
        products = data.get("products", [])
        return [_clean_product(p) for p in products]

    except Exception:
        return []


def get_product_image_url(barcode: str) -> str | None:
    """
    Returns the front image URL for a product.
    """
    product = fetch_product(barcode)
    if product and "image_url" in product:
        return product["image_url"]
    return None