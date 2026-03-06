"""
app.py
======
ShelfScan — Streamlit entry point.

Orchestrates the full pipeline:
  Photo/barcode → CV decode → Open Food Facts → ML score → CLIP classify
  → Price intelligence → Styled UI

Run:
    streamlit run app.py
"""

import logging
import streamlit as st

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="ShelfScan — Health Verdict",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------
# IMPORTS (after page config)
# -----------------------------------------------------------------------
from PIL import Image

from components.styles import inject_css
from components.ui import (
    render_hero,
    render_product_card,
    render_compare_panel,
    render_history_table,
    render_favourites_table,
    render_auth_sidebar,
)
from session.session import (
    init_session,
    is_logged_in,
    set_logged_in,
    logout as session_logout,
    get_user_id,
    get_token,
    set_scan_result,
    get_scan_result,
    clear_scan,
    add_to_compare,
    get_compare_items,
    clear_compare,
)
from services.api import fetch_product
from services.barcode import decode_barcode_from_image, validate_barcode
from ml.predict import predict_health
from ml.price_intelligence import PriceIntelligence
from auth.auth import login, signup, revoke_token
from database.db import (
    save_scan, get_history, save_favourite, remove_favourite,
    is_favourite, get_favourites,
)


# -----------------------------------------------------------------------
# CACHED RESOURCES
# -----------------------------------------------------------------------

@st.cache_resource
def load_image_classifier():
    """Load CLIP + OpenCV classifier once per process."""
    try:
        from ml.image_classifier import ProductImageClassifier
        return ProductImageClassifier()
    except Exception as e:
        st.warning(f"Image classifier unavailable: {e}")
        return None


@st.cache_resource
def load_price_engine():
    """Load price intelligence engine once per process."""
    return PriceIntelligence()


# -----------------------------------------------------------------------
# AUTH HANDLERS
# -----------------------------------------------------------------------

def _handle_login(email: str, password: str) -> dict:
    result = login(email, password)
    if result.get("ok"):
        set_logged_in(result["user_id"], result["email"], result["token"])
    return result


def _handle_signup(email: str, password: str) -> dict:
    result = signup(email, password)
    if result.get("ok"):
        set_logged_in(result["user_id"], result["email"], result["token"])
    return result


def _handle_logout():
    token = get_token()
    if token:
        revoke_token(token)
    session_logout()


# -----------------------------------------------------------------------
# SCAN PIPELINE
# -----------------------------------------------------------------------

def run_scan_pipeline(barcode: str, uploaded_image: Image.Image | None = None):
    """Execute the full scan pipeline for a barcode."""

    with st.spinner("🔍 Fetching product data..."):
        product = fetch_product(barcode)

    # Handle missing product or API errors
    if not product or not isinstance(product, dict):
        st.error(
            f"Product not found for barcode **{barcode}**. "
            f"[Add it to Open Food Facts →]"
            f"(https://world.openfoodfacts.org/cgi/product.pl?code={barcode})"
        )
        return

    if "error" in product:
        st.error(f"API error: {product['error']}")
        return

    # --- ML health scoring (with error boundary) ---
    try:
        with st.spinner("🧬 Calculating health score..."):
            health = predict_health(product)
    except FileNotFoundError:
        st.error(
            "⚠️ Health model not found. Ensure `models/health_model.pkl` exists. "
            "Run `python ml/health_model.py` to train it."
        )
        return
    except Exception as e:
        logger.exception("predict_health failed")
        st.error(f"⚠️ Health scoring failed: {e}")
        return

    # --- Image classification (if image provided) ---
    image_result = None
    classifier = load_image_classifier()
    if classifier and uploaded_image:
        try:
            with st.spinner("🖼️ Analysing image..."):
                image_result = classifier.classify(uploaded_image, product)
        except Exception as e:
            logger.warning("Image classification failed: %s", e)

    # --- Price intelligence ---
    category = image_result.get("category") if image_result else None
    price_engine = load_price_engine()
    price = price_engine.analyze(product, health["score"], category)

    # --- Store in session ---
    set_scan_result(product, health, price, image_result)

    # --- Save to history (logged-in users) ---
    if is_logged_in():
        save_scan(get_user_id(), product, health["score"], health["verdict"])


# -----------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------

def main():
    # Initialise
    init_session()
    inject_css()

    # Auth sidebar — FIX #1: use correct session key "email" not "user_email"
    render_auth_sidebar(
        on_login=_handle_login,
        on_signup=_handle_signup,
        on_logout=_handle_logout,
        logged_in=is_logged_in(),
        email=st.session_state.get("email", ""),
    )

    # Sidebar extras
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            '<div style="font-size: 0.75rem; color: var(--text-muted);">'
            '🛒 ShelfScan<br>'
            'Built by codershanks</div>',
            unsafe_allow_html=True,
        )

    # Hero
    render_hero()

    # Tabs — FIX #21: added Favourites tab
    tab_scan, tab_compare, tab_favs, tab_history = st.tabs(
        ["🔍 Scan", "⚖️ Compare", "❤️ Favourites", "📋 History"]
    )

    # ---- SCAN TAB ----
    with tab_scan:
        col_input, col_or, col_manual = st.columns([2, 0.3, 2])

        with col_input:
            # FIX #18: Camera input + file upload
            input_mode = st.radio(
                "Input method",
                ["📸 Upload photo", "📷 Camera"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if input_mode == "📷 Camera":
                uploaded = st.camera_input(
                    "Take a photo of the barcode",
                    help="Point your camera at a product barcode.",
                )
            else:
                uploaded = st.file_uploader(
                    "📸 Upload a barcode photo",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Take a photo of any product barcode.",
                )

        with col_or:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:100%;color:var(--text-muted);font-size:0.85rem;">or</div>',
                unsafe_allow_html=True,
            )

        with col_manual:
            manual_barcode = st.text_input(
                "⌨️ Enter barcode manually",
                placeholder="e.g. 3017620422003",
                help="Type or paste an EAN-13, UPC-A, or EAN-8 barcode number.",
            )

        # FIX #5: Guard against double scan on rerun.
        # Only run the pipeline if we don't already have a result for this input.
        already_scanned = st.session_state.get("current_product") is not None

        # --- Process uploaded image ---
        if uploaded is not None and not already_scanned:
            image = Image.open(uploaded)

            # Show preview
            with st.expander("📷 Image preview", expanded=False):
                st.image(image, use_container_width=True)

            with st.spinner("📱 Decoding barcode from image..."):
                barcode = decode_barcode_from_image(image)

            if barcode:
                st.success(f"Barcode detected: **{barcode}**")
                run_scan_pipeline(barcode, uploaded_image=image)
            else:
                st.warning(
                    "No barcode found in image. Try a clearer photo, "
                    "or enter the barcode manually below."
                )

        # --- Process manual barcode ---
        if manual_barcode and not already_scanned:
            barcode = manual_barcode.strip()
            if validate_barcode(barcode):
                run_scan_pipeline(barcode)
            else:
                st.error("Invalid barcode. Enter an 8–13 digit EAN/UPC code.")

        # --- Display current result ---
        product, health, price, image_result = get_scan_result()
        if product and health:
            st.markdown("---")
            render_product_card(product, health, price, image_result)

            # Action buttons
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("⚖️ Add to Compare", use_container_width=True):
                    add_to_compare(product, health, price)
                    st.success("Added to compare!")

            with col_b:
                if is_logged_in():
                    uid = get_user_id()
                    bc = product.get("barcode", "")
                    if is_favourite(uid, bc):
                        if st.button("💔 Remove Favourite", use_container_width=True):
                            remove_favourite(uid, bc)
                            st.rerun()
                    else:
                        if st.button("❤️ Save Favourite", use_container_width=True):
                            save_favourite(uid, product, health["score"])
                            st.rerun()

            with col_c:
                if st.button("🔄 New Scan", use_container_width=True):
                    clear_scan()
                    st.rerun()

    # ---- COMPARE TAB ----
    with tab_compare:
        items = get_compare_items()

        if items:
            render_compare_panel(items)

            if st.button("🗑️ Clear Compare List", use_container_width=True):
                clear_compare()
                st.rerun()
        else:
            st.info(
                "No products to compare yet. Scan a product and click "
                "**Add to Compare** to start comparing."
            )

    # ---- FAVOURITES TAB ---- (FIX #21)
    with tab_favs:
        if is_logged_in():
            favs = get_favourites(get_user_id())
            render_favourites_table(favs)
        else:
            st.info("Log in to see your favourite products.")

    # ---- HISTORY TAB ----
    with tab_history:
        if is_logged_in():
            history = get_history(get_user_id())
            render_history_table(history)
        else:
            st.info("Log in to see your scan history.")


# -----------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------

if __name__ == "__main__":
    main()
