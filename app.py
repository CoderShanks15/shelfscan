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
    page_icon="S",
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
    render_skeleton,
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
from services.barcode import decode_barcode_from_image
from ml.predict import predict_health
from ml.price_intelligence import PriceIntelligence
from auth.auth import login, signup, revoke_token
from auth.oauth import (
    google_available, github_available,
    google_auth_url, github_auth_url,
    google_exchange, github_exchange,
)
from database.db import (
    save_scan, get_history, save_favourite, remove_favourite,
    is_favourite, get_favourites, create_user, get_user_by_email,
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


def _handle_oauth_callback():
    """
    Process OAuth callback query params.
    After Google/GitHub redirects back with ?code=xxx&state=provider,
    exchange the code for user info and log them in.
    """
    params = st.query_params
    code     = params.get("code")
    provider = params.get("state")  # "google" or "github"

    if not code or not provider:
        return

    # Already logged in — clear params and return
    if is_logged_in():
        st.query_params.clear()
        return

    # Exchange code for user info
    if provider == "google" and google_available():
        result = google_exchange(code)
    elif provider == "github" and github_available():
        result = github_exchange(code)
    else:
        st.query_params.clear()
        return

    st.query_params.clear()

    if not result.get("ok"):
        st.error(result.get("error", "OAuth login failed."))
        return

    email = result["email"]
    if not email:
        st.error("Could not get email from OAuth provider.")
        return

    # Find or create user
    from auth.auth import _hash_password, _create_token
    import secrets

    user = get_user_by_email(email)
    if user:
        user_id = user["id"]
    else:
        # Create user with a random password (they log in via OAuth)
        random_pw = secrets.token_urlsafe(32)
        pw_hash = _hash_password(random_pw)
        user_id = create_user(email, pw_hash)
        if user_id is None:
            st.error("Failed to create account.")
            return

    token = _create_token(user_id, email)
    set_logged_in(user_id, email, token)
    st.rerun()


# -----------------------------------------------------------------------
# SCAN PIPELINE
# -----------------------------------------------------------------------

def run_scan_pipeline(barcode: str, uploaded_image: Image.Image | None = None):
    """Execute the full scan pipeline for a barcode."""

    with st.spinner("Fetching product data..."):
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
        with st.spinner("Calculating health score..."):
            health = predict_health(product)
    except FileNotFoundError:
        st.error(
            "Health model not found. Ensure `models/health_model.pkl` exists. "
            "Run `python ml/health_model.py` to train it."
        )
        return
    except Exception as e:
        logger.exception("predict_health failed")
        st.error(f"Health scoring failed: {e}")
        return

    # --- Image classification (if image provided) ---
    image_result = None
    classifier = load_image_classifier()
    if classifier and uploaded_image:
        try:
            with st.spinner("Analysing image..."):
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

    # Handle OAuth callback (before rendering UI)
    _handle_oauth_callback()

    # Inject Custom Animated Motion Graphics Background (Pure CSS Mesh Gradient)
    import streamlit.components.v1 as components
    components.html("""
        <style>
            body { margin: 0; padding: 0; overflow: hidden; background-color: #0b0d1a !important; }
            .bg-animation {
                position: fixed;
                top: 0; left: 0; width: 100vw; height: 100vh;
                z-index: -999;
                pointer-events: none;
                overflow: hidden;
            }
            .color-blob {
                position: absolute;
                border-radius: 50%;
                filter: blur(80px); /* Slightly lower blur for more definition */
                opacity: 0.85; /* Increased opacity dramatically so it's super visible */
                animation: float 20s infinite ease-in-out alternate;
            }
            .blob-1 {
                top: -10%; left: -10%; width: 55vw; height: 55vw;
                background: radial-gradient(circle, rgba(52,211,153,0.3) 0%, rgba(52,211,153,0) 70%); /* Stronger teal */
                animation-delay: 0s;
            }
            .blob-2 {
                bottom: -20%; right: -10%; width: 65vw; height: 65vw;
                background: radial-gradient(circle, rgba(129,140,248,0.3) 0%, rgba(129,140,248,0) 70%); /* Stronger purple/indigo */
                animation-delay: -5s;
                animation-duration: 25s;
            }
            .blob-3 {
                top: 40%; left: 30%; width: 45vw; height: 45vw;
                background: radial-gradient(circle, rgba(45,212,191,0.25) 0%, rgba(45,212,191,0) 70%); /* Stronger cyan */
                animation-delay: -10s;
                animation-duration: 30s;
            }
            @keyframes float {
                0% { transform: translate(0, 0) scale(1) rotate(0deg); }
                33% { transform: translate(5%, 10%) scale(1.1) rotate(10deg); }
                66% { transform: translate(-5%, 5%) scale(0.9) rotate(-10deg); }
                100% { transform: translate(0, -10%) scale(1.05) rotate(5deg); }
            }
        </style>
        <div class="bg-animation">
            <div class="color-blob blob-1"></div>
            <div class="color-blob blob-2"></div>
            <div class="color-blob blob-3"></div>
        </div>
    """, height=0)

    # Auth sidebar
    render_auth_sidebar(
        on_login=_handle_login,
        on_signup=_handle_signup,
        on_logout=_handle_logout,
        logged_in=is_logged_in(),
        email=st.session_state.get("email", ""),
        google_url=google_auth_url() if google_available() else None,
        github_url=github_auth_url() if github_available() else None,
    )

    # Sidebar extras
    with st.sidebar:
        st.markdown("---")

        # --- Theme toggle DISABLED (light mode off) ---
        # To re-enable: uncomment the block below
        # import streamlit.components.v1 as components
        # components.html("""
        # <style>
        #     body { margin: 0; background: transparent; overflow: hidden; }
        #     .toggle-btn {
        #         display: inline-flex;
        #         align-items: center;
        #         justify-content: center;
        #         gap: 6px;
        #         padding: 7px 16px;
        #         border-radius: 100px;
        #         background: rgba(128,128,128,0.1);
        #         border: 1px solid rgba(128,128,128,0.2);
        #         color: inherit;
        #         font-size: 13px;
        #         font-weight: 600;
        #         font-family: 'Inter', -apple-system, sans-serif;
        #         cursor: pointer;
        #         transition: all 0.15s ease;
        #         width: 100%;
        #         box-sizing: border-box;
        #     }
        #     .toggle-btn:hover {
        #         background: rgba(128,128,128,0.2);
        #         transform: translateY(-1px);
        #     }
        # </style>
        # <button class="toggle-btn" id="themeBtn" onclick="toggleTheme()">
        #     &#9790; Switch Theme
        # </button>
        # <script>
        #     function getTheme() {
        #         try {
        #             return window.parent.document.documentElement.getAttribute('data-theme')
        #                 || localStorage.getItem('shelfscan-theme')
        #                 || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
        #         } catch(e) { return 'light'; }
        #     }
        #     function updateLabel() {
        #         var t = getTheme();
        #         var btn = document.getElementById('themeBtn');
        #         btn.innerHTML = (t === 'dark') ? '&#9788; Light Mode' : '&#9790; Dark Mode';
        #     }
        #     function toggleTheme() {
        #         var current = getTheme();
        #         var next = (current === 'dark') ? 'light' : 'dark';
        #         try {
        #             window.parent.document.documentElement.setAttribute('data-theme', next);
        #         } catch(e) {}
        #         localStorage.setItem('shelfscan-theme', next);
        #         updateLabel();
        #     }
        #     updateLabel();
        # </script>
        # """, height=42)
        # --- End theme toggle DISABLED ---

        st.markdown(
            '<div style="font-size: 0.75rem; color: var(--text-muted);">'
            'ShelfScan<br>'
            'Built by codershanks</div>',
            unsafe_allow_html=True,
        )

    # Hero
    render_hero()

    # Tabs — FIX #21: added Favourites tab
    tab_scan, tab_compare, tab_favs, tab_history = st.tabs(
        ["Scan", "Compare", "Favourites", "History"]
    )

    # ---- SCAN TAB ----
    with tab_scan:
        # BUG 8 FIX: File uploader key counter — "New Scan" increments
        # this to reset the widget and clear the uploaded file.
        upload_key = st.session_state.get("_upload_key", 0)

        # BUG 1 FIX: Track scan *attempts*, not just successes.
        already_scanned = (
            st.session_state.get("current_product") is not None
            or st.session_state.get("_scan_attempted", False)
        )

        col_input, col_or, col_manual = st.columns([2, 0.3, 2])

        with col_input:
            # FIX #18: Camera input + file upload
            input_mode = st.radio(
                "Input method",
                ["Upload Photo", "Camera"],
                horizontal=True,
                label_visibility="collapsed",
            )

            if input_mode == "Camera":
                uploaded = st.camera_input(
                    "Capture barcode",
                    help="Point your camera at a product barcode.",
                )
            else:
                # The native file uploader has been restyled purely via CSS to be a unified,
                # curvy dropzone with an integrated Browse Files button.
                pass
                uploaded = st.file_uploader(
                    "Upload barcode",
                    type=["jpg", "jpeg", "png", "webp"],
                    help="Upload a photo of any product barcode.",
                    label_visibility="collapsed",
                    key=f"file_upload_{upload_key}",
                )

        with col_or:
            st.markdown(
                '<div style="display:flex;align-items:center;justify-content:center;'
                'height:100%;color:var(--text-muted);font-size:0.78rem;'
                'text-transform:uppercase;letter-spacing:1px;">or</div>',
                unsafe_allow_html=True,
            )

        with col_manual:
            manual_barcode = st.text_input(
                "Enter barcode manually",
                placeholder="e.g. 3017620422003",
                help="Type or paste an EAN-13, UPC-A, or EAN-8 barcode number.",
            )



        # --- Process uploaded image ---
        if uploaded is not None and not already_scanned:
            image = Image.open(uploaded)

            # Show preview
            with st.expander("Image preview", expanded=False):
                st.image(image, use_container_width=True)

            with st.spinner("Decoding barcode..."):
                scan_placeholder = st.empty()
                scan_placeholder.markdown("")
                barcode = decode_barcode_from_image(image)

            if barcode:
                st.success(f"Barcode detected: **{barcode}**")
                st.session_state["_scan_attempted"] = True
                run_scan_pipeline(barcode, uploaded_image=image)
            else:
                st.session_state["_scan_attempted"] = True
                st.warning(
                    "No barcode found in image. Try a clearer photo, "
                    "or enter the barcode manually below."
                )

        # --- Process manual barcode ---
        if manual_barcode and not already_scanned:
            barcode = manual_barcode.strip()
            # BUG 6 FIX: Accept any 8-14 digit string. Checksum validation
            # was too strict — many valid Open Food Facts products have
            # non-standard barcodes that fail checksum.
            if barcode.isdigit() and 8 <= len(barcode) <= 14:
                st.session_state["_scan_attempted"] = True
                run_scan_pipeline(barcode)
            else:
                st.error("Invalid barcode. Enter an 8–14 digit barcode number.")

        # --- Display current result ---
        product, health, price, image_result = get_scan_result()
        if product and health:
            st.markdown("---")
            render_product_card(product, health, price, image_result)

            # Action buttons
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                if st.button("Add to Compare", use_container_width=True):
                    add_to_compare(product, health, price)
                    st.success("Added to compare")

            with col_b:
                if is_logged_in():
                    uid = get_user_id()
                    bc = product.get("barcode", "")
                    if is_favourite(uid, bc):
                        if st.button("Remove Favourite", use_container_width=True):
                            remove_favourite(uid, bc)
                            st.rerun()
                    else:
                        if st.button("Save Favourite", use_container_width=True):
                            save_favourite(uid, product, health["score"])
                            st.rerun()
                else:
                    # BUG 4 FIX: inform user why button is missing
                    st.button(
                        "Save Favourite",
                        use_container_width=True,
                        disabled=True,
                        help="Log in to save favourites",
                    )

            with col_c:
                if st.button("New Scan", use_container_width=True):
                    clear_scan()
                    # BUG 8 FIX: increment upload key to reset file_uploader
                    st.session_state["_upload_key"] = upload_key + 1
                    st.session_state["_scan_attempted"] = False
                    st.rerun()

    # ---- COMPARE TAB ----
    with tab_compare:
        items = get_compare_items()

        if items:
            render_compare_panel(items)

            if st.button("Clear Compare List", use_container_width=True):
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
