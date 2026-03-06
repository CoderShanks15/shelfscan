"""
components/ui.py
================
Reusable Streamlit rendering functions for ShelfScan.

All functions render via st.markdown with HTML for custom styling,
or standard Streamlit widgets where appropriate.

Security: All API-sourced text is escaped via html.escape() before
injection into HTML templates to prevent XSS.
"""

import html
import streamlit as st
from typing import Optional


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _esc(text) -> str:
    """Escape any value for safe HTML injection."""
    return html.escape(str(text)) if text else ""


# -----------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------

def render_hero():
    """Render the app title and tagline."""
    st.markdown("""
    <div class="hero-container animate-in">
        <div class="hero-title">🛒 ShelfScan</div>
        <div class="hero-subtitle">
            Point your phone at a barcode. Get a health verdict in 3 seconds.
            Powered by computer vision, machine learning, and real food data.
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# PRODUCT CARD
# -----------------------------------------------------------------------

def render_product_card(
    product: dict,
    health: dict,
    price: Optional[dict] = None,
    image_result: Optional[dict] = None,
):
    """
    Full product display: name, score bar, verdict badge,
    nutrient breakdown, adjustments trail, and price intelligence.
    """
    name    = _esc(product.get("name", "Unknown Product"))
    brand   = _esc(product.get("brand", ""))
    barcode = _esc(product.get("barcode", ""))
    score   = health.get("score", 0)
    verdict = _esc(health.get("verdict", "OK"))
    emoji   = _esc(health.get("verdict_emoji", ""))
    css_cls = health.get("verdict", "OK").lower().replace(" ", "")

    nutriscore = _esc(product.get('nutriscore', '?'))
    nova = _esc(product.get('nova_group', '?'))
    quantity = _esc(product.get('quantity', ''))

    # --- Header ---
    st.markdown(f"""
    <div class="glass-card animate-in">
        <div class="product-name">{emoji} {name}</div>
        <div class="product-brand">{brand} · {barcode}</div>
        <div class="product-meta">
            <span class="meta-tag">Nutri-Score {nutriscore}</span>
            <span class="meta-tag">NOVA {nova}</span>
            <span class="meta-tag">{quantity}</span>
        </div>
    """, unsafe_allow_html=True)

    # --- Score bar ---
    st.markdown(f"""
        <div class="score-container">
            <div class="score-header">
                <span class="score-value {css_cls}">{score}/100</span>
                <span class="verdict-badge {css_cls}">{verdict}</span>
            </div>
            <div class="score-bar-track">
                <div class="score-bar-fill {css_cls}" style="width: {score}%"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Nutrient breakdown ---
    breakdown = health.get("breakdown", {})
    if breakdown:
        items_html = ""
        label_map = {
            "sugar": "Sugar", "sat_fat": "Sat Fat", "salt": "Salt",
            "fiber": "Fiber", "protein": "Protein", "additives": "Additives",
        }
        for key, label in label_map.items():
            val = _esc(breakdown.get(key, "—"))
            items_html += f"""
                <div class="nutrient-item">
                    <div class="nutrient-label">{label}</div>
                    <div class="nutrient-value">{val}</div>
                </div>"""

        st.markdown(f"""
            <div class="section-header">Nutrient Breakdown</div>
            <div class="nutrient-grid">{items_html}</div>
        """, unsafe_allow_html=True)

    # --- Adjustments trail ---
    adjustments = health.get("adjustments", [])
    if adjustments:
        adj_html = ""
        for adj in adjustments:
            adj_escaped = _esc(adj)
            css = "adj-positive" if adj.startswith("+") else "adj-negative"
            adj_html += f'<div class="adjustment-item {css}">{adj_escaped}</div>'

        st.markdown(f"""
            <div class="section-header">Score Adjustments</div>
            {adj_html}
        """, unsafe_allow_html=True)

    # --- Price intelligence ---
    if price:
        _render_price_section(price)

    # --- Image classification ---
    if image_result:
        _render_image_section(image_result)

    # Close card
    st.markdown("</div>", unsafe_allow_html=True)


def _render_price_section(price: dict):
    """Render price intelligence within a product card."""
    density   = price.get("nutrition_density", 0)
    benchmark = price.get("category_benchmark", 0)
    vs        = price.get("vs_benchmark", 0)
    verdict   = _esc(price.get("value_verdict", ""))
    emoji     = _esc(price.get("value_emoji", ""))
    hpp       = price.get("health_per_penny")

    items_html = f"""
        <div class="nutrient-item">
            <div class="nutrient-label">Nutrition Density</div>
            <div class="nutrient-value">{density:.0f}/100</div>
        </div>
        <div class="nutrient-item">
            <div class="nutrient-label">Category Avg</div>
            <div class="nutrient-value">{benchmark:.0f}/100</div>
        </div>
        <div class="nutrient-item">
            <div class="nutrient-label">vs Benchmark</div>
            <div class="nutrient-value">{"+" if vs >= 0 else ""}{vs:.0f}</div>
        </div>"""

    if hpp is not None:
        items_html += f"""
        <div class="nutrient-item">
            <div class="nutrient-label">Health/Penny</div>
            <div class="nutrient-value">{hpp:.2f}</div>
        </div>"""

    explanation = _esc(price.get('explanation', ''))

    st.markdown(f"""
        <div class="section-header">{emoji} Value — {verdict}</div>
        <div class="nutrient-grid">{items_html}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.3rem;">
            {explanation}
        </div>
    """, unsafe_allow_html=True)


def _render_image_section(image_result: dict):
    """Render image classification results within a product card."""
    category   = _esc(image_result.get("category", "unknown"))
    confidence = image_result.get("confidence", 0)
    clip_on    = image_result.get("clip_available", False)
    processing = _esc(image_result.get("processing_level", "unknown"))
    naturalness = image_result.get("colour_naturalness", 0)
    reasoning  = image_result.get("reasoning", [])

    items_html = f"""
        <div class="nutrient-item">
            <div class="nutrient-label">Category</div>
            <div class="nutrient-value">{category}</div>
        </div>
        <div class="nutrient-item">
            <div class="nutrient-label">Confidence</div>
            <div class="nutrient-value">{confidence:.0%}</div>
        </div>
        <div class="nutrient-item">
            <div class="nutrient-label">Processing</div>
            <div class="nutrient-value">{processing}</div>
        </div>
        <div class="nutrient-item">
            <div class="nutrient-label">Naturalness</div>
            <div class="nutrient-value">{naturalness:.0f}/100</div>
        </div>"""

    method = "CLIP Vision" if clip_on else "Heuristic"

    st.markdown(f"""
        <div class="section-header">🖼️ Image Analysis ({method})</div>
        <div class="nutrient-grid">{items_html}</div>
    """, unsafe_allow_html=True)

    if reasoning:
        for r in reasoning:
            st.markdown(
                f'<div class="adjustment-item">{_esc(r)}</div>',
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------------
# COMPARE PANEL
# -----------------------------------------------------------------------

def render_compare_panel(items: list):
    """Side-by-side comparison of scanned products."""
    if len(items) < 2:
        st.info("Scan at least 2 products and add them to Compare to see a side-by-side view.")
        return

    cols = st.columns(len(items))
    for i, item in enumerate(items):
        with cols[i]:
            render_product_card(
                item["product"],
                item["health"],
                item.get("price"),
            )

    # Summary comparison
    scores = [item["health"].get("score", 0) for item in items]
    names  = [_esc(item["product"].get("name", "Product")) for item in items]
    best_idx = scores.index(max(scores))

    st.markdown(f"""
    <div class="glass-card animate-in" style="text-align: center; margin-top: 1rem;">
        <div style="font-size: 1.1rem; font-weight: 600; color: var(--accent-green);">
            ✅ {names[best_idx]} wins with {scores[best_idx]}/100
        </div>
        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.3rem;">
            Score difference: {max(scores) - min(scores):.1f} points
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# HISTORY TABLE (FIX #22: added timestamps)
# -----------------------------------------------------------------------

def render_history_table(history: list):
    """Render scan history as a styled list with timestamps."""
    if not history:
        st.info("No scan history yet. Scan some products to build your history!")
        return

    st.markdown("""
    <div class="history-row">
        <span>Product</span>
        <span>Score</span>
        <span>Verdict</span>
        <span>When</span>
    </div>
    """, unsafe_allow_html=True)

    for row in history:
        score = row.get("score", 0) or 0
        verdict = row.get("verdict", "")
        css_cls = verdict.lower().replace(" ", "") if verdict else ""
        name = _esc(row.get('name', 'Unknown'))
        scanned_at = _esc(row.get('scanned_at', ''))
        # Show just date portion if available
        date_display = scanned_at[:10] if len(scanned_at) >= 10 else scanned_at

        st.markdown(f"""
        <div class="history-row">
            <span style="color: var(--text-primary);">{name}</span>
            <span class="score-value {css_cls}" style="font-size: 0.9rem;">{score:.0f}</span>
            <span class="verdict-badge {css_cls}" style="font-size: 0.7rem;">{_esc(verdict)}</span>
            <span style="color: var(--text-muted); font-size: 0.75rem;">{date_display}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# FAVOURITES TABLE (FIX #21: new)
# -----------------------------------------------------------------------

def render_favourites_table(favourites: list):
    """Render user's favourite products."""
    if not favourites:
        st.info("No favourites yet. Scan a product and click ❤️ Save Favourite!")
        return

    st.markdown("""
    <div class="history-row">
        <span>Product</span>
        <span>Brand</span>
        <span>Score</span>
        <span>Saved</span>
    </div>
    """, unsafe_allow_html=True)

    for row in favourites:
        score = row.get("score", 0) or 0
        name = _esc(row.get('name', 'Unknown'))
        brand = _esc(row.get('brand', ''))
        added_at = _esc(row.get('added_at', ''))
        date_display = added_at[:10] if len(added_at) >= 10 else added_at

        st.markdown(f"""
        <div class="history-row">
            <span style="color: var(--text-primary);">{name}</span>
            <span style="color: var(--text-secondary);">{brand}</span>
            <span style="font-weight: 600; font-size: 0.9rem;">{score:.0f}</span>
            <span style="color: var(--text-muted); font-size: 0.75rem;">{date_display}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# AUTH SIDEBAR
# -----------------------------------------------------------------------

def render_auth_sidebar(on_login, on_signup, on_logout, logged_in: bool, email: str = ""):
    """
    Render login/signup forms in the sidebar.

    Args:
        on_login:   callable(email, password) → result dict
        on_signup:  callable(email, password) → result dict
        on_logout:  callable() → None
        logged_in:  whether user is currently authenticated
        email:      current user's email if logged in
    """
    with st.sidebar:
        st.markdown("---")

        if logged_in:
            st.markdown(f"""
            <div class="auth-header">👤 {_esc(email)}</div>
            """, unsafe_allow_html=True)

            if st.button("🚪 Logout", use_container_width=True):
                on_logout()
                st.rerun()
        else:
            auth_tab = st.radio(
                "Account",
                ["Login", "Sign Up"],
                horizontal=True,
                label_visibility="collapsed",
            )

            email_input = st.text_input("Email", key="auth_email")
            password_input = st.text_input("Password", type="password", key="auth_pass")

            if auth_tab == "Login":
                if st.button("Login", use_container_width=True):
                    if email_input and password_input:
                        result = on_login(email_input, password_input)
                        if result.get("ok"):
                            st.success("Logged in!")
                            st.rerun()
                        else:
                            st.error(result.get("error", "Login failed"))
                    else:
                        st.warning("Enter email and password")
            else:
                if st.button("Sign Up", use_container_width=True):
                    if email_input and password_input:
                        result = on_signup(email_input, password_input)
                        if result.get("ok"):
                            st.success("Account created!")
                            st.rerun()
                        else:
                            st.error(result.get("error", "Signup failed"))
                    else:
                        st.warning("Enter email and password")
