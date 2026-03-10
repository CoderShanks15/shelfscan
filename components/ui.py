"""
components/ui.py
================
Reusable Streamlit rendering functions for ShelfScan.

All functions render via st.markdown with HTML for custom styling,
or standard Streamlit widgets where appropriate.

Security: All API-sourced text is escaped via html.escape() before
injection into HTML templates to prevent XSS.

UI Elements:
  - Circular SVG score ring
  - Pill tags for labels/allergens
  - Skeleton loading placeholders
  - Glassmorphism product cards
"""

import html
import math
import streamlit as st
from typing import Optional


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------

def _esc(text) -> str:
    """Escape any value for safe HTML injection."""
    return html.escape(str(text)) if text else ""


def _verdict_class(verdict: str) -> str:
    """Map verdict string to CSS class."""
    return (verdict or "ok").lower().replace(" ", "")


# -----------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------

def render_hero():
    """Render the premium hero section with features and trust indicators."""
    import streamlit.components.v1 as components

    # Title + subtitle — works fine in st.markdown
    st.markdown("""
    <div class="hero-container animate-in">
        <div class="hero-title">ShelfScan</div>
        <div class="hero-subtitle">
            Scan any barcode. Get an instant health verdict powered by
            computer vision, ML scoring, and real food data.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards + stats — rendered via components.html for full HTML support
    components.html("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            background: transparent; font-family: 'Inter', sans-serif; overflow: hidden; color: #eef2f7; 
        }
        body::after {
            content: ''; position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background-image: 
                url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.08'/%3E%3C/svg%3E"),
                radial-gradient(circle, rgba(255,255,255,0.05) 1px, transparent 1px);
            background-size: 200px 200px, 24px 24px; opacity: 0.8; mix-blend-mode: overlay; pointer-events: none; z-index: 0;
        }

        .features-row {
            display: grid; grid-template-columns: repeat(3, 1fr);
            gap: 12px; padding: 0 4px; margin-bottom: 12px; position: relative; z-index: 1;
        }
        .feature-card {
            background: rgba(255,255,255,0.03);
            border: 1px solid transparent;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08), 0 2px 8px rgba(0,0,0,0.4), 0 0 1px rgba(52,211,153,0.06);
            border-radius: 14px; padding: 18px 12px; text-align: center;
            transition: all 0.25s ease; position: relative; overflow: hidden;
            backdrop-filter: blur(12px);
        }
        .feature-card:hover { 
            transform: translateY(-3px); 
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.12), 0 8px 32px rgba(52,211,153,0.12), 0 0 1px rgba(52,211,153,0.2); 
            background: rgba(255,255,255,0.05);
        }

        .feature-icon {
            width: 40px; height: 40px; border-radius: 10px;
            background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(45,212,191,0.1));
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 1.1rem; margin-bottom: 8px; color: #34d399;
        }
        .feature-title { font-size: 0.8rem; font-weight: 700; color: #eef2f7; margin-bottom: 3px; }
        .feature-desc { font-size: 0.7rem; color: #9ba3b5; line-height: 1.4; }

        .stats-row {
            display: flex; justify-content: center; gap: 40px; padding: 6px 0;
        }
        .stat-number { font-size: 1.3rem; font-weight: 800; color: #34d399; letter-spacing: -0.5px; }
        .stat-label { font-size: 0.62rem; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }

        /* Light mode styles (DISABLED — uncomment when re-enabling light mode)
        body.light .feature-card { background: rgba(0,0,0,0.03); border-color: rgba(0,0,0,0.06); }
        body.light .feature-card:hover { box-shadow: 0 8px 24px rgba(13,148,136,0.12); }
        body.light .feature-icon { background: linear-gradient(135deg, rgba(13,148,136,0.15), rgba(16,185,129,0.10)); color: #0d9488; }
        body.light .feature-title { color: #1a1a2e; }
        body.light .feature-desc { color: #555; }
        body.light .stat-number { color: #0d9488; }
        body.light .stat-label { color: #777; }
        */
    </style>
    <script>
        // Dark mode is always on — no detection needed
        // When re-enabling light mode, restore the theme detection logic here
    </script>
    <div class="features-row">
        <div class="feature-card">
            <div class="feature-icon">&#x2750;</div>
            <div class="feature-title">Instant Scan</div>
            <div class="feature-desc">Photo or camera barcode detection with real-time API lookup</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#x2261;</div>
            <div class="feature-title">Health Scoring</div>
            <div class="feature-desc">ML-powered 0-100 score with nutrient analysis and adjustments</div>
        </div>
        <div class="feature-card">
            <div class="feature-icon">&#x29D6;</div>
            <div class="feature-title">Compare + Track</div>
            <div class="feature-desc">Side-by-side product showdown, history log, and favourites</div>
        </div>
    </div>
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-number">2.5M+</div>
            <div class="stat-label">Products</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">3 sec</div>
            <div class="stat-label">Avg Scan</div>
        </div>
        <div class="stat-item">
            <div class="stat-number">12+</div>
            <div class="stat-label">Nutrients</div>
        </div>
    </div>
    """, height=200)


# -----------------------------------------------------------------------
# SKELETON LOADER
# -----------------------------------------------------------------------

def render_skeleton():
    """Show skeleton loading placeholders while results load."""
    st.markdown("""
    <div class="glass-card">
        <div class="skeleton skeleton-line" style="width: 60%"></div>
        <div class="skeleton skeleton-line short"></div>
        <div style="display: flex; gap: 1.5rem; margin: 1.5rem 0;">
            <div class="skeleton skeleton-circle"></div>
            <div style="flex:1;">
                <div class="skeleton skeleton-line full"></div>
                <div class="skeleton skeleton-line" style="width: 80%"></div>
                <div class="skeleton skeleton-line" style="width: 50%"></div>
            </div>
        </div>
        <div class="skeleton skeleton-line full"></div>
        <div class="skeleton skeleton-line" style="width: 75%"></div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# SCORE RING (SVG circular gauge)
# -----------------------------------------------------------------------

def _render_score_ring(score: float, verdict: str, emoji: str) -> str:
    """Generate SVG circular score ring HTML."""
    radius = 45
    circumference = 2 * math.pi * radius
    pct = max(0, min(100, score)) / 100
    offset = circumference * (1 - pct)
    css_cls = _verdict_class(verdict)

    verdict_descriptions = {
        "great":   "Excellent choice — minimal processing, strong nutrition",
        "ok":      "Decent option — moderate nutrition profile",
        "caution": "Use sparingly — some nutritional concerns",
        "avoid":   "Poor choice — high in negatives, low in positives",
    }
    sub = verdict_descriptions.get(css_cls, "")

    return f"""
    <div class="score-ring-container animate-scale">
        <div class="score-ring">
            <svg viewBox="0 0 110 110">
                <circle class="score-ring-bg" cx="55" cy="55" r="{radius}"/>
                <circle class="score-ring-fill {css_cls}" cx="55" cy="55" r="{radius}"
                    stroke-dasharray="{circumference}"
                    stroke-dashoffset="{offset}"/>
            </svg>
            <div class="score-ring-value {css_cls}">{score:.0f}</div>
        </div>
        <div class="score-info">
            <div class="verdict-label {css_cls}">{_esc(emoji)} {_esc(verdict)}</div>
            <div class="verdict-sub">{sub}</div>
        </div>
    </div>"""


# -----------------------------------------------------------------------
# PILL TAGS
# -----------------------------------------------------------------------

def _render_pills(product: dict) -> str:
    """Render label/allergen pill tags."""
    pills = []

    # Labels
    labels = product.get("labels", [])
    label_styles = {
        "organic": ("Organic", "green"),
        "vegan": ("Vegan", "green"),
        "vegetarian": ("Vegetarian", "green"),
        "gluten-free": ("Gluten-Free", "blue"),
        "no-gluten": ("Gluten-Free", "blue"),
        "no-palm-oil": ("No Palm Oil", "green"),
        "wholegrain": ("Wholegrain", "green"),
        "whole-grain": ("Wholegrain", "green"),
        "fair-trade": ("Fair Trade", "purple"),
        "halal": ("Halal", "blue"),
        "kosher": ("Kosher", "blue"),
    }
    for label in labels:
        label_lower = label.lower().split(":")[-1]  # remove "en:" prefix
        if label_lower in label_styles:
            text, color = label_styles[label_lower]
            pills.append(f'<span class="pill pill-{color}">{text}</span>')

    # Allergens
    allergens = product.get("allergens", [])
    for allergen in allergens[:5]:  # limit to 5
        name = _esc(allergen.split(":")[-1].replace("-", " ").title())
        pills.append(f'<span class="pill pill-red">{name}</span>')

    if not pills:
        return ""

    return f'<div style="margin: 0.5rem 0; display: flex; flex-wrap: wrap; gap: 3px;">{"".join(pills)}</div>'


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
    Full product display: name, score ring, verdict,
    nutrient breakdown, adjustments trail, and price intelligence.
    """
    name    = _esc(product.get("name", "Unknown Product"))
    brand   = _esc(product.get("brand", ""))
    barcode = _esc(product.get("barcode", ""))
    score   = health.get("score", 0)
    verdict = health.get("verdict", "OK")
    emoji   = health.get("verdict_emoji", "")

    nutriscore = _esc(product.get('nutriscore', '?'))
    nova = _esc(product.get('nova_group', '?'))
    quantity = _esc(product.get('quantity', ''))

    # --- Card open ---
    st.markdown(f"""
    <div class="glass-card animate-in">
        <div class="product-name">{_esc(emoji)} {name}</div>
        <div class="product-brand">{brand} · {barcode}</div>
        <div class="product-meta">
            <span class="meta-tag">Nutri-Score {nutriscore}</span>
            <span class="meta-tag">NOVA {nova}</span>
            {'<span class="meta-tag">' + quantity + '</span>' if quantity else ''}
        </div>
    """, unsafe_allow_html=True)

    # --- Pill tags ---
    pills_html = _render_pills(product)
    if pills_html:
        st.markdown(pills_html, unsafe_allow_html=True)

    # --- Score ring ---
    ring_html = _render_score_ring(score, verdict, emoji)
    st.markdown(ring_html, unsafe_allow_html=True)

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

        # NOVA group
        nova_val = breakdown.get("nova", "?")
        items_html += f"""
            <div class="nutrient-item">
                <div class="nutrient-label">NOVA</div>
                <div class="nutrient-value">{_esc(nova_val)}</div>
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
    p_emoji   = _esc(price.get("value_emoji", ""))
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
        <div class="section-header">{p_emoji} Value — {verdict}</div>
        <div class="nutrient-grid">{items_html}</div>
        <div style="font-size: 0.78rem; color: var(--text-muted); margin-top: 0.5rem;
                    padding: 6px 10px; border-radius: var(--radius-sm);
                    background: rgba(255,255,255,0.02);">
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
        <div class="section-header">Image Analysis ({method})</div>
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
        st.info("Scan at least 2 products and add them to Compare.")
        return

    cols = st.columns(len(items))
    for i, item in enumerate(items):
        with cols[i]:
            render_product_card(
                item["product"],
                item["health"],
                item.get("price"),
            )

    # Summary — only show winner when comparing 2+ items
    if len(items) >= 2:
        scores = [item["health"].get("score", 0) for item in items]
        names  = [_esc(item["product"].get("name", "Product")) for item in items]
        best_idx = scores.index(max(scores))

        st.markdown(f"""
        <div class="glass-card animate-in" style="text-align: center; margin-top: 0.5rem;">
            <div style="font-size: 1.1rem; font-weight: 700; color: var(--accent-green);">
                {names[best_idx]} wins with {scores[best_idx]:.0f}/100
            </div>
            <div style="font-size: 0.82rem; color: var(--text-secondary); margin-top: 0.3rem;">
                Score difference: {max(scores) - min(scores):.1f} points
            </div>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# HISTORY TABLE (with timestamps)
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
        try:
            score = float(row.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        verdict = row.get("verdict", "")
        css_cls = _verdict_class(verdict)
        name = _esc(row.get('name', 'Unknown'))
        scanned_at = row.get('scanned_at') or ''
        date_display = str(scanned_at)[:10] if scanned_at else '—'

        st.markdown(f"""
        <div class="history-row">
            <span style="color: var(--text-primary); font-weight: 500;">{name}</span>
            <span class="score-value {css_cls}" style="font-size: 0.9rem;">{score:.0f}</span>
            <span class="verdict-badge {css_cls}" style="font-size: 0.65rem;">{_esc(verdict)}</span>
            <span style="color: var(--text-muted); font-size: 0.75rem;">{_esc(date_display)}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# USER STATS WIDGET (GAMIFICATION)
# -----------------------------------------------------------------------

def render_user_stats(stats: dict):
    """Render a gamified stats dashboard for the user."""
    total = stats.get("total_scans", 0)
    avg = stats.get("avg_score", 0.0)
    streak = stats.get("active_days", 0)
    
    # Simple logic to determine a rank
    if total > 100:
        rank = "Nutrition Expert 🏆"
        color = "#fbbf24"
    elif total > 20:
        rank = "Health Enthusiast 🌟"
        color = "#60a5fa"
    elif total > 5:
        rank = "Label Reader 🔍"
        color = "#34d399"
    else:
        rank = "Novice Explorer 🥚"
        color = "#94a3b8"

    st.markdown(f"""
    <div class="glass-card animate-in" style="margin-bottom: 20px;">
        <div class="section-header" style="text-align: center; margin-bottom: 15px; color: {color};">
            Rank: {rank}
        </div>
        <div class="stats-row" style="gap: 20px;">
            <div class="stat-item" style="text-align: center;">
                <div class="stat-number" style="font-size: 1.5rem;">{total}</div>
                <div class="stat-label">Total Scans</div>
            </div>
            <div class="stat-item" style="text-align: center;">
                <div class="stat-number" style="font-size: 1.5rem;">{avg:.1f}</div>
                <div class="stat-label">Avg Health Score</div>
            </div>
            <div class="stat-item" style="text-align: center;">
                <div class="stat-number" style="font-size: 1.5rem;">{streak} 🔥</div>
                <div class="stat-label">Active Days</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# FAVOURITES TABLE
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
        try:
            score = float(row.get("score", 0) or 0)
        except (TypeError, ValueError):
            score = 0.0
        name = _esc(row.get('name', 'Unknown'))
        brand = _esc(row.get('brand', ''))
        added_at = row.get('added_at') or ''
        date_display = str(added_at)[:10] if added_at else '—'

        st.markdown(f"""
        <div class="history-row">
            <span style="color: var(--text-primary); font-weight: 500;">{name}</span>
            <span style="color: var(--text-secondary);">{brand}</span>
            <span style="font-weight: 700; font-size: 0.9rem;">{score:.0f}</span>
            <span style="color: var(--text-muted); font-size: 0.75rem;">{_esc(date_display)}</span>
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------
# AUTH SIDEBAR
# -----------------------------------------------------------------------

_SOCIAL_BUTTONS_CSS = """
<style>
.social-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    width: 100%;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 6px;
    text-decoration: none;
    transition: all 0.15s ease;
}
.social-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.social-btn-google {
    background: #ffffff;
    color: #3c4043;
}
.social-btn-google:hover { background: #f8f9fa; }
.social-btn-apple {
    background: #000000;
    color: #ffffff;
}
.social-btn-apple:hover { background: #1a1a1a; }
.social-btn-github {
    background: #24292f;
    color: #ffffff;
}
.social-btn-github:hover { background: #2f363d; }
.social-btn svg { flex-shrink: 0; }
.divider-text {
    display: flex;
    align-items: center;
    color: var(--text-muted, #6b6b8d);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 10px 0;
}
.divider-text::before, .divider-text::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.08);
}
.divider-text::before { margin-right: 10px; }
.divider-text::after { margin-left: 10px; }
</style>
"""

_GOOGLE_ICON = '<svg width="18" height="18" viewBox="0 0 24 24"><path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z"/><path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/><path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/><path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/></svg>'

_APPLE_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="white"><path d="M17.05 20.28c-.98.95-2.05.88-3.08.4-1.09-.5-2.08-.48-3.24 0-1.44.62-2.2.44-3.06-.4C2.79 15.25 3.51 7.59 9.05 7.31c1.35.07 2.29.74 3.08.8 1.18-.24 2.31-.93 3.57-.84 1.51.12 2.65.72 3.4 1.8-3.12 1.87-2.38 5.98.48 7.13-.57 1.5-1.31 2.99-2.54 4.09zM12.03 7.25c-.15-2.23 1.66-4.07 3.74-4.25.29 2.58-2.34 4.5-3.74 4.25z"/></svg>'

_GITHUB_ICON = '<svg width="18" height="18" viewBox="0 0 24 24" fill="white"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.43 9.8 8.21 11.39.6.11.79-.26.79-.58 0-.29-.01-1.24-.02-2.25-3.34.73-4.04-1.42-4.04-1.42-.55-1.39-1.34-1.76-1.34-1.76-1.08-.74.08-.73.08-.73 1.2.09 1.84 1.24 1.84 1.24 1.07 1.84 2.81 1.31 3.49 1 .11-.78.42-1.31.76-1.61-2.67-.3-5.47-1.33-5.47-5.93 0-1.31.47-2.38 1.24-3.22-.14-.3-.54-1.52.1-3.18 0 0 1-.32 3.3 1.23a11.5 11.5 0 016.02 0c2.28-1.55 3.29-1.23 3.29-1.23.64 1.66.24 2.88.12 3.18.77.84 1.23 1.91 1.23 3.22 0 4.61-2.81 5.63-5.48 5.92.42.36.81 1.1.81 2.22 0 1.61-.01 2.9-.01 3.29 0 .32.18.7.8.58C20.57 21.8 24 17.31 24 12c0-6.63-5.37-12-12-12z"/></svg>'


def render_auth_sidebar(
    on_login, on_signup, on_logout,
    logged_in: bool, email: str = "",
    google_url: str | None = None,
    github_url: str | None = None,
):
    """
    Render login/signup forms in the sidebar with social login options.

    Args:
        on_login:    callable(email, password) -> result dict
        on_signup:   callable(email, password) -> result dict
        on_logout:   callable() -> None
        logged_in:   whether user is currently authenticated
        email:       current user's email if logged in
        google_url:  OAuth authorization URL for Google (None if not configured)
        github_url:  OAuth authorization URL for GitHub (None if not configured)
    """
    with st.sidebar:
        # Premium Logo / Header for the vacant upper part
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: center; gap: 14px; margin-bottom: 28px; margin-top: 12px;">
                <!-- Logo Icon -->
                <div style="position: relative; width: 56px; height: 56px; border-radius: 14px; background: #0b0d1a; box-shadow: -6px 0 20px -4px rgba(52,211,153,0.5), 6px 0 20px -4px rgba(129,140,248,0.5), inset 0 0 0 1px rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <svg width="34" height="34" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <!-- Barcode lines -->
                        <rect x="6" y="10" width="3" height="28" rx="1.5" fill="#34d399"/>
                        <rect x="13" y="6"  width="3" height="36" rx="1.5" fill="#2dd4bf"/>
                        <rect x="20" y="10" width="3" height="28" rx="1.5" fill="#38bdf8"/>
                        <rect x="27" y="14" width="2" height="20" rx="1" fill="#60a5fa"/>
                        <!-- Leaf -->
                        <path d="M30 34C30 34 26 16 42 12C42 12 48 26 38 36C33 40 30 34 30 34Z" fill="url(#leafGrad)"/>
                        <path d="M31 33L40 18" stroke="#0b0d1a" stroke-width="1.5" stroke-linecap="round"/>
                        <defs>
                            <linearGradient id="leafGrad" x1="30" y1="34" x2="42" y2="12" gradientUnits="userSpaceOnUse">
                                <stop stop-color="#818cf8"/>
                                <stop offset="1" stop-color="#34d399"/>
                            </linearGradient>
                        </defs>
                    </svg>
                </div>
                <!-- Text -->
                <div style="display: flex; flex-direction: column;">
                    <div style="display: flex; align-items: center; line-height: 1;">
                        <span style="font-size: 2.1rem; font-weight: 500; font-family: 'Inter', sans-serif; letter-spacing: -0.5px; background: linear-gradient(90deg, #34d399, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Shelf</span>
                        <span style="font-size: 2.1rem; font-weight: 300; font-family: 'Inter', sans-serif; color: #f1f5f9; margin-left: 2px;">Scan</span>
                    </div>
                    <div style="font-size: 0.65rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600; margin-top: 6px;">Health Intelligence</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        if logged_in:
            st.markdown(f"""
            <div class="auth-header">{_esc(email)}</div>
            """, unsafe_allow_html=True)

            if st.button("Logout", use_container_width=True):
                on_logout()
                st.rerun()
        else:
            # Social login buttons CSS
            st.markdown(_SOCIAL_BUTTONS_CSS, unsafe_allow_html=True)

            # Google — real redirect or config hint
            if google_url:
                st.link_button(
                    "Continue with Google",
                    url=google_url,
                    use_container_width=True,
                )
            else:
                if st.button("Continue with Google", use_container_width=True, key="btn_google", disabled=True):
                    pass
                st.caption("Set GOOGLE_CLIENT_ID in .env")

            # GitHub — real redirect or config hint
            if github_url:
                st.link_button(
                    "Continue with GitHub",
                    url=github_url,
                    use_container_width=True,
                )
            else:
                if st.button("Continue with GitHub", use_container_width=True, key="btn_github", disabled=True):
                    pass
                st.caption("Set GITHUB_CLIENT_ID in .env")

            # Special spacer / aesthetic filler replacing Apple login
            st.markdown(
                """
                <div style="height: 1px; width: 40%; margin: 24px auto; background: linear-gradient(90deg, transparent, rgba(129,140,248,0.5), transparent); box-shadow: 0 0 12px rgba(129,140,248,0.8);"></div>
                """, 
                unsafe_allow_html=True
            )

            # Divider
            st.markdown('<div class="divider-text">or continue with email</div>', unsafe_allow_html=True)

            # Email/password form
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


