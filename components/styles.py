"""
components/styles.py
====================
CSS design system for ShelfScan, injected via st.markdown(unsafe_allow_html=True).

Design language:
  - Dark gradient background (deep navy → charcoal)
  - Glassmorphism cards with backdrop-filter blur
  - Animated score bars with colour transitions
  - Inter font from Google Fonts
  - Responsive layout that works on mobile
"""

import streamlit as st


def inject_css():
    """Inject the full ShelfScan CSS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<style>
/* ---------------------------------------------------------------
   FONTS
   --------------------------------------------------------------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------------------------------------------------------------
   ROOT VARIABLES
   --------------------------------------------------------------- */
:root {
    --bg-primary:     #0f1117;
    --bg-secondary:   #1a1d29;
    --bg-card:        rgba(255, 255, 255, 0.04);
    --bg-card-hover:  rgba(255, 255, 255, 0.07);
    --border-subtle:  rgba(255, 255, 255, 0.08);
    --text-primary:   #e8eaed;
    --text-secondary: #9aa0a6;
    --text-muted:     #6b7280;
    --accent-green:   #34d399;
    --accent-yellow:  #fbbf24;
    --accent-orange:  #f97316;
    --accent-red:     #ef4444;
    --accent-blue:    #60a5fa;
    --accent-purple:  #a78bfa;
    --glass-bg:       rgba(255, 255, 255, 0.05);
    --glass-border:   rgba(255, 255, 255, 0.1);
    --radius:         12px;
    --radius-lg:      16px;
    --shadow:         0 4px 24px rgba(0, 0, 0, 0.3);
}

/* ---------------------------------------------------------------
   GLOBAL
   --------------------------------------------------------------- */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ---------------------------------------------------------------
   HERO
   --------------------------------------------------------------- */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    margin-bottom: 1.5rem;
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34d399, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 300;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ---------------------------------------------------------------
   GLASS CARD
   --------------------------------------------------------------- */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
}

/* ---------------------------------------------------------------
   SCORE BAR
   --------------------------------------------------------------- */
.score-container {
    margin: 1rem 0;
}

.score-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.5rem;
}

.score-value {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.score-value.great  { color: var(--accent-green); }
.score-value.ok     { color: var(--accent-yellow); }
.score-value.caution { color: var(--accent-orange); }
.score-value.avoid  { color: var(--accent-red); }

.score-bar-track {
    width: 100%;
    height: 10px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 5px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
    background: linear-gradient(90deg, var(--accent-red), var(--accent-orange),
                var(--accent-yellow), var(--accent-green));
    background-size: 300% 100%;
}

.score-bar-fill.great  { background-position: 100% 0; }
.score-bar-fill.ok     { background-position: 66% 0; }
.score-bar-fill.caution { background-position: 33% 0; }
.score-bar-fill.avoid  { background-position: 0% 0; }

/* ---------------------------------------------------------------
   VERDICT BADGE
   --------------------------------------------------------------- */
.verdict-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}

.verdict-badge.great   { background: rgba(52, 211, 153, 0.15); color: var(--accent-green); border: 1px solid rgba(52, 211, 153, 0.3); }
.verdict-badge.ok      { background: rgba(251, 191, 36, 0.15); color: var(--accent-yellow); border: 1px solid rgba(251, 191, 36, 0.3); }
.verdict-badge.caution { background: rgba(249, 115, 22, 0.15); color: var(--accent-orange); border: 1px solid rgba(249, 115, 22, 0.3); }
.verdict-badge.avoid   { background: rgba(239, 68, 68, 0.15); color: var(--accent-red); border: 1px solid rgba(239, 68, 68, 0.3); }

/* ---------------------------------------------------------------
   NUTRIENT GRID
   --------------------------------------------------------------- */
.nutrient-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 0.6rem;
    margin: 1rem 0;
}

.nutrient-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius);
    padding: 0.7rem;
    text-align: center;
    font-size: 0.82rem;
}

.nutrient-label {
    color: var(--text-muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.2rem;
}

.nutrient-value {
    font-weight: 600;
    font-size: 0.9rem;
}

/* ---------------------------------------------------------------
   ADJUSTMENTS TRAIL
   --------------------------------------------------------------- */
.adjustment-item {
    display: flex;
    align-items: center;
    padding: 0.35rem 0;
    font-size: 0.82rem;
    color: var(--text-secondary);
    border-bottom: 1px solid rgba(255, 255, 255, 0.03);
}

.adjustment-item:last-child { border-bottom: none; }

.adj-positive { color: var(--accent-green); }
.adj-negative { color: var(--accent-red); }

/* ---------------------------------------------------------------
   PRODUCT INFO
   --------------------------------------------------------------- */
.product-name {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.15rem;
}

.product-brand {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
}

.product-meta {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
}

.meta-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-subtle);
    border-radius: 6px;
    padding: 0.25rem 0.6rem;
    font-size: 0.75rem;
    color: var(--text-secondary);
}

/* ---------------------------------------------------------------
   COMPARE
   --------------------------------------------------------------- */
.compare-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

@media (max-width: 768px) {
    .compare-grid {
        grid-template-columns: 1fr;
    }
    .hero-title { font-size: 2rem; }
}

/* ---------------------------------------------------------------
   SECTION HEADERS
   --------------------------------------------------------------- */
.section-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ---------------------------------------------------------------
   AUTH
   --------------------------------------------------------------- */
.auth-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.8rem;
}

/* ---------------------------------------------------------------
   HISTORY TABLE
   --------------------------------------------------------------- */
.history-row {
    display: grid;
    grid-template-columns: 2fr 1fr 80px 80px;
    gap: 0.5rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
    font-size: 0.82rem;
    align-items: center;
}

.history-row:first-child {
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
}

/* ---------------------------------------------------------------
   ANIMATIONS
   --------------------------------------------------------------- */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.animate-in {
    animation: fadeIn 0.5s ease forwards;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.6; }
}

.scanning-pulse {
    animation: pulse 1.5s ease infinite;
    color: var(--accent-blue);
    font-weight: 500;
}
</style>
"""
