"""
components/styles.py
====================
ShelfScan premium dual-theme design system v3.

Design philosophy:
  Light: Warm off-white (#f5f3ef) with subtle ambient color blobs,
         frosted-glass cards, teal-green brand accent
  Dark:  Deep indigo-black with glowing ambient orbs,
         glassmorphic cards with luminous borders
  Both:  Subtle background texture, floating gradient orbs,
         premium shadows with color tints, rich micro-interactions
"""

import streamlit as st


def inject_css():
    """Inject the full design system + theme toggle JS into the Streamlit page."""
    st.markdown(_CSS, unsafe_allow_html=True)


_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

<style>
/* ===================================================================
   LIGHT THEME — DISABLED (commented out, not deleted)
   To re-enable: uncomment this block and comment out the dark :root below
   ===================================================================
:root, [data-theme="light"] {
    --bg-primary:     #f5f3ef;
    --bg-secondary:   #faf9f7;
    --bg-card:        rgba(255, 255, 255, 0.72);
    --bg-card-hover:  rgba(255, 255, 255, 0.92);
    --bg-card-border: rgba(16, 185, 129, 0.10);
    --bg-input:       rgba(255, 255, 255, 0.8);
    --bg-input-focus: rgba(255, 255, 255, 0.95);

    --text-primary:   #1a1a2e;
    --text-secondary: #2d2d44;
    --text-muted:     #555566;

    --accent-primary: #0d9488;
    --accent-bright:  #10b981;
    --accent-cta:     #f97316;
    --accent-blue:    #3b82f6;
    --accent-red:     #ef4444;
    --accent-purple:  #8b5cf6;
    --accent-amber:   #f59e0b;

    --gradient-hero:   linear-gradient(135deg, #0d9488 0%, #10b981 35%, #3b82f6 100%);
    --gradient-cta:    linear-gradient(135deg, #f97316, #f59e0b);
    --gradient-green:  linear-gradient(135deg, #10b981, #059669);
    --gradient-blue:   linear-gradient(135deg, #3b82f6, #2563eb);
    --gradient-amber:  linear-gradient(135deg, #f59e0b, #d97706);
    --gradient-red:    linear-gradient(135deg, #ef4444, #dc2626);
    --gradient-purple: linear-gradient(135deg, #8b5cf6, #7c3aed);

    --header-bg:      rgba(245, 243, 239, 0.85);
    --sidebar-bg:     #faf9f7;
    --sidebar-border: rgba(0, 0, 0, 0.06);
    --overlay-soft:   rgba(13, 148, 136, 0.04);
    --overlay-hover:  rgba(13, 148, 136, 0.08);
    --border-subtle:  rgba(0, 0, 0, 0.06);
    --border-medium:  rgba(0, 0, 0, 0.1);
    --border-accent:  rgba(13, 148, 136, 0.18);
    --scrollbar-thumb: rgba(0, 0, 0, 0.1);
    --scrollbar-hover: rgba(0, 0, 0, 0.18);
    --shimmer-a:      rgba(0, 0, 0, 0.03);
    --shimmer-b:      rgba(0, 0, 0, 0.06);

    --radius-sm: 10px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 28px;

    --shadow-sm:  0 1px 3px rgba(0,0,0,0.03), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md:  0 4px 16px rgba(0,0,0,0.05), 0 1px 3px rgba(0,0,0,0.04);
    --shadow-lg:  0 12px 48px rgba(0,0,0,0.07), 0 4px 12px rgba(0,0,0,0.03);
    --shadow-glow-green: 0 4px 24px rgba(13,148,136,0.12);
    --shadow-glow-red:   0 4px 20px rgba(239,68,68,0.1);
    --shadow-card: 0 1px 4px rgba(0,0,0,0.03), 0 4px 16px rgba(13,148,136,0.04);
    --shadow-card-hover: 0 8px 32px rgba(13,148,136,0.1), 0 4px 12px rgba(0,0,0,0.06);

    --transition-fast: 0.18s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-med:  0.3s cubic-bezier(0.4, 0, 0.2, 1);
    --transition-slow: 0.5s cubic-bezier(0.4, 0, 0.2, 1);

    --card-topline: linear-gradient(90deg, transparent, rgba(13,148,136,0.15), transparent);
    --ring-bg: rgba(0,0,0,0.04);
    --tab-active-bg: var(--gradient-hero);
    --btn-secondary: rgba(255,255,255,0.7);
    --btn-secondary-hover: rgba(255,255,255,0.95);
    --divider-color: rgba(0,0,0,0.06);

    --orb-1: radial-gradient(600px circle at 15% 25%, rgba(13,148,136,0.08), transparent 70%);
    --orb-2: radial-gradient(500px circle at 85% 15%, rgba(59,130,246,0.06), transparent 70%);
    --orb-3: radial-gradient(400px circle at 50% 85%, rgba(249,115,22,0.04), transparent 70%);

    --feature-icon-bg: linear-gradient(135deg, rgba(13,148,136,0.12), rgba(16,185,129,0.08));
    --feature-icon-color: #0d9488;
    --stat-number-color: #0d9488;
}
   END LIGHT THEME DISABLED */

/* ===================================================================
   DARK THEME — deep indigo-black with luminous orbs (NOW DEFAULT)
   =================================================================== */
:root, [data-theme="dark"] {
    --bg-primary:     #0b0d1a;
    --bg-secondary:   #111427;
    --bg-card:        rgba(255, 255, 255, 0.03);
    --bg-card-hover:  rgba(255, 255, 255, 0.06);
    --bg-card-border: rgba(52, 211, 153, 0.08);
    --bg-input:       rgba(255, 255, 255, 0.05);
    --bg-input-focus: rgba(255, 255, 255, 0.08);

    --text-primary:   #eef2f7;
    --text-secondary: #9ba3b5;
    --text-muted:     #5e6478;

    --accent-primary: #34d399;
    --accent-bright:  #6ee7b7;
    --accent-cta:     #fb923c;
    --accent-blue:    #60a5fa;
    --accent-red:     #f87171;
    --accent-purple:  #a78bfa;
    --accent-amber:   #fbbf24;

    --gradient-hero:   linear-gradient(135deg, #34d399 0%, #2dd4bf 35%, #60a5fa 100%);
    --gradient-cta:    linear-gradient(135deg, #fb923c, #fbbf24);
    --gradient-green:  linear-gradient(135deg, #34d399, #10b981);
    --gradient-blue:   linear-gradient(135deg, #60a5fa, #3b82f6);
    --gradient-amber:  linear-gradient(135deg, #fbbf24, #f59e0b);
    --gradient-red:    linear-gradient(135deg, #f87171, #ef4444);
    --gradient-purple: linear-gradient(135deg, #a78bfa, #8b5cf6);

    --header-bg:      rgba(11, 13, 26, 0.85);
    --sidebar-bg:     #111427;
    --sidebar-border: rgba(255, 255, 255, 0.04);
    --overlay-soft:   rgba(52, 211, 153, 0.04);
    --overlay-hover:  rgba(52, 211, 153, 0.08);
    --border-subtle:  rgba(255, 255, 255, 0.05);
    --border-medium:  rgba(255, 255, 255, 0.08);
    --border-accent:  rgba(52, 211, 153, 0.15);
    --scrollbar-thumb: rgba(255, 255, 255, 0.08);
    --scrollbar-hover: rgba(255, 255, 255, 0.15);
    --shimmer-a:      rgba(255, 255, 255, 0.03);
    --shimmer-b:      rgba(255, 255, 255, 0.06);

    --shadow-sm:  0 2px 8px rgba(0,0,0,0.3);
    --shadow-md:  0 4px 20px rgba(0,0,0,0.4);
    --shadow-lg:  0 12px 48px rgba(0,0,0,0.5);
    --shadow-glow-green: 0 4px 24px rgba(52, 211, 153, 0.15), 0 0 0 1px rgba(52, 211, 153, 0.3);
    --shadow-glow-red:   0 4px 24px rgba(248, 113, 113, 0.12);
    /* Spotlight card shadows */
    --shadow-card: inset 0 1px 0 rgba(255,255,255,0.08), 0 2px 8px rgba(0,0,0,0.4), 0 0 1px rgba(52,211,153,0.06);
    --shadow-card-hover: inset 0 1px 0 rgba(255,255,255,0.12), 0 8px 32px rgba(52,211,153,0.12), 0 0 1px rgba(52,211,153,0.2);
    --shadow-input-recessed: inset 0 2px 4px rgba(0,0,0,0.3), inset 0 0 0 1px rgba(255,255,255,0.04);

    --card-topline: linear-gradient(90deg, transparent, rgba(52,211,153,0.2), transparent);
    --ring-bg: rgba(255, 255, 255, 0.05);
    --tab-active-bg: var(--gradient-hero);
    --btn-secondary: rgba(255, 255, 255, 0.05);
    --btn-secondary-hover: rgba(255, 255, 255, 0.08);
    --divider-color: rgba(255, 255, 255, 0.05);

    --orb-1: radial-gradient(600px circle at 15% 25%, rgba(52,211,153,0.07), transparent 70%);
    --orb-2: radial-gradient(500px circle at 85% 15%, rgba(96,165,250,0.05), transparent 70%);
    --orb-3: radial-gradient(400px circle at 50% 85%, rgba(251,146,60,0.04), transparent 70%);

    --feature-icon-bg: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(45,212,191,0.10));
    --feature-icon-color: #34d399;
    --stat-number-color: #34d399;
}

/* System preference auto-detection — DISABLED (dark mode is now always-on)
@media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
        --bg-primary:#0b0d1a;--bg-secondary:#111427;
        --bg-card:rgba(255,255,255,0.03);--bg-card-hover:rgba(255,255,255,0.06);
        --bg-card-border:rgba(52,211,153,0.08);
        --bg-input:rgba(255,255,255,0.05);--bg-input-focus:rgba(255,255,255,0.08);
        --text-primary:#eef2f7;--text-secondary:#9ba3b5;--text-muted:#5e6478;
        --accent-primary:#34d399;--accent-bright:#6ee7b7;--accent-cta:#fb923c;
        --accent-blue:#60a5fa;--accent-red:#f87171;--accent-purple:#a78bfa;--accent-amber:#fbbf24;
        --gradient-hero:linear-gradient(135deg,#34d399 0%,#2dd4bf 35%,#60a5fa 100%);
        --gradient-cta:linear-gradient(135deg,#fb923c,#fbbf24);
        --gradient-green:linear-gradient(135deg,#34d399,#10b981);
        --gradient-blue:linear-gradient(135deg,#60a5fa,#3b82f6);
        --gradient-amber:linear-gradient(135deg,#fbbf24,#f59e0b);
        --gradient-red:linear-gradient(135deg,#f87171,#ef4444);
        --gradient-purple:linear-gradient(135deg,#a78bfa,#8b5cf6);
        --header-bg:rgba(11,13,26,0.85);--sidebar-bg:#111427;--sidebar-border:rgba(255,255,255,0.04);
        --overlay-soft:rgba(52,211,153,0.04);--overlay-hover:rgba(52,211,153,0.08);
        --border-subtle:rgba(255,255,255,0.05);--border-medium:rgba(255,255,255,0.08);
        --border-accent:rgba(52,211,153,0.15);
        --scrollbar-thumb:rgba(255,255,255,0.08);--scrollbar-hover:rgba(255,255,255,0.15);
        --shimmer-a:rgba(255,255,255,0.03);--shimmer-b:rgba(255,255,255,0.06);
        --shadow-sm:0 2px 8px rgba(0,0,0,0.3);--shadow-md:0 4px 20px rgba(0,0,0,0.4);
        --shadow-lg:0 12px 48px rgba(0,0,0,0.5);
        --shadow-glow-green:0 4px 24px rgba(52,211,153,0.15);--shadow-glow-red:0 4px 24px rgba(248,113,113,0.12);
        --shadow-card:0 2px 8px rgba(0,0,0,0.2),0 0 1px rgba(52,211,153,0.06);
        --shadow-card-hover:0 8px 32px rgba(52,211,153,0.12),0 0 1px rgba(52,211,153,0.2);
        --card-topline:linear-gradient(90deg,transparent,rgba(52,211,153,0.2),transparent);
        --ring-bg:rgba(255,255,255,0.05);
        --btn-secondary:rgba(255,255,255,0.05);--btn-secondary-hover:rgba(255,255,255,0.08);
        --divider-color:rgba(255,255,255,0.05);
        --orb-1:radial-gradient(600px circle at 15% 25%,rgba(52,211,153,0.07),transparent 70%);
        --orb-2:radial-gradient(500px circle at 85% 15%,rgba(96,165,250,0.05),transparent 70%);
        --orb-3:radial-gradient(400px circle at 50% 85%,rgba(251,146,60,0.04),transparent 70%);
        --feature-icon-bg:linear-gradient(135deg,rgba(52,211,153,0.15),rgba(45,212,191,0.10));
        --feature-icon-color:#34d399;--stat-number-color:#34d399;
    }
}
END auto-detection DISABLED */

/* ===================================================================
   GLOBAL RESETS + AMBIENT BACKGROUND
   =================================================================== */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    transition: background 0.4s ease, color 0.4s ease;
}

/* Ambient gradient orbs — create depth and atmosphere */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: var(--orb-1), var(--orb-2), var(--orb-3);
    pointer-events: none;
    z-index: 0;
}

/* SVG Noise Overlay (Linear-style grain) + Subtle dot-grid */
.stApp::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: 
        url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)' opacity='0.08'/%3E%3C/svg%3E"),
        radial-gradient(circle, var(--border-subtle) 1px, transparent 1px);
    background-size: 200px 200px, 24px 24px;
    background-repeat: repeat;
    opacity: 0.8;
    pointer-events: none;
    z-index: 0;
    mix-blend-mode: overlay;
}

/* ===================================================================
   STREAMLIT OVERRIDES
   =================================================================== */

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    color: var(--text-primary) !important;
    border-right: 1px solid var(--sidebar-border) !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main .block-container,
[data-testid="stMainBlockContainer"] {
    background-color: transparent !important;
    color: var(--text-primary) !important;
}

[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background: var(--header-bg) !important;
    backdrop-filter: blur(12px);
}

/* Inputs */
.stTextInput > div > div,
.stTextArea > div > div,
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div,
[data-testid="stTextInput"] input,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: var(--bg-input) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important;
    border-radius: 100px !important;
    padding: 0 8px !important;
    transition: all var(--transition-fast) !important;
    box-shadow: var(--shadow-input-recessed) !important;
}
[data-baseweb="input"],[data-baseweb="textarea"],[data-baseweb="base-input"] {
    background-color: var(--bg-input) !important;
}
.stApp .stTextInput input:focus,
.stApp .stTextArea textarea:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(13,148,136,0.1) !important;
    background: var(--bg-input-focus) !important;
}

/* Buttons */
.stButton > button {
    background-color: var(--btn-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 100px !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all var(--transition-fast) !important;
    backdrop-filter: blur(8px) !important;
}
.stButton > button:hover {
    background-color: var(--btn-secondary-hover) !important;
    border-color: var(--accent-primary) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button[kind="primary"],
.stButton > button[data-testid="baseButton-primary"] {
    background: var(--gradient-hero) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.2) !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow: var(--shadow-glow-green) !important;
    transform: translateY(-2px) !important;
}

/* Link buttons */
.stLinkButton > a {
    background-color: var(--btn-secondary) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    transition: all var(--transition-fast) !important;
    backdrop-filter: blur(8px) !important;
}
.stLinkButton > a:hover {
    background-color: var(--btn-secondary-hover) !important;
    border-color: var(--accent-primary) !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
.stApp [data-testid="stFileUploader"] {
    border: 2px dashed var(--border-medium) !important;
    border-radius: 100px !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    transition: all var(--transition-med) !important;
    padding: 1.5rem !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}
.stApp [data-testid="stFileUploader"]:hover {
    border-color: var(--accent-primary) !important;
    background: var(--overlay-soft) !important;
    box-shadow: var(--shadow-glow-green) !important;
    transform: scale(1.01) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important; color: var(--text-secondary) !important; border: none !important;
    display: flex !important; flex-direction: column !important; align-items: center !important;
}
[data-testid="stFileUploaderDropzone"] > div:first-child { display: none !important; /* Hide original svg icon */ }
[data-testid="stFileUploaderDropzone"]::before {
    content: "Drop barcode image here or";
    font-size: 0.95rem !important; color: var(--text-primary) !important; font-weight: 600 !important; margin-bottom: 8px !important;
}
[data-testid="stFileUploaderDropzone"] small { display: none !important; }

/* The inner Browse files button */
.stApp [data-testid="stFileUploader"] button {
    background: var(--gradient-hero) !important;
    color: white !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 600 !important;
    padding: 8px 30px !important;
    box-shadow: var(--shadow-glow-green) !important;
    margin-top: 5px !important;
}
.stApp [data-testid="stFileUploader"] button:hover {
    filter: brightness(1.2) !important;
    transform: translateY(-2px) !important;
}

/* Expanders */
[data-testid="stExpander"],[data-testid="stExpander"] > details,[data-testid="stExpander"] summary {
    background-color: var(--bg-card) !important; color: var(--text-primary) !important;
    border-color: var(--border-subtle) !important; backdrop-filter: blur(8px) !important;
}

/* Alerts */
[data-testid="stAlert"],.stAlert {
    background-color: var(--bg-card) !important; color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important; backdrop-filter: blur(8px) !important;
}

[data-testid="stStatusWidget"],[data-testid="stToolbar"] { background-color: transparent !important; }
.stMarkdown,.stMarkdown p,.stApp label { color: var(--text-primary) !important; }
[data-testid="stCameraInput"] > div { background-color: var(--overlay-soft) !important; border-color: var(--border-subtle) !important; }
[data-baseweb="popover"],[data-baseweb="menu"],[data-baseweb="select"] > div {
    background-color: var(--bg-secondary) !important; color: var(--text-primary) !important;
    border-radius: var(--radius-md) !important; backdrop-filter: blur(12px) !important;
}

/* ===================================================================
   RADIO → SEGMENTED TOGGLE
   =================================================================== */
[data-testid="stRadio"] > div {
    flex-direction: row !important; gap: 0 !important;
    background: var(--bg-card) !important; border: 1px solid var(--border-subtle) !important;
    border-radius: 100px !important; padding: 6px !important;
    display: inline-flex !important; width: auto !important;
    backdrop-filter: blur(12px) !important;
}
[data-testid="stRadio"] > div > label {
    display: flex !important; align-items: center !important; justify-content: center !important;
    padding: 8px 20px !important; border-radius: 100px !important; cursor: pointer !important;
    font-size: 0.82rem !important; font-weight: 500 !important; color: var(--text-muted) !important;
    background: transparent !important; transition: all var(--transition-fast) !important;
    border: none !important; margin: 0 !important; white-space: nowrap !important;
    flex: 1 !important;
}
[data-testid="stRadio"] > div > label:hover { color: var(--text-primary) !important; background: var(--overlay-soft) !important; }
[data-testid="stRadio"] > div > label[data-checked="true"],
[data-testid="stRadio"] > div > label:has(input:checked) {
    background: var(--gradient-hero) !important; color: #ffffff !important;
    font-weight: 600 !important; box-shadow: var(--shadow-glow-green) !important;
}
[data-testid="stRadio"] > div > label > div:first-child,
[data-testid="stRadio"] input[type="radio"] { display: none !important; }
[data-testid="stRadio"] > label { display: none !important; }

/* ===================================================================
   TABS
   =================================================================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: var(--bg-card); border-radius: 100px;
    padding: 4px; border: 1px solid var(--border-subtle);
    backdrop-filter: blur(12px);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 100px; color: var(--text-muted) !important;
    font-weight: 500; transition: all var(--transition-fast); padding: 8px 20px;
}
.stTabs [data-baseweb="tab"]:hover { color: var(--text-primary) !important; background: var(--overlay-soft); }
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--tab-active-bg) !important; color: white !important;
    font-weight: 600 !important; box-shadow: var(--shadow-glow-green);
}
.stTabs [data-baseweb="tab-highlight"] { display: none; }

/* ===================================================================
   HERO — premium floating header
   =================================================================== */
.hero-container {
    text-align: center;
    padding: 2rem 1rem 0.5rem;
    position: relative;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    background: var(--gradient-hero);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -1.5px;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 0.92rem;
    color: var(--text-muted);
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ===================================================================
   FEATURE CARDS ROW — fills the empty space
   =================================================================== */
.features-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin: 1.5rem 0 0.5rem;
    padding: 0 0.5rem;
}
.feature-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--bg-card-border);
    border-radius: var(--radius-md);
    padding: 1.2rem 1rem;
    text-align: center;
    transition: all var(--transition-med);
    box-shadow: var(--shadow-card);
    position: relative;
    overflow: hidden;
}
.feature-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--gradient-hero);
    opacity: 0;
    transition: opacity var(--transition-med);
}
.feature-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-card-hover);
    border-color: var(--border-accent);
}
.feature-card:hover::before { opacity: 1; }
.feature-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    background: var(--feature-icon-bg);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 0.6rem;
    color: var(--feature-icon-color);
}
.feature-title {
    font-size: 0.82rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
    letter-spacing: -0.2px;
}
.feature-desc {
    font-size: 0.72rem;
    color: var(--text-muted);
    line-height: 1.4;
}

/* ===================================================================
   STAT COUNTERS — trust indicators
   =================================================================== */
.stats-row {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    padding: 0.8rem 0;
    margin: 0.5rem 0;
}
.stat-item { text-align: center; }
.stat-number {
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--stat-number-color);
    letter-spacing: -0.5px;
}
.stat-label {
    font-size: 0.68rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

/* ===================================================================
   GLASS CARDS
   =================================================================== */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--bg-card-border);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all var(--transition-med);
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-card);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: var(--card-topline);
}
.glass-card:hover {
    background: var(--bg-card-hover);
    transform: translateY(-3px);
    box-shadow: var(--shadow-card-hover);
    border-color: var(--border-accent);
}

/* ===================================================================
   PRODUCT HEADER
   =================================================================== */
.product-name { font-size: 1.5rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.25rem; letter-spacing: -0.3px; }
.product-brand { font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.75rem; }
.product-meta { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 1rem; }
.meta-tag {
    font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;
    padding: 4px 10px; border-radius: 100px;
    background: var(--overlay-soft); border: 1px solid var(--border-accent);
    color: var(--accent-primary); transition: all var(--transition-fast);
}
.meta-tag:hover { background: var(--overlay-hover); transform: scale(1.05); }

/* ===================================================================
   SCORE RING
   =================================================================== */
.score-ring-container { display: flex; align-items: center; gap: 1.5rem; margin: 1rem 0; }
.score-ring { position: relative; width: 110px; height: 110px; flex-shrink: 0; }
.score-ring svg { transform: rotate(-90deg); width: 110px; height: 110px; }
.score-ring-bg { fill: none; stroke: var(--ring-bg); stroke-width: 8; }
.score-ring-fill { fill: none; stroke-width: 8; stroke-linecap: round; transition: stroke-dashoffset 1.2s cubic-bezier(0.4, 0, 0.2, 1); }
.score-ring-fill.great { stroke: url(#grad-green); }
.score-ring-fill.ok { stroke: url(#grad-blue); }
.score-ring-fill.caution { stroke: url(#grad-amber); }
.score-ring-fill.avoid { stroke: url(#grad-red); }
.score-ring-value { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.8rem; font-weight: 800; letter-spacing: -1px; }
.score-ring-value.great { color: var(--accent-bright); }
.score-ring-value.ok { color: var(--accent-blue); }
.score-ring-value.caution { color: var(--accent-amber); }
.score-ring-value.avoid { color: var(--accent-red); }
.score-info { flex: 1; }
.score-info .verdict-label { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.2rem; }
.score-info .verdict-label.great { color: var(--accent-bright); }
.score-info .verdict-label.ok { color: var(--accent-blue); }
.score-info .verdict-label.caution { color: var(--accent-amber); }
.score-info .verdict-label.avoid { color: var(--accent-red); }
.score-info .verdict-sub { font-size: 0.8rem; color: var(--text-muted); }

/* ===================================================================
   SCORE BAR
   =================================================================== */
.score-container { margin: 1rem 0; }
.score-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.score-value { font-size: 1.3rem; font-weight: 700; }
.score-value.great { color: var(--accent-bright); }
.score-value.ok { color: var(--accent-blue); }
.score-value.caution { color: var(--accent-amber); }
.score-value.avoid { color: var(--accent-red); }
.verdict-badge { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; padding: 4px 14px; border-radius: 100px; }
.verdict-badge.great { background: var(--gradient-green); color: white; }
.verdict-badge.ok { background: var(--gradient-blue); color: white; }
.verdict-badge.caution { background: var(--gradient-amber); color: #3d2800; }
.verdict-badge.avoid { background: var(--gradient-red); color: white; }
.score-bar-track { width: 100%; height: 8px; border-radius: 100px; background: var(--ring-bg); overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 100px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
.score-bar-fill.great { background: var(--gradient-green); box-shadow: var(--shadow-glow-green); }
.score-bar-fill.ok { background: var(--gradient-blue); }
.score-bar-fill.caution { background: var(--gradient-amber); }
.score-bar-fill.avoid { background: var(--gradient-red); box-shadow: var(--shadow-glow-red); }

/* ===================================================================
   NUTRIENT GRID
   =================================================================== */
.section-header {
    font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.4px;
    color: var(--accent-primary); margin: 1.2rem 0 0.6rem; padding-bottom: 0.4rem;
    border-bottom: 2px solid var(--border-accent);
}
.nutrient-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(130px, 1fr)); gap: 8px; }
.nutrient-item {
    background: var(--bg-card); backdrop-filter: blur(8px); border-radius: var(--radius-sm);
    padding: 10px 12px; border: 1px solid var(--border-subtle); transition: all var(--transition-fast);
}
.nutrient-item:hover { background: var(--bg-card-hover); transform: translateY(-2px); border-color: var(--border-accent); box-shadow: var(--shadow-sm); }
.nutrient-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.8px; color: var(--text-muted); margin-bottom: 2px; }
.nutrient-value { font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }

/* ===================================================================
   ADJUSTMENTS
   =================================================================== */
.adjustment-item { font-size: 0.78rem; padding: 6px 12px; margin: 3px 0; border-radius: var(--radius-sm); border-left: 3px solid; background: var(--bg-card); backdrop-filter: blur(8px); transition: all var(--transition-fast); }
.adjustment-item:hover { background: var(--bg-card-hover); padding-left: 16px; }
.adj-positive { border-color: var(--accent-primary); color: var(--accent-primary); }
.adj-negative { border-color: var(--accent-red); color: var(--accent-red); }

/* ===================================================================
   UPLOAD ZONE
   =================================================================== */
.dropzone { border: 2px dashed var(--border-medium); border-radius: 100px; padding: 2.5rem 1.5rem; text-align: center; cursor: pointer; transition: all var(--transition-med); background: var(--bg-card); backdrop-filter: blur(12px); position: relative; overflow: hidden; }
.dropzone::before { content: ''; position: absolute; top: 0; left: -100%; width: 200%; height: 100%; background: linear-gradient(90deg, transparent, rgba(13,148,136,0.04), transparent); transition: left 0.6s ease; }
.dropzone:hover { border-color: var(--accent-primary); background: var(--overlay-soft); transform: scale(1.01); box-shadow: var(--shadow-glow-green); }
.dropzone:hover::before { left: 100%; }
.dropzone-icon { font-size: 2.5rem; margin-bottom: 0.5rem; opacity: 0.7; }
.dropzone-text { font-size: 0.9rem; color: var(--text-secondary); margin-bottom: 0.3rem; }
.dropzone-hint { font-size: 0.72rem; color: var(--text-muted); }

/* ===================================================================
   SKELETON
   =================================================================== */
@keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
.skeleton { border-radius: var(--radius-sm); background: linear-gradient(90deg, var(--shimmer-a) 25%, var(--shimmer-b) 50%, var(--shimmer-a) 75%); background-size: 800px 100%; animation: shimmer 1.5s ease-in-out infinite; }
.skeleton-card { height: 280px; border-radius: var(--radius-lg); margin-bottom: 1rem; }
.skeleton-line { height: 14px; margin: 8px 0; width: 70%; } .skeleton-line.short { width: 40%; } .skeleton-line.full { width: 100%; }
.skeleton-circle { width: 110px; height: 110px; border-radius: 50%; }

/* ===================================================================
   HISTORY ROWS
   =================================================================== */
.history-row { display: grid; grid-template-columns: 2fr 0.8fr 1fr 1fr; align-items: center; padding: 10px 14px; border-radius: var(--radius-sm); font-size: 0.82rem; transition: all var(--transition-fast); border-bottom: 1px solid var(--divider-color); }
.history-row:first-child { font-weight: 700; color: var(--text-muted); text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.8px; border-bottom: 2px solid var(--border-accent); margin-bottom: 4px; }
.history-row:not(:first-child):hover { background: var(--overlay-soft); transform: translateX(4px); border-color: var(--border-accent); }

/* Auth */
.auth-header { font-size: 0.9rem; font-weight: 600; color: var(--text-primary); padding: 8px 0; margin-bottom: 4px; }

/* Tooltip */
.tooltip-wrapper { position: relative; display: inline-block; cursor: help; }
.tooltip-wrapper .tooltip-text { visibility: hidden; opacity: 0; position: absolute; bottom: 120%; left: 50%; transform: translateX(-50%) translateY(4px); background: var(--bg-secondary); color: var(--text-secondary); font-size: 0.72rem; padding: 6px 10px; border-radius: var(--radius-sm); border: 1px solid var(--border-medium); white-space: nowrap; box-shadow: var(--shadow-md); transition: var(--transition-fast); z-index: 100; backdrop-filter: blur(12px); }
.tooltip-wrapper:hover .tooltip-text { visibility: visible; opacity: 1; transform: translateX(-50%) translateY(0); }

/* ===================================================================
   PILLS
   =================================================================== */
.pill { display: inline-flex; align-items: center; gap: 4px; font-size: 0.68rem; font-weight: 600; padding: 3px 10px; border-radius: 100px; margin: 2px; transition: all var(--transition-fast); }
.pill:hover { transform: scale(1.08); box-shadow: var(--shadow-sm); }
.pill-green { background: rgba(16,185,129,0.1); color: var(--accent-primary); border: 1px solid rgba(16,185,129,0.18); }
.pill-red { background: rgba(239,68,68,0.1); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.15); }
.pill-amber { background: rgba(245,158,11,0.1); color: var(--accent-amber); border: 1px solid rgba(245,158,11,0.15); }
.pill-blue { background: rgba(59,130,246,0.1); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.15); }
.pill-purple { background: rgba(139,92,246,0.1); color: var(--accent-purple); border: 1px solid rgba(139,92,246,0.15); }

/* ===================================================================
   ANIMATIONS
   =================================================================== */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeInScale { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.6; } }
@keyframes float { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
.animate-in { animation: fadeInUp 0.5s ease both; }
.animate-scale { animation: fadeInScale 0.4s ease both; }
.animate-float { animation: float 3s ease-in-out infinite; }
.glass-card:nth-child(1) { animation-delay: 0.05s; }
.glass-card:nth-child(2) { animation-delay: 0.1s; }
.glass-card:nth-child(3) { animation-delay: 0.15s; }
.glass-card:nth-child(4) { animation-delay: 0.2s; }

/* ===================================================================
   RESPONSIVE
   =================================================================== */
@media (max-width: 768px) {
    .hero-title { font-size: 2rem; }
    .features-row { grid-template-columns: 1fr; gap: 10px; }
    .stats-row { gap: 1.5rem; }
    .nutrient-grid { grid-template-columns: repeat(2, 1fr); }
    .history-row { grid-template-columns: 1.5fr 0.6fr 0.8fr 0.8fr; font-size: 0.75rem; }
    .score-ring { width: 80px; height: 80px; }
    .score-ring svg { width: 80px; height: 80px; }
    .score-ring-value { font-size: 1.3rem; }
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--scrollbar-thumb); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: var(--scrollbar-hover); }
</style>

<!-- SVG gradient defs -->
<svg width="0" height="0" style="position:absolute">
  <defs>
    <linearGradient id="grad-green" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" style="stop-color:#10b981"/><stop offset="100%" style="stop-color:#059669"/></linearGradient>
    <linearGradient id="grad-blue" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" style="stop-color:#3b82f6"/><stop offset="100%" style="stop-color:#2563eb"/></linearGradient>
    <linearGradient id="grad-amber" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" style="stop-color:#f59e0b"/><stop offset="100%" style="stop-color:#d97706"/></linearGradient>
    <linearGradient id="grad-red" x1="0%" y1="0%" x2="100%" y2="0%"><stop offset="0%" style="stop-color:#ef4444"/><stop offset="100%" style="stop-color:#dc2626"/></linearGradient>
  </defs>
</svg>

<!-- Theme toggle JS -->
<script>
(function(){
    function gp(){var s=localStorage.getItem('shelfscan-theme');if(s)return s;return window.matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
    function ap(t){document.documentElement.setAttribute('data-theme',t);localStorage.setItem('shelfscan-theme',t);}
    ap(gp());
    window.matchMedia('(prefers-color-scheme:dark)').addEventListener('change',function(e){if(!localStorage.getItem('shelfscan-theme'))ap(e.matches?'dark':'light');});
    window.toggleShelfScanTheme=function(){var c=document.documentElement.getAttribute('data-theme')||gp();ap(c==='dark'?'light':'dark');};
})();
</script>
"""
